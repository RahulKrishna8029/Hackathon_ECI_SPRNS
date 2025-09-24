# sprns_ui.py
import sys
import os
import streamlit as st

# Add the parent directory to the path to import retrieval modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retrieval_service import RetrievalService

st.set_page_config(page_title="SPRNS Chat UI", layout="centered")
st.title("🤖 SPRNS - Smart Retrieval & Knowledge System")

# Initialize retrieval service
@st.cache_resource
def init_retrieval_service():
    """Initialize the retrieval service with caching."""
    try:
        return RetrievalService()
    except Exception as e:
        st.error(f"Failed to initialize retrieval service: {e}")
        return None

retrieval_service = init_retrieval_service()

# Session-state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm SPRNS, your Smart Retrieval & Knowledge System. Ask me anything and I'll search through the knowledge base to provide you with accurate answers."}
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
        
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(msg["sources"], 1):
                    st.write(f"**Source {i}: {source.get('metadata', {}).get('title', 'Untitled')}**")
                    st.write(f"*Relevance: {source.get('relevance_score', 0):.2f}*")
                    st.write(source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', ''))
                    st.divider()

# Input box
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Save and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Process query using retrieval service
    with st.spinner("Searching knowledge base and generating answer..."):
        if retrieval_service:
            result = retrieval_service.process_query(user_input)
            reply = result.get('answer', 'Sorry, I could not generate an answer.')
            sources = result.get('sources', [])
            status = result.get('status', 'unknown')
            
            # Add status indicator
            if status == 'error':
                st.error("⚠️ An error occurred while processing your query.")
            elif status == 'no_results':
                st.warning("ℹ️ No relevant documents found in the knowledge base.")
        else:
            reply = "Sorry, the retrieval service is not available. Please check the system configuration."
            sources = []
            status = 'service_unavailable'

    # Save assistant message with sources
    assistant_message = {
        "role": "assistant", 
        "content": reply,
        "sources": sources,
        "status": status
    }
    st.session_state["messages"].append(assistant_message)
    
    # Display assistant message
    st.chat_message("assistant").write(reply)
    
    # Display sources
    if sources:
        with st.expander("📚 Sources", expanded=False):
            for i, source in enumerate(sources, 1):
                st.write(f"**Source {i}: {source.get('metadata', {}).get('title', 'Untitled')}**")
                st.write(f"*Relevance: {source.get('relevance_score', 0):.2f}*")
                st.write(source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', ''))
                st.divider()

# Sidebar with system information
with st.sidebar:
    st.header("System Status")
    
    if retrieval_service:
        st.success("✅ Retrieval Service: Online")
        if retrieval_service.neo4j_connector:
            st.success("✅ Neo4j Database: Connected")
        else:
            st.warning("⚠️ Neo4j Database: Using Mock Data")
    else:
        st.error("❌ Retrieval Service: Offline")
    
    st.header("About SPRNS")
    st.write("""
    SPRNS (Smart Retrieval & Knowledge System) uses advanced RAG (Retrieval-Augmented Generation) 
    to provide accurate answers by searching through a knowledge base and generating contextual responses.
    
    **Features:**
    - Semantic search through documents
    - Context-aware answer generation
    - Source attribution and relevance scoring
    - Graph-based knowledge representation
    """)
    
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm SPRNS, your Smart Retrieval & Knowledge System. Ask me anything and I'll search through the knowledge base to provide you with accurate answers."}
        ]
        st.rerun()

