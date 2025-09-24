# demo_ui.py - Lightweight demo version of SPRNS UI
import sys
import os
import streamlit as st

# Add the parent directory to the path to import demo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.service import DemoRetrievalService

st.set_page_config(page_title="SPRNS Demo", layout="centered")
st.title("🤖 SPRNS Demo - Smart Retrieval & Knowledge System")

# Initialize demo retrieval service
@st.cache_resource
def init_demo_service():
    """Initialize the demo retrieval service with caching."""
    try:
        return DemoRetrievalService()
    except Exception as e:
        st.error(f"Failed to initialize demo service: {e}")
        return None

# Clear cache button in sidebar for testing
if st.sidebar.button("🔄 Refresh Service"):
    st.cache_resource.clear()
    st.rerun()

demo_service = init_demo_service()

# Session-state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm SPRNS Demo, your Smart Retrieval & Knowledge System. I have knowledge about machine learning, data science, programming, and databases. Ask me anything!"}
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
                    st.write(f"*Category: {source.get('metadata', {}).get('category', 'Unknown')}*")
                    st.write(f"*Relevance: {source.get('relevance_score', 0):.2f}*")
                    st.write(source.get('content', '')[:300] + "..." if len(source.get('content', '')) > 300 else source.get('content', ''))
                    st.divider()

# Input box
user_input = st.chat_input("Ask me about AI, ML, data science, or programming...")
if user_input:
    # Save and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Process query using demo service
    with st.spinner("Searching knowledge base and generating answer..."):
        if demo_service:
            result = demo_service.process_query(user_input)
            reply = result.get('answer', 'Sorry, I could not generate an answer.')
            sources = result.get('sources', [])
            status = result.get('status', 'unknown')
            
            # Add status indicator
            if status == 'error':
                st.error("⚠️ An error occurred while processing your query.")
            elif status == 'no_results':
                st.warning("ℹ️ No relevant documents found in the knowledge base.")
        else:
            reply = "Sorry, the demo service is not available."
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
                st.write(f"*Category: {source.get('metadata', {}).get('category', 'Unknown')}*")
                st.write(f"*Relevance: {source.get('relevance_score', 0):.2f}*")
                st.write(source.get('content', '')[:300] + "..." if len(source.get('content', '')) > 300 else source.get('content', ''))
                st.divider()

# Sidebar with system information
with st.sidebar:
    st.header("Demo System Status")
    
    if demo_service:
        st.success("✅ Demo Service: Online")
        st.info("📝 Using simulated knowledge base")
    else:
        st.error("❌ Demo Service: Offline")
    
    st.header("Knowledge Base Topics")
    st.write("""
    The demo knowledge base contains information about:
    
    - **Machine Learning**: Algorithms, patterns, predictions
    - **Deep Learning**: Neural networks, computer vision
    - **Natural Language Processing**: Text analysis, linguistics
    - **Data Science**: Statistics, analysis, visualization
    - **Python Programming**: Development, automation
    - **Database Systems**: SQL, data storage
    """)
    
    st.header("Sample Questions")
    sample_questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is natural language processing?",
        "Tell me about data science",
        "What is Python used for?",
        "How do databases work?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            # Add the question to the chat
            st.session_state["messages"].append({"role": "user", "content": question})
            st.rerun()
    
    st.header("About SPRNS Demo")
    st.write("""
    This is a lightweight demo of the SPRNS (Smart Retrieval & Knowledge System) 
    that simulates RAG functionality without requiring heavy ML dependencies.
    
    **Demo Features:**
    - Simulated semantic search
    - Context-aware responses
    - Source attribution
    - Relevance scoring
    """)
    
    # Debug information
    if demo_service and st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("**Debug Information:**")
        st.sidebar.write(f"Knowledge base size: {len(demo_service.knowledge_base)}")
        
        # Show document categories
        categories = {}
        for doc in demo_service.knowledge_base:
            cat = doc.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        st.sidebar.write("**Document Categories:**")
        for cat, count in categories.items():
            st.sidebar.write(f"- {cat}: {count}")
        
        # Show if mock connector is active
        has_mock = hasattr(demo_service, 'mock_connector') and demo_service.mock_connector
        st.sidebar.write(f"Mock connector active: {'✅' if has_mock else '❌'}")
    
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm SPRNS Demo, your Smart Retrieval & Knowledge System. I have knowledge about machine learning, data science, programming, and databases. Ask me anything!"}
        ]
        st.rerun()