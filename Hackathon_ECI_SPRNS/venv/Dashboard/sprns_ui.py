# sprns_ui.py
import time
import streamlit as st

st.set_page_config(page_title="sprns Chat UI", layout="centered")
st.title("🤖 Sprns Chat UI")

# Session-state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant inside sprns."}
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")
if user_input:
    # Save and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Simple reply (replace get_response to call a real API)
    with st.spinner("Assistant is thinking..."):
        def get_response(prompt: str) -> str:
            time.sleep(0.25)  # small UX delay
            return f"sprns says: {prompt}"

        reply = get_response(user_input)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

