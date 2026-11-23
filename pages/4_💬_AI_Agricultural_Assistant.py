"""
Page 4: AI Agricultural Assistant
Module 4 of Phase 2
Interactive chat-style agricultural expert agent.
Explains fertilizer profiles, soil types, N-P-K nutrient dynamics,
application safety, and contextual recommendations.
Operates 100% offline without requiring paid API keys or external LLM connections.
"""

import streamlit as st
from src.ui_helpers import render_header, render_sidebar, apply_custom_css
from src.agent import assistant

st.set_page_config(
    page_title="AI Assistant | AgriAgent",
    page_icon="💬",
    layout="wide"
)

render_sidebar()
render_header(
    title="AI Agricultural Assistant",
    subtitle="Interactive domain-expert assistant for fertilizer science, soil health, crop nutrition, and farm safety.",
    icon="💬"
)

# Transparency & Cost Notice
st.markdown("""
<div style="background-color: #E8F5E9; border-left: 4px solid #2E7D32; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
    <b>🤖 Offline Domain Intelligence Engine:</b>
    <span style="color: #1B5E20; font-size: 0.9rem;">
        This assistant is powered by an offline agronomic rule-and-knowledge reasoning system built directly for this project. 
        It operates <b>without requiring external paid cloud APIs</b> (such as OpenAI or Google Cloud API keys), 
        providing instant, completely free, and reproducible agricultural guidance.
    </span>
</div>
""", unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": assistant.get_welcome_message()}
    ]

# Pre-defined prompt pills
st.markdown("##### 💡 Suggested Questions")
prompt_cols = st.columns(4)

if prompt_cols[0].button("🌾 What is DAP fertilizer?"):
    st.session_state["user_query_input"] = "What is DAP fertilizer?"
if prompt_cols[1].button("🌱 How to manage Sandy Soil?"):
    st.session_state["user_query_input"] = "How do I manage fertilizer for Sandy soil?"
if prompt_cols[2].button("🧪 Nitrogen Deficiency Signs"):
    st.session_state["user_query_input"] = "What are the symptoms of Nitrogen deficiency?"
if prompt_cols[3].button("🛡️ Fertilizer Safety Rules"):
    st.session_state["user_query_input"] = "What safety precautions should I take when applying fertilizers?"

# Additional contextual query if recommendation exists
if "last_rec" in st.session_state:
    last_r = st.session_state["last_rec"]
    if st.button(f"🔍 Explain why {last_r['fertilizer']} was recommended for my {last_r['crop']}"):
        st.session_state["user_query_input"] = f"Explain my recommendation for {last_r['crop']}"

# Display chat messages
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about fertilizers, soil, crops, or N-P-K nutrients...")

# Handle pre-selected prompts or typed inputs
query_to_process = None
if user_input:
    query_to_process = user_input
elif "user_query_input" in st.session_state and st.session_state["user_query_input"]:
    query_to_process = st.session_state["user_query_input"]
    st.session_state["user_query_input"] = None

if query_to_process:
    # Append user message
    st.session_state["chat_messages"].append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # Process response
    context = st.session_state.get("last_rec", None)
    response = assistant.process_query(query_to_process, context=context)

    # Append assistant response
    st.session_state["chat_messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Action to clear conversation
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🗑️ Clear Chat History"):
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": assistant.get_welcome_message()}
    ]
    st.rerun()
