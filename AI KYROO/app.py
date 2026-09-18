"""
Meeting Intelligence & Follow-up Assistant
Streamlit Web Interface

A beginner-friendly interface demonstrating:
1. RAG question-answering with ChromaDB and source citations.
2. Agentic decision making with ambiguity detection.
3. Local Mock MCP email tool invocation and disk persistence in `sent_emails/`.
"""

import os
import glob
import streamlit as st
from dotenv import load_dotenv

from rag.rag_pipeline import RAGPipeline, get_rag_pipeline
from agent.agent import MeetingAssistantAgent, get_agent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Meeting Intelligence & Follow-up Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for rich, beginner-friendly aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
    .email-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .email-header {
        font-weight: 600;
        color: #065F46;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your **Meeting Intelligence & Follow-up Assistant**.\n\n"
                "I can answer questions based strictly on your meeting transcripts and draft follow-up emails "
                "via our local MCP mock email server.\n\n"
                "**Try asking:**\n"
                "- *\"What is David responsible for?\"*\n"
                "- *\"What database did the team choose?\"*\n"
                "- *\"Send David an email reminding him about his authentication task.\"*\n"
                "- *\"Send Sarah an email about her deadline.\"*\n"
                "- *\"What did Michael agree to do?\"*"
            ),
            "sources": "",
            "type": "welcome"
        }
    ]

# -----------------------------------------------------------------------------
# Sidebar: Settings, Sent Emails & Quick Evaluation Prompts
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    # OpenAI API Key Input
    saved_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key (Optional):",
        value=saved_key,
        type="password",
        help="If not provided, the assistant uses the local deterministic evaluation engine for testing."
    )

    if api_key_input:
        st.success("🟢 OpenAI API Configured", icon="✅")
    else:
        st.info("🟡 Using Local Grounded Engine (Offline Evaluation Mode)", icon="ℹ️")

    # Initialize RAG and Agent with key
    rag = get_rag_pipeline(api_key=api_key_input if api_key_input else None)
    agent = get_agent(rag_pipeline=rag)

    st.divider()

    # Index management
    st.subheader("📚 Meeting Knowledge Base")
    doc_count = rag.collection.count()
    st.caption(f"**ChromaDB Chunks Indexed:** `{doc_count}`")

    if st.button("🔄 Re-index Transcripts", use_container_width=True):
        count = rag.ingest_documents()
        st.success(f"Indexed {count} chunks from data/!")
        st.rerun()

    st.divider()

    # Quick test prompts for evaluator
    st.subheader("🧪 Evaluation Test Prompts")
    st.caption("Click any test case to run it immediately:")

    eval_prompts = [
        ("Test 1: David's Tasks", "What is David responsible for?"),
        ("Test 2: Database Choice", "What database did the team choose?"),
        ("Test 3: Send David Email", "Send David an email reminding him about his authentication task."),
        ("Test 4: Ambiguous Request (Sarah)", "Send Sarah an email about her deadline."),
        ("Test 5: Michael Not Found", "What did Michael agree to do?"),
    ]

    selected_eval_prompt = None
    for label, prompt_text in eval_prompts:
        if st.button(label, use_container_width=True):
            selected_eval_prompt = prompt_text

    st.divider()

    # Local Sent Emails Viewer
    st.subheader("📬 Sent Emails (`sent_emails/`)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sent_dir = os.path.join(base_dir, "sent_emails")
    if os.path.exists(sent_dir):
        files = [f for f in os.listdir(sent_dir) if f.startswith("email_") and f.endswith(".txt")]
        files.sort(reverse=True)
        if files:
            st.caption(f"Total Emails Generated: **{len(files)}**")
            selected_email = st.selectbox("Inspect Sent Email:", files, key="email_selector")
            if selected_email:
                with open(os.path.join(sent_dir, selected_email), "r", encoding="utf-8") as ef:
                    email_text = ef.read()
                st.code(email_text, language="text")
        else:
            st.caption("No emails sent yet.")
    else:
        st.caption("Directory `sent_emails/` does not exist yet.")

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📋 Meeting Intelligence & Follow-up Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded RAG analysis over meeting transcripts with Agentic Ambiguity Handling and MCP Mock Email delivery.</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render citations if present
        if msg.get("sources"):
            with st.expander("📖 Source Citation", expanded=False):
                st.markdown(f'<div class="source-box">{msg["sources"]}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Handle Clarification Quick-Select Buttons (if agent is waiting for user choice)
# -----------------------------------------------------------------------------
clarification_input = None
if agent.pending_clarification:
    st.info("💡 **Agent needs clarification:** Please select which task you'd like to include in the email:")
    col1, col2 = st.columns(2)
    tasks = agent.pending_clarification.get("tasks", [])
    if len(tasks) >= 1:
        with col1:
            if st.button(f"1. {tasks[0]['label']} ({tasks[0]['deadline']})", use_container_width=True):
                clarification_input = "1"
    if len(tasks) >= 2:
        with col2:
            if st.button(f"2. {tasks[1]['label']} ({tasks[1]['deadline']})", use_container_width=True):
                clarification_input = "2"

# -----------------------------------------------------------------------------
# Process Input
# -----------------------------------------------------------------------------
user_query = st.chat_input("Type a question or email request (e.g. 'What is David responsible for?')...")

# Determine active prompt (from chat_input, quick eval buttons, or clarification button)
active_input = clarification_input or selected_eval_prompt or user_query

if active_input:
    # 1. Append and display user message
    st.session_state.messages.append({"role": "user", "content": active_input})
    with st.chat_message("user"):
        st.markdown(active_input)

    # 2. Process via Agent
    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            response = agent.process_request(active_input)
            
            ans_text = response.get("answer", "")
            citations = response.get("formatted_sources", "")
            resp_type = response.get("response_type", "general")

            # Display response
            if resp_type == "email_sent":
                st.success("✉️ **Email successfully dispatched via Local Mock MCP Tool!**")
                st.markdown(ans_text)
            elif resp_type == "clarification_needed":
                st.warning("⚠️ **Ambiguity Detected**")
                st.markdown(ans_text)
            elif resp_type == "not_found":
                st.error("🔍 **Information Not Found**")
                st.markdown(ans_text)
            else:
                st.markdown(ans_text)

            # Display citations if available
            if citations:
                with st.expander("📖 Source Citation", expanded=True):
                    st.markdown(f'<div class="source-box">{citations}</div>', unsafe_allow_html=True)

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ans_text,
                "sources": citations,
                "type": resp_type
            })

    # Rerun to update sidebar email list and button state if an email was sent
    if resp_type == "email_sent":
        st.rerun()
