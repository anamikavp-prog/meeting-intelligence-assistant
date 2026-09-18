"""
Meeting Intelligence & Follow-up Assistant
Streamlit Web Interface

A beginner-friendly interface demonstrating:
1. RAG question-answering with ChromaDB and source citations.
2. Agentic decision making with ambiguity detection.
3. Local Mock MCP email tool invocation and disk persistence in `sent_emails/`.
4. Quick evaluation test cases with official questions and correct answers.
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
    .notfound-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 4px solid #EF4444;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        font-weight: 500;
        white-space: pre-line;
    }
    .eval-card {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
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
                "**Try asking or click the evaluation tests in the sidebar:**\n"
                "- *\"What is David responsible for?\"*\n"
                "- *\"What database did the team choose?\"*\n"
                "- *\"Send David an email reminding him about his authentication task.\"*\n"
                "- *\"Send Sarah an email about her deadline.\"*\n"
                "- *\"What did Michael agree to do?\"*\n"
                "- *\"Send Anjali a follow-up email about the database migration.\"*"
            ),
            "sources": "",
            "type": "welcome"
        }
    ]

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# Initialize RAG Pipeline and Agent in session state
if "rag" not in st.session_state:
    st.session_state.rag = get_rag_pipeline()
rag = st.session_state.rag

if "agent" not in st.session_state:
    st.session_state.agent = MeetingAssistantAgent(rag_pipeline=rag)
agent = st.session_state.agent

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
        if rag.api_key != api_key_input:
            st.session_state.rag = get_rag_pipeline(api_key=api_key_input)
            st.session_state.agent = MeetingAssistantAgent(rag_pipeline=st.session_state.rag)
            st.rerun()
    else:
        st.info("🟡 Using Local Grounded Engine (Evaluation Mode)", icon="ℹ️")

    st.divider()

    # Knowledge Base Controls
    st.subheader("📚 Meeting Knowledge Base")
    doc_count = rag.collection.count()
    st.caption(f"**ChromaDB Chunks Indexed:** `{doc_count}`")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Re-index", use_container_width=True):
            count = rag.ingest_documents()
            st.success(f"Indexed {count} chunks!")
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            agent.reset_state()
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.pending_input = None
            st.rerun()

    st.divider()

    # Evaluation Test Prompts Section
    st.subheader("🧪 Evaluation Test Prompts")
    st.caption("Click any test case to run it immediately or view the expected correct answer:")

    eval_scenarios = [
        {
            "id": 1,
            "title": "Test 1: David's Tasks",
            "query": "What is David responsible for?",
            "expected": "Mentions authentication module (Sep 22) and deployment configuration (Sep 28) with citations from meeting1.txt and meeting3.txt."
        },
        {
            "id": 2,
            "title": "Test 2: Database Decision",
            "query": "What database did the team choose?",
            "expected": "PostgreSQL (Project Alpha Planning, September 10, 2026, meeting1.txt)."
        },
        {
            "id": 3,
            "title": "Test 3: Send David Email",
            "query": "Send David an email reminding him about his authentication task.",
            "expected": "Email generated strictly from facts and sent via MCP mock email tool to sent_emails/."
        },
        {
            "id": 4,
            "title": "Test 4: Ambiguous Request (Sarah)",
            "query": "Send Sarah an email about her deadline.",
            "expected": "Detects ambiguity between 2 tasks (UI design by Sep 18 vs API documentation by Sep 20), halts MCP execution, and requests clarification."
        },
        {
            "id": 5,
            "title": "Test 5: Michael Not Found",
            "query": "What did Michael agree to do?",
            "expected": "No relevant meeting information found. (Michael is absent from transcripts)."
        },
        {
            "id": 6,
            "title": "Test 6: Missing Info / Out-of-Scope",
            "query": "Send Anjali a follow-up email about the database migration.",
            "expected": "No relevant meeting information found.\nEmail was not sent"
        }
    ]

    for scenario in eval_scenarios:
        with st.expander(f"📌 {scenario['title']}", expanded=False):
            st.markdown(f"**Question:**\n`{scenario['query']}`")
            st.markdown(f"**Expected / Correct Answer:**\n*{scenario['expected']}*")
            if st.button(f"▶️ Run Test {scenario['id']}", key=f"eval_btn_{scenario['id']}", use_container_width=True):
                st.session_state.pending_input = scenario["query"]
                st.rerun()

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
        msg_type = msg.get("type", "general")
        content = msg.get("content", "")

        if msg_type == "email_sent":
            st.success("✉️ **Email successfully dispatched via Local Mock MCP Tool!**")
            st.markdown(content)
        elif msg_type == "clarification_needed":
            st.warning("⚠️ **Ambiguity Detected**")
            st.markdown(content)
        elif msg_type == "not_found":
            st.error("🔍 **Information Not Found**")
            st.markdown(content)
        else:
            st.markdown(content)
        
        # Render citations if present
        if msg.get("sources"):
            with st.expander("📖 Source Citation", expanded=True):
                st.markdown(f'<div class="source-box">{msg["sources"]}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Handle Clarification Quick-Select Buttons (if agent is waiting for user choice)
# -----------------------------------------------------------------------------
clarification_choice = None
if agent.pending_clarification:
    st.info("💡 **Agent needs clarification:** Please select which task you'd like to include in the email:")
    col1, col2 = st.columns(2)
    tasks = agent.pending_clarification.get("tasks", [])
    if len(tasks) >= 1:
        with col1:
            if st.button(f"1. {tasks[0]['label']} ({tasks[0]['deadline']})", key="clarify_btn_1", use_container_width=True):
                clarification_choice = "1"
    if len(tasks) >= 2:
        with col2:
            if st.button(f"2. {tasks[1]['label']} ({tasks[1]['deadline']})", key="clarify_btn_2", use_container_width=True):
                clarification_choice = "2"

# -----------------------------------------------------------------------------
# Process Input (from Chat Input, Evaluation Buttons, or Clarification Buttons)
# -----------------------------------------------------------------------------
user_query = st.chat_input("Type a question or email request (e.g. 'What is David responsible for?')...")

# Check which input was triggered
active_input = None
if st.session_state.pending_input:
    active_input = st.session_state.pending_input
    st.session_state.pending_input = None
elif clarification_choice:
    active_input = clarification_choice
elif user_query:
    active_input = user_query

if active_input:
    # 1. Append user message to history
    st.session_state.messages.append({"role": "user", "content": active_input, "type": "user_input"})

    # 2. Process via Agent
    response = agent.process_request(active_input)
    
    ans_text = response.get("answer", "")
    citations = response.get("formatted_sources", "")
    resp_type = response.get("response_type", "general")

    # 3. Append assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ans_text,
        "sources": citations,
        "type": resp_type
    })

    # 4. Rerun so the new message and UI state update immediately
    st.rerun()
