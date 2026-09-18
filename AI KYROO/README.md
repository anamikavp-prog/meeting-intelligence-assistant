# 📋 Meeting Intelligence & Follow-up Assistant

A beginner-friendly, evaluation-ready AI assistant that analyzes meeting transcripts using **Retrieval-Augmented Generation (RAG)**, handles user requests with **Agentic Decision Making**, and executes actions using a local **Model Context Protocol (MCP)** mock email tool.

---

## 🌟 Project Overview

During busy sprints, project teams make key decisions and assign action items across multiple meetings. Remembering who is responsible for what, when deadlines are due, or drafting follow-up emails manually is error-prone.

The **Meeting Intelligence & Follow-up Assistant**:
1. **Indexes Meeting Transcripts**: Ingests meeting notes, discussions, decisions, and action items into a ChromaDB vector store.
2. **Answers Questions Accurately**: Retrieves relevant context to answer questions (e.g., *"What is David responsible for?"*) and **strictly cites sources** (Meeting Name, Date, and File).
3. **Drafts & Sends Follow-Up Emails via MCP**: Generates follow-up emails based strictly on verified meeting facts and uses a local **Model Context Protocol (MCP)** tool to simulate sending the email by persisting it to `sent_emails/`.
4. **Detects & Resolves Ambiguity**: If a user asks to email someone with multiple pending tasks (e.g., *"Send Sarah an email about her deadline"*), the assistant does not guess—it asks for clarification first!

---

## 🏛️ System Architecture

Here is the high-level architecture of how data flows through the application:

```text
+-------------------------------------------------------------------------+
|                            USER / STREAMLIT UI                          |
|         (Questions, Email Requests, Clarification Selections)           |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                              AGENT LAYER                                |
|                        (agent/agent.py)                                 |
|                                                                         |
|  1. Classifies user intent (Normal Question vs. Email Request)          |
|  2. Extracts participants and targets                                   |
|  3. Detects Ambiguity:                                                  |
|     - Multiple tasks found? -> Prompts user for clarification           |
|     - Specific task known?  -> Proceeds to email drafting               |
+-------------------+---------------------------------+-------------------+
                    |                                 |
         RAG Search |                      Tool Call  | (send_email)
                    v                                 v
+-----------------------------------+   +---------------------------------+
|            RAG PIPELINE           |   |      LOCAL MOCK EMAIL MCP       |
|       (rag/rag_pipeline.py)       |   |  (mcp_server/email_server.py)   |
|                                   |   |                                 |
| - Document Loader (data/*.txt)    |   | - Exposes MCP tool: send_email  |
| - Text Chunker & Metadata Parser  |   | - Simulates mail dispatch       |
| - ChromaDB Vector Store           |   | - Writes to sent_emails/        |
| - OpenAI / Local Grounded LLM     |   |   (e.g., email_001.txt)         |
| - Grounded Answers with Citations |   +---------------------------------+
+-----------------------------------+
```

---

## 💡 How Key Technologies Work (In Simple Terms)

### 1. How RAG (Retrieval-Augmented Generation) Works
Large Language Models have general knowledge, but they do not know private internal company meetings. RAG solves this in 5 simple steps:
1. **Document Loading**: Transcripts from `data/meeting1.txt`, `data/meeting2.txt`, etc., are read from disk.
2. **Chunking**: Transcripts are broken down into logical chunks (Discussions, Decisions, Individual Action Items) with metadata (Meeting Title, Date, File, Person).
3. **Embeddings**: Text chunks are converted into numerical vector representations.
4. **Vector Storage (ChromaDB)**: The vectors are stored in a persistent local database (`chroma_db/`).
5. **Similarity Search & Generation**: When you ask a question, ChromaDB finds the most relevant chunks. The LLM then answers the question using **only** those retrieved chunks and attaches citations. If the information isn't in the meetings, it says so instead of making things up.

### 2. How the Agent Works
The agent (`agent/agent.py`) acts as the "brain":
- When given a question like *"What database did the team choose?"*, it routes the query directly to RAG.
- When given an action request like *"Send David an email reminding him about his authentication task"*, it extracts the person and task, retrieves meeting details, drafts the message, and executes the MCP email tool.

### 3. How Ambiguity Handling Works
A common flaw in AI assistants is guessing when instructions are incomplete.
- Suppose Sarah has two deadlines:
  1. UI design — September 18, 2026
  2. API documentation — September 20, 2026
- If the user says: *"Send Sarah an email about her deadline"*, the agent detects that Sarah has more than one task and pauses:
  > *"I found multiple tasks for Sarah. Which one would you like me to mention?*
  > *1. UI design — September 18, 2026*
  > *2. API documentation — September 20, 2026"*
- Only after the user chooses `1` or `2` does the agent send the email.

### 4. How MCP (Model Context Protocol) Works
**Model Context Protocol (MCP)** is an open standard that allows AI models to connect securely to tools and data sources.
- In this project, `mcp_server/email_server.py` implements an MCP server defining the tool:
  `send_email(to: str, subject: str, body: str)`.
- Instead of connecting to a real email provider (avoiding complex OAuth, SMTP credentials, or risk of accidental real emails), this **Local Mock MCP Tool** writes the email to `sent_emails/email_XXX.txt` with timestamp, recipient, subject, and body.

---

## 📁 Project Structure

```text
c:\AI KYROO\
├── app.py                     # Streamlit Web Application
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template for OpenAI API key
├── .gitignore                 # Excludes .env, chroma_db, and cache files
├── README.md                  # Beginner-friendly guide and documentation
│
├── data/                      # Synthetic meeting transcripts
│   ├── meeting1.txt           # Project Alpha Planning (Sep 10, 2026)
│   ├── meeting2.txt           # API Integration Meeting (Sep 12, 2026)
│   └── meeting3.txt           # Product Release Meeting (Sep 14, 2026)
│
├── rag/
│   ├── __init__.py
│   └── rag_pipeline.py        # Loading, chunking, ChromaDB vector indexing & citations
│
├── agent/
│   ├── __init__.py
│   └── agent.py               # Request routing, ambiguity detection, email generation
│
├── mcp_server/
│   ├── __init__.py
│   └── email_server.py        # Local Mock Email MCP Tool saving to sent_emails/
│
├── sent_emails/               # Folder storing generated mock emails (email_001.txt, etc.)
│
└── tests/
    └── test_assistant.py      # Automated test suite covering all 5 evaluation test cases
```

---

## 📦 Requirements & Prerequisites

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **Required Libraries**:
  - `streamlit` (Web UI)
  - `chromadb` (Vector database)
  - `openai` (Embeddings & Chat Completions)
  - `mcp` (Model Context Protocol standard)
  - `python-dotenv` (Configuration loader)
  - `pydantic` (Data validation)

---

## 🚀 Installation Steps

1. **Open your terminal** in the project directory:
   ```bash
   cd "c:\AI KYROO"
   ```

2. **(Optional but recommended) Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Environment Configuration

1. Copy the `.env.example` file to create your `.env`:
   ```bash
   copy .env.example .env     # Windows
   cp .env.example .env       # macOS / Linux
   ```

2. Open `.env` in any text editor and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=sk-proj-yourActualOpenAIKeyHere
   ```

> **Note on Evaluation / Offline Mode:**
> If you do not have an OpenAI API key immediately available, the application includes a **built-in local grounded evaluation engine**. All 5 evaluation test cases and the Streamlit UI function immediately out-of-the-box!

---

## 🖥️ How to Run the Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your web browser at:
`http://localhost:8501`

### What You Can Do in the UI:
- **Ask normal questions**: Enter any query in the chat input.
- **Request emails**: Request reminders for any meeting participant.
- **Quick Evaluation Buttons**: Click any of the 5 test buttons in the left sidebar to execute evaluation scenarios instantly.
- **Inspect Sent Emails**: View all mock emails created by the MCP tool right inside the sidebar inspector.
- **Re-index Knowledge Base**: Click "Re-index Transcripts" anytime you modify or add meeting files to `data/`.

---

## 🧪 Automated Testing (Evaluation Scenarios)

To run the automated evaluation test suite, execute:

```bash
python tests/test_assistant.py
```

### Verified Test Cases:

| Test Case | Prompt / Action | Expected Result | Status |
|---|---|---|:---:|
| **Test 1** | *"What is David responsible for?"* | Retrieves David's tasks (Authentication module by Sep 22, 2026; Deployment config by Sep 28, 2026) with full meeting citations. | ✅ **PASSED** |
| **Test 2** | *"What database did the team choose?"* | Identifies PostgreSQL from Meeting 1 with exact file citation (`meeting1.txt`). | ✅ **PASSED** |
| **Test 3** | *"Send David an email reminding him about his authentication task."* | Retrieves task, drafts email, invokes MCP `send_email`, saves to `sent_emails/`, and returns confirmation. | ✅ **PASSED** |
| **Test 4** | *"Send Sarah an email about her deadline."* | Detects multiple tasks for Sarah (UI design vs. API documentation), prompts for clarification; upon choosing `1`, sends clarified email. | ✅ **PASSED** |
| **Test 5** | *"What did Michael agree to do?"* | Confirms Michael is not in the meeting records and clearly reports information not found. | ✅ **PASSED** |

---

## 📝 Example Requests & Responses

### Example 1: Normal Question
**User:**
> "What is David responsible for?"

**Assistant:**
> Based on the meeting transcripts, David is responsible for:
> - David: Prepare deployment configuration by September 28, 2026.
> - David: Implement authentication module by September 22, 2026.
>
> **Source:**
> Meeting: Product Release Meeting
> Date: September 14, 2026
> File: meeting3.txt
>
> Meeting: Project Alpha Planning
> Date: September 10, 2026
> File: meeting1.txt

---

### Example 2: Email Request
**User:**
> "Send David an email reminding him about his authentication task."

**Assistant:**
> **Email successfully sent via MCP tool!**
> **Saved to:** `email_004.txt`
>
> ```text
> To: David
> Subject: Reminder: Authentication Module Deadline
>
> Hi David,
>
> This is a reminder that the implement authentication module is due on September 22, 2026.
>
> Best regards,
> Project Team
> ```

---

### Example 3: Ambiguity Clarification
**User:**
> "Send Sarah an email about her deadline."

**Assistant:**
> I found multiple tasks for Sarah. Which one would you like me to mention?
>
> 1. UI design — September 18, 2026
> 2. API documentation — September 20, 2026

**User:**
> "1"

**Assistant:**
> **Email successfully sent via MCP tool!**
> **Saved to:** `email_005.txt`
>
> ```text
> To: Sarah
> Subject: Reminder: Ui Design Deadline
>
> Hi Sarah,
>
> This is a reminder that the prepare ui design is due on September 18, 2026.
>
> Best regards,
> Project Team
> ```

---

## ⚠️ Limitations

1. **Mock Email Transport**: The MCP tool writes `.txt` files to `sent_emails/` instead of connecting to an external mail server (by design for safe local evaluation).
2. **Text Transcript Format**: Currently ingests structured `.txt` files in `data/`. Audio/video ingestion requires preprocessing (e.g. Whisper).
3. **Local Vector Storage**: ChromaDB runs locally in SQLite/filesystem mode inside `chroma_db/`.

---

## 🔮 Future Improvements

1. **Live Email Integration**: Add an optional SMTP / Gmail API adapter behind the MCP tool.
2. **Audio Transcription**: Add automated speech-to-text with OpenAI Whisper for MP3/WAV meeting recordings.
3. **Calendar Integration**: Extend the MCP server with a `create_calendar_invite` tool to schedule deadline checkpoints automatically.
4. **Slack/Teams Webhook**: Allow dispatching follow-up summaries directly to project team channels.
