"""
Local Mock Email MCP Tool & Server.

This module implements a Model Context Protocol (MCP) tool that simulates
sending emails by persisting them as structured text files in a local
`sent_emails/` folder.

Beginner Explanation:
- Instead of connecting to a live mail server (like Gmail/SMTP), this tool mocks
  the process by writing the email content to disk.
- It exposes a standard MCP tool `send_email(to, subject, body)` so LLMs and
  agents can discover and invoke it using the official MCP standard.
"""

import os
import re
import asyncio
from datetime import datetime
from typing import Dict, Any

# Attempt to import MCPServer from official python mcp SDK
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        MCPServer = None

# Base directory for sent emails
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_EMAILS_DIR = os.path.join(BASE_DIR, "sent_emails")


def get_next_email_filepath(sent_dir: str = SENT_EMAILS_DIR) -> str:
    """
    Determines the next incremental email file path (e.g., email_003.txt).
    Scans the directory for existing email_*.txt files.
    """
    os.makedirs(sent_dir, exist_ok=True)
    existing_files = os.listdir(sent_dir)
    indices = []

    for fname in existing_files:
        match = re.match(r"^email_(\d+)\.txt$", fname)
        if match:
            indices.append(int(match.group(1)))

    next_idx = max(indices) + 1 if indices else 1
    filename = f"email_{next_idx:03d}.txt"
    return os.path.join(sent_dir, filename)


def save_email_locally(to: str, subject: str, body: str, sent_dir: str = SENT_EMAILS_DIR) -> Dict[str, Any]:
    """
    Core business logic: formats email and writes it to a file.
    
    Required file format:
    To: <recipient>
    Subject: <subject>
    Date: <timestamp>
    Body:
    <email text>
    """
    os.makedirs(sent_dir, exist_ok=True)
    filepath = get_next_email_filepath(sent_dir)
    filename = os.path.basename(filepath)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_content = (
        f"To: {to}\n"
        f"Subject: {subject}\n"
        f"Date: {now_str}\n"
        f"Body:\n"
        f"{body.strip()}\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(file_content)

    return {
        "status": "success",
        "filename": filename,
        "filepath": filepath,
        "to": to,
        "subject": subject,
        "body": body.strip(),
        "message": f"Email successfully sent to {to} and saved to sent_emails/{filename}"
    }


# Initialize the MCP Server instance
mcp_server = MCPServer("MeetingFollowUpEmailServer") if MCPServer is not None else None

# Define tool function
def send_email(to: str, subject: str, body: str) -> str:
    """
    Mock send an email to a recipient and save it to the local sent_emails directory.

    Args:
        to: Recipient name or email address (e.g. 'David' or 'david@example.com')
        subject: Subject line of the email
        body: Main body content of the email

    Returns:
        Confirmation message with path to saved email file.
    """
    res = save_email_locally(to=to, subject=subject, body=body)
    return res["message"]


# Register with MCP server if available
if mcp_server is not None:
    try:
        # Register tool using decorator
        mcp_server.tool()(send_email)
    except Exception as e:
        print(f"[MCP Server] Note: Tool registration notice: {e}")


def call_send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Helper function used by the Agent to invoke the email tool.
    Invokes via MCP Server if available or falls back directly to tool logic.
    Always returns structured dictionary with status and full details.
    """
    # Execute through MCP Server if available
    mcp_message = None
    if mcp_server is not None:
        try:
            # Check if there is an active event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None

            async def _invoke():
                tool_res = await mcp_server.call_tool("send_email", {"to": to, "subject": subject, "body": body})
                return tool_res

            # Run MCP tool call
            if loop and loop.is_running():
                # In nested event loop scenarios, call direct tool logic
                mcp_message = send_email(to, subject, body)
            else:
                call_res = asyncio.run(_invoke())
                # Extract text from CallToolResult
                if hasattr(call_res, "content") and call_res.content:
                    mcp_message = getattr(call_res.content[0], "text", str(call_res.content[0]))
        except Exception:
            # Direct tool fallback
            mcp_message = send_email(to, subject, body)
    else:
        mcp_message = send_email(to, subject, body)

    # Get the latest saved email record
    latest_file = None
    if os.path.exists(SENT_EMAILS_DIR):
        files = [f for f in os.listdir(SENT_EMAILS_DIR) if f.startswith("email_") and f.endswith(".txt")]
        if files:
            files.sort()
            latest_file = files[-1]

    return {
        "status": "success",
        "to": to,
        "subject": subject,
        "body": body,
        "filename": latest_file,
        "filepath": os.path.join(SENT_EMAILS_DIR, latest_file) if latest_file else None,
        "message": mcp_message or f"Email successfully sent and saved to sent_emails/{latest_file}"
    }


if __name__ == "__main__":
    # If run as a script, can run as a stdio MCP server for external clients
    if mcp_server is not None:
        print("Starting Mock Email MCP Server on stdio...")
        mcp_server.run()
    else:
        print("MCPServer library not available. Direct tool mode active.")
