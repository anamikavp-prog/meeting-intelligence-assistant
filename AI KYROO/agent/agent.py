"""
Meeting Intelligence & Follow-up Assistant Agent.

This module implements agentic decision making:
1. Normal Question -> RAG retrieval -> Grounded answer with source citations.
2. Email Request -> RAG task lookup -> Ambiguity check:
   - If ambiguous (e.g. multiple tasks found and none specified): Ask user for clarification.
   - If clear: Draft email from meeting facts -> Invoke MCP send_email tool -> Return confirmation & email preview.
3. Clarification follow-up -> Resolve chosen task -> Generate email -> Invoke MCP tool.
4. Not Found: If a person or task is absent from transcripts, clearly indicate it could not be found.
"""

import os
import re
from typing import Dict, Any, Optional, List

from rag.rag_pipeline import RAGPipeline, get_rag_pipeline
from mcp_server.email_server import call_send_email


class MeetingAssistantAgent:
    """
    Intelligent Assistant Agent coordinating RAG and the mock MCP Email Server.
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag = rag_pipeline or get_rag_pipeline()
        # Session state for handling multi-turn ambiguity clarification
        self.pending_clarification: Optional[Dict[str, Any]] = None

    def reset_state(self):
        """Clears any pending clarification requests."""
        self.pending_clarification = None

    def is_email_request(self, text: str) -> bool:
        """
        Determines whether the user is requesting to send or compose an email.
        """
        patterns = [
            r"\bsend\b.*\bemail\b",
            r"\bemail\b\s+[A-Za-z]+",
            r"\bwrite\b.*\bemail\b",
            r"\bshoot\b.*\bemail\b",
            r"\bremind\b.*\bemail\b",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def extract_person_name(self, text: str) -> Optional[str]:
        """
        Extracts participant name mentioned in user query.
        """
        known_participants = ["David", "Sarah", "Mike", "John"]
        for p in known_participants:
            if re.search(rf"\b{p}\b", text, re.IGNORECASE):
                return p

        # Check for other names e.g. Michael, Alice, Bob
        other_names = ["Michael", "Alice", "Bob", "Charlie", "Peter", "Jane"]
        for o in other_names:
            if re.search(rf"\b{o}\b", text, re.IGNORECASE):
                return o

        return None

    def parse_tasks_for_person(self, person: str) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for all tasks assigned to a specific person.
        Returns a list of structured task dicts.
        """
        results = self.rag.collection.get(where={"person": person})
        tasks = []

        if results and "metadatas" in results:
            for meta in results["metadatas"]:
                raw_task = meta.get("task", "")
                title = meta.get("title", "")
                date = meta.get("date", "")
                fname = meta.get("file", "")

                # Parse specific task name and deadline
                # Format: Person: Task description by Date.
                # e.g.: Sarah: Prepare UI design by September 18, 2026.
                task_desc = raw_task
                deadline = date
                if ":" in raw_task:
                    task_desc = raw_task.split(":", 1)[1].strip()

                deadline_match = re.search(r"by\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", task_desc, re.IGNORECASE)
                if deadline_match:
                    deadline = deadline_match.group(1).strip()
                    short_task = re.sub(r"by\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\.?$", "", task_desc, flags=re.IGNORECASE).strip()
                else:
                    short_task = task_desc

                # Clean verb prefix for concise label (e.g. "Prepare UI design" -> "UI design")
                label = re.sub(r"^(Prepare|Implement|Integrate|Review|Coordinate)\s+", "", short_task, flags=re.IGNORECASE).strip()

                tasks.append({
                    "raw": raw_task,
                    "short_task": short_task,
                    "label": label,
                    "deadline": deadline,
                    "meeting": title,
                    "date": date,
                    "file": fname,
                    "source": f"Meeting: {title}\nDate: {date}\nFile: {fname}"
                })

        # Deduplicate tasks based on raw text
        seen = set()
        deduped = []
        for t in tasks:
            if t["raw"] not in seen:
                seen.add(t["raw"])
                deduped.append(t)

        return deduped

    def draft_email_content(self, recipient: str, task: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates the email subject and body strictly from meeting information.
        Follows the project's exact required formatting.
        """
        short_task = task.get("short_task", task.get("label", "task"))
        deadline = task.get("deadline", "the scheduled date")

        # Format subject (e.g., Reminder: Authentication Module Deadline)
        label_title = task.get("label", "Task").title()
        subject = f"Reminder: {label_title} Deadline"

        # Format body:
        # Hi <Person>,
        # This is a reminder that the <task> is due on <deadline>.
        # Best regards,
        # Project Team
        body = (
            f"Hi {recipient},\n\n"
            f"This is a reminder that the {short_task.lower()} is due on {deadline}.\n\n"
            f"Best regards,\n"
            f"Project Team"
        )

        return {
            "to": recipient,
            "subject": subject,
            "body": body
        }

    def execute_send_email(self, recipient: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drafts the email and invokes the MCP send_email tool to save it locally.
        """
        draft = self.draft_email_content(recipient, task)
        
        # Invoke MCP tool
        mcp_res = call_send_email(
            to=draft["to"],
            subject=draft["subject"],
            body=draft["body"]
        )

        # Clear any pending clarification
        self.pending_clarification = None

        citation = f"Source:\nMeeting: {task['meeting']}\nDate: {task['date']}\nFile: {task['file']}"

        return {
            "response_type": "email_sent",
            "message": mcp_res["message"],
            "to": draft["to"],
            "subject": draft["subject"],
            "body": draft["body"],
            "saved_file": mcp_res.get("filename"),
            "formatted_sources": citation,
            "answer": (
                f"**Email successfully sent via MCP tool!**\n\n"
                f"**Saved to:** `{mcp_res.get('filename')}`\n\n"
                f"```text\n"
                f"To: {draft['to']}\n"
                f"Subject: {draft['subject']}\n\n"
                f"{draft['body']}\n"
                f"```"
            )
        }

    def handle_clarification_response(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Resolves a previous ambiguity question if user was prompted to select a task.
        """
        if not self.pending_clarification:
            return None

        state = self.pending_clarification
        recipient = state["recipient"]
        tasks = state["tasks"]

        choice_idx = None
        input_lower = user_input.strip().lower()

        # Check for numeric option: "1", "2", "option 1", "#1", "first", "second"
        if input_lower in ["1", "option 1", "#1", "first", "the first one", "1."]:
            choice_idx = 0
        elif input_lower in ["2", "option 2", "#2", "second", "the second one", "2."]:
            choice_idx = 1
        elif input_lower in ["3", "option 3", "#3", "third", "the third one", "3."]:
            choice_idx = 2
        else:
            # Check for keyword matching in task label or short task
            for idx, t in enumerate(tasks):
                # e.g., if user types "ui design" or "api documentation"
                if t["label"].lower() in input_lower or t["short_task"].lower() in input_lower:
                    choice_idx = idx
                    break

        if choice_idx is not None and choice_idx < len(tasks):
            selected_task = tasks[choice_idx]
            return self.execute_send_email(recipient, selected_task)

        # Still ambiguous or invalid option
        options_text = "\n".join(
            f"{i+1}. {t['label']} — {t['deadline']}" for i, t in enumerate(tasks)
        )
        return {
            "response_type": "clarification_needed",
            "message": f"Please choose one of the available options:\n\n{options_text}",
            "answer": f"I didn't quite catch that. Please select which task for {recipient}:\n\n{options_text}",
            "formatted_sources": ""
        }

    def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point: analyzes user request, coordinates RAG, ambiguity handling,
        and MCP tool execution.
        """
        clean_input = user_input.strip()

        # 1. Check if user is responding to an existing clarification prompt
        if self.pending_clarification:
            clarified = self.handle_clarification_response(clean_input)
            if clarified:
                return clarified

        # 2. Check if this is an Email Request
        if self.is_email_request(clean_input):
            recipient = self.extract_person_name(clean_input)
            
            if not recipient:
                return {
                    "response_type": "error",
                    "answer": "I could not determine who you would like to email. Please specify a participant name (e.g., David, Sarah, Mike, John).",
                    "formatted_sources": ""
                }

            # Search tasks for this person
            tasks = self.parse_tasks_for_person(recipient)

            # Case: Person not in transcripts (e.g. Michael)
            if not tasks:
                return {
                    "response_type": "not_found",
                    "answer": f"The requested information regarding {recipient} could not be found in the meeting transcripts. No email was sent.",
                    "formatted_sources": ""
                }

            # Check if user input already narrows down to a specific task
            # e.g. "Send David an email reminding him about his authentication task"
            matching_tasks = []
            for t in tasks:
                # Check for keywords like "authentication", "ui design", "api", "deployment"
                keywords = [
                    w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", t["short_task"])
                    if w.lower() not in ["and", "the", "for", "prepare", "implement", "integrate"]
                ]
                if any(kw in clean_input.lower() for kw in keywords):
                    matching_tasks.append(t)

            # If user specified a distinct task
            if len(matching_tasks) == 1:
                return self.execute_send_email(recipient, matching_tasks[0])

            # If person only has 1 task in total
            if len(tasks) == 1 and not matching_tasks:
                return self.execute_send_email(recipient, tasks[0])

            # Ambiguity detected: Multiple tasks found and request is ambiguous!
            # Example: "Send Sarah an email about her deadline."
            self.pending_clarification = {
                "recipient": recipient,
                "tasks": tasks
            }

            options_text = "\n".join(
                f"{i+1}. {t['label']} — {t['deadline']}" for i, t in enumerate(tasks)
            )

            clarification_message = (
                f"I found multiple tasks for {recipient}. Which one would you like me to mention?\n\n"
                f"{options_text}"
            )

            return {
                "response_type": "clarification_needed",
                "answer": clarification_message,
                "options": [f"{t['label']} — {t['deadline']}" for t in tasks],
                "recipient": recipient,
                "formatted_sources": ""
            }

        # 3. Normal Question -> RAG Retrieval
        rag_res = self.rag.query(clean_input)
        return {
            "response_type": "rag_answer",
            "answer": rag_res["answer"],
            "sources": rag_res.get("sources", []),
            "formatted_sources": rag_res.get("formatted_sources", "")
        }


# Singleton factory
_agent_instance = None

def get_agent(rag_pipeline: Optional[RAGPipeline] = None) -> MeetingAssistantAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MeetingAssistantAgent(rag_pipeline=rag_pipeline)
    return _agent_instance
