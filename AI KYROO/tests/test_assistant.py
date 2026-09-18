"""
Evaluation Test Suite for Meeting Intelligence & Follow-up Assistant.

Verifies the 5 required evaluation test cases:
1. Normal Question: 'What is David responsible for?'
2. Normal Question: 'What database did the team choose?'
3. Direct Email Request: 'Send David an email reminding him about his authentication task.'
4. Ambiguous Email Request: 'Send Sarah an email about her deadline.' (followed by resolution)
5. Out-of-scope / Not Found: 'What did Michael agree to do?'

Compatible with both pytest (`python -m pytest`) and direct script execution (`python tests/test_assistant.py`).
"""

import os
import sys
import pytest

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent.agent import MeetingAssistantAgent
from rag.rag_pipeline import RAGPipeline


@pytest.fixture(scope="module")
def shared_agent():
    """Initializes and returns a shared MeetingAssistantAgent instance."""
    rag = RAGPipeline()
    return MeetingAssistantAgent(rag_pipeline=rag)


def test_david_responsibilities(shared_agent):
    """Test 1: What is David responsible for?"""
    shared_agent.reset_state()
    res = shared_agent.process_request("What is David responsible for?")
    ans = res["answer"]
    assert "authentication" in ans.lower(), "David's authentication task missing."
    assert "september 22" in ans.lower(), "September 22 deadline missing."
    assert len(res.get("formatted_sources", "")) > 0, "Missing source citations."


def test_database_choice(shared_agent):
    """Test 2: What database did the team choose?"""
    shared_agent.reset_state()
    res = shared_agent.process_request("What database did the team choose?")
    ans = res["answer"]
    assert "postgresql" in ans.lower(), "PostgreSQL decision not found."
    assert "meeting1.txt" in res.get("formatted_sources", "").lower(), "meeting1.txt not cited."


def test_email_request_david(shared_agent):
    """Test 3: Send David an email reminding him about his authentication task."""
    shared_agent.reset_state()
    res = shared_agent.process_request("Send David an email reminding him about his authentication task.")
    assert res["response_type"] == "email_sent", "Email was not sent."
    assert res["to"] == "David", "Recipient is not David."
    assert "authentication" in res["subject"].lower() or "authentication" in res["body"].lower(), "Authentication task missing."
    assert "september 22" in res["body"].lower(), "September 22 deadline missing from email body."
    
    saved_file = res.get("saved_file")
    assert saved_file is not None, "No saved file returned."
    saved_path = os.path.join(BASE_DIR, "sent_emails", saved_file)
    assert os.path.exists(saved_path), f"Saved email file {saved_path} does not exist on disk."


def test_ambiguous_email_sarah(shared_agent):
    """Test 4: Send Sarah an email about her deadline. (Ambiguous request & clarification)"""
    shared_agent.reset_state()
    res = shared_agent.process_request("Send Sarah an email about her deadline.")
    assert res["response_type"] == "clarification_needed", "Expected clarification prompt for Sarah."
    assert "multiple tasks" in res["answer"].lower(), "Ambiguity message did not mention multiple tasks."
    assert "ui design" in res["answer"].lower(), "UI design option missing from clarification."
    assert "api documentation" in res["answer"].lower(), "API documentation option missing."

    # Follow-up resolution
    res_clarified = shared_agent.process_request("1")
    assert res_clarified["response_type"] == "email_sent", "Email was not sent after clarification."
    assert res_clarified["to"] == "Sarah", "Recipient should be Sarah."
    assert "ui design" in res_clarified["body"].lower(), "Selected task (UI design) missing from body."
    assert "september 18" in res_clarified["body"].lower(), "September 18 deadline missing from body."


def test_not_found_michael(shared_agent):
    """Test 5: What did Michael agree to do? (Not found)"""
    shared_agent.reset_state()
    res = shared_agent.process_request("What did Michael agree to do?")
    ans = res["answer"]
    assert "could not be found" in ans.lower() or "not found" in ans.lower(), "Did not report information as missing."
    assert len(res.get("sources", [])) == 0, "Should not cite sources for absent participant."


def test_missing_info_email_anjali(shared_agent):
    """Test 6: Send Anjali a follow-up email about the database migration. (Missing information)"""
    shared_agent.reset_state()
    res = shared_agent.process_request("Send Anjali a follow-up email about the database migration.")
    assert res["response_type"] == "not_found", f"Expected not_found, got {res['response_type']}"
    assert "no relevant meeting information found" in res["answer"].lower(), "Expected 'No relevant meeting information found.'"
    assert "email was not sent" in res["answer"].lower(), "Expected 'Email was not sent'"


def run_tests():
    """Standalone CLI runner for direct execution."""
    print("=" * 70)
    print("RUNNING MEETING INTELLIGENCE ASSISTANT EVALUATION TEST SUITE")
    print("=" * 70)

    rag = RAGPipeline()
    agent = MeetingAssistantAgent(rag_pipeline=rag)
    passed_count = 0

    print("\n[TEST 1] Query: 'What is David responsible for?'")
    test_david_responsibilities(agent)
    print(">>> TEST 1 PASSED: David's responsibilities and citations correctly returned.")
    passed_count += 1

    print("\n[TEST 2] Query: 'What database did the team choose?'")
    test_database_choice(agent)
    print(">>> TEST 2 PASSED: PostgreSQL decision and source correctly returned.")
    passed_count += 1

    print("\n[TEST 3] Query: 'Send David an email reminding him about his authentication task.'")
    test_email_request_david(agent)
    print(">>> TEST 3 PASSED: Email successfully generated and saved via MCP tool.")
    passed_count += 1

    print("\n[TEST 4] Query: 'Send Sarah an email about her deadline.'")
    test_ambiguous_email_sarah(agent)
    print(">>> TEST 4 PASSED: Ambiguity detected, clarified, and email dispatched via MCP tool.")
    passed_count += 1

    print("\n[TEST 5] Query: 'What did Michael agree to do?'")
    test_not_found_michael(agent)
    print(">>> TEST 5 PASSED: Handled missing participant with clear not-found response.")
    passed_count += 1

    print("\n[TEST 6] Query: 'Send Anjali a follow-up email about the database migration.'")
    test_missing_info_email_anjali(agent)
    print(">>> TEST 6 PASSED: Handled missing info email request with 'No relevant meeting information found. Email was not sent'.")
    passed_count += 1

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {passed_count}/6 TEST CASES PASSED SUCCESSFULLY (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
