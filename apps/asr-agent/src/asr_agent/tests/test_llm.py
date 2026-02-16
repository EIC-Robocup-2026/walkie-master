import pytest

from asr_agent.core.client import QwenAgentClient


def test_llm_server_ping():
    """Verify that the server on port 8000 is ready and responding."""
    client = QwenAgentClient(base_url="http://localhost:8000/v1")

    # 1. Prepare the outgoing message
    msg = [{"role": "user", "content": "ping"}]
    print(f"\n[User]: {msg[0]['content']}")

    # 2. Receive the response from the LLM
    response = client.generate_response(msg)

    # 3. Display the results
    if response and hasattr(response, "content"):
        print(f"[LLM]: {response.content}")
    else:
        print("[LLM]: No content returned or error occurred.")

    # Assertions
    assert response is not None
    assert hasattr(response, "content")
