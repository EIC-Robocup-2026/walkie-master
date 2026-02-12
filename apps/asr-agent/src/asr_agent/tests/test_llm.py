import pytest
from asr_agent.core.client import QwenAgentClient

def test_llm_server_ping():
    """ตรวจสอบว่า Server พอร์ต 8000 พร้อมทำงาน"""
    client = QwenAgentClient(base_url="http://localhost:8000/v1")
    msg = [{"role": "user", "content": "ping"}]
    response = client.generate_response(msg)
    assert response is not None
    assert hasattr(response, "content")
