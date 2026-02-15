import pytest

from asr_agent.core.client import QwenAgentClient


def test_llm_server_ping():
    """ตรวจสอบว่า Server พอร์ต 8000 พร้อมทำงาน และแสดงข้อความตอบโต้"""
    client = QwenAgentClient(base_url="http://localhost:8000/v1")

    # 1. เตรียมข้อความที่ส่งไป
    msg = [{"role": "user", "content": "ping"}]
    print(f"\n[User]: {msg[0]['content']}")

    # 2. รับการตอบกลับจาก LLM
    response = client.generate_response(msg)

    # 3. แสดงผลลัพธ์ที่ได้
    if response and hasattr(response, "content"):
        print(f"[LLM]: {response.content}")
    else:
        print("[LLM]: No content returned or error occurred.")

    # Assertions
    assert response is not None
    assert hasattr(response, "content")
