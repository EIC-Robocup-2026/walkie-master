import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel


class QwenAgentClient:
    """
    Client สำหรับจัดการการสื่อสารกับ Qwen3 8B (FC)
    โดยเน้นความนิ่งของผลลัพธ์ (Deterministic) และการทำ Function Calling
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-walkie-master",
        model_name: str = "qwen3-8b"  # ปรับให้ตรงกับชื่อที่ตั้งใน serve_llm.sh
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # ล็อคพารามิเตอร์เพื่อให้ผลลัพธ์นิ่งที่สุดสำหรับงาน Robot Control
        self.default_params = {
            "temperature": 0.0,  # ลดความสุ่มให้เป็นศูนย์
            "top_p": 0.1,
            "seed": 42,  # กำหนด Seed เพื่อให้ผลลัพธ์เดิมเสมอในการเทส
            "max_tokens": 1024,
        }

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        ส่ง Prompt ไปยัง Qwen และรับผลลัพธ์กลับมา
        รองรับทั้งข้อความทั่วไปและการเรียกใช้ Tools
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                **self.default_params,
            )
            return response.choices[0].message
        except Exception as e:
            print(f"❌ LLM Client Error: {e}")
            return None

    def parse_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        """แปลงผลลัพธ์จาก LLM ให้เป็นรายการการเรียกใช้ Tool ที่อ่านง่าย"""
        if not message.tool_calls:
            return []

        calls = []
        for tool_call in message.tool_calls:
            calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )
        return calls
