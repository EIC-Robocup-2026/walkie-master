import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel
from langchain_core.messages import AIMessage


class QwenAgentClient:
    """
    Client สำหรับจัดการการสื่อสารกับ Qwen3 8B (FC)
    โดยเน้นความนิ่งของผลลัพธ์ (Deterministic) และการทำ Function Calling
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-walkie-master",
        model_name: str = "qwen3-8b"
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # ล็อคพารามิเตอร์เพื่อให้ผลลัพธ์นิ่งที่สุดสำหรับงาน Robot Control
        self.default_params = {
            "temperature": 0.0,  # ลดความสุ่มให้เป็นศูนย์
            "top_p": 0.1,
            "seed": 42,          # กำหนด Seed เพื่อให้ผลลัพธ์เดิมเสมอในการเทส
            "max_tokens": 1024,
        }

    def generate_response(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        ส่ง Message ไปยัง LLM และจัดการแปลง Format ระหว่าง LangChain และ OpenAI
        """
        # --- แปลง LangChain Message เป็น OpenAI Format ---
        formatted_messages = []
        for m in messages:
            # ถ้าเป็น LangChain Message Object (HumanMessage, AIMessage, etc.)
            if hasattr(m, "type"):
                role = "assistant" if m.type == "ai" else m.type
                # แปลง 'human' เป็น 'user' ให้ตรงมาตรฐาน OpenAI API
                role = "user" if role == "human" else role
                formatted_messages.append({"role": role, "content": m.content})
            # ถ้าเป็น Dictionary ปกติ
            elif isinstance(m, dict):
                formatted_messages.append(m)
            else:
                # กรณีเป็น String หรือ Type อื่นๆ ที่หลุดมา
                formatted_messages.append({"role": "user", "content": str(m)})

        try:
            # vLLM/OpenAI จะบ่นถ้าส่ง tool_choice="auto" มาทั้งที่ไม่มี tools
            actual_tool_choice = tool_choice if tools else None

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                tools=tools,
                tool_choice=actual_tool_choice,
                **self.default_params,
            )
            return response.choices[0].message

        except Exception as e:
            print(f"❌ LLM Client Error: {e}")
            # คืนค่าเป็น AIMessage เพื่อให้ LangGraph State ยังทำงานต่อไปได้โดยไม่ Crash
            return AIMessage(
                content=f"Error: I encountered an issue with the LLM server: {str(e)}"
            )

    def parse_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        """
        แปลงผลลัพธ์จาก LLM ให้เป็นรายการการเรียกใช้ Tool ที่อ่านง่าย
        """
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []

        calls = []
        for tool_call in message.tool_calls:
            try:
                calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                )
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse arguments for tool: {tool_call.function.name}")
                continue

        return calls
