from typing import Any, Dict, List

from asr_agent.core.client import QwenAgentClient
from asr_agent.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from asr_agent.tools.db_tools import DB_TOOL_SCHEMA
from asr_agent.tools.robot_tools import ROBOT_TOOL_SCHEMA
from asr_agent.tools.vision_tools import VISION_TOOL_SCHEMA


class AgentOrchestrator:
    def __init__(self):
        self.client = QwenAgentClient()
        # รวบรวม Schemas ทั้งหมดส่งให้ LLM
        self.tools_schema = [VISION_TOOL_SCHEMA, DB_TOOL_SCHEMA, ROBOT_TOOL_SCHEMA]
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # เพิ่มตัวอย่างการคิดเพื่อให้ผลลัพธ์นิ่ง (Deterministic)
        self.messages.extend(FEW_SHOT_EXAMPLES)

    def run_command(self, user_text: str):
        """
        รับคำสั่ง Text (จาก ASR หรือ Prompt) แล้วตัดสินใจทำงาน
        """
        self.messages.append({"role": "user", "content": user_text})

        # 1. LLM ตัดสินใจ (Thought + Tool Call)
        response = self.client.generate_response(self.messages, tools=self.tools_schema)

        if response.content:
            print(f"🧠 Agent Thought: {response.content}")

        # 2. ตรวจสอบการเรียกใช้ Tool
        tool_calls = self.client.parse_tool_calls(response)

        for call in tool_calls:
            print(f"🛠️ Executing Tool: {call['name']} with {call['arguments']}")
            # ในส่วนนี้คุณจะเขียน Logic ในการเรียกใช้ Class Tools จริงๆ
            # และส่งผลลัพธ์กลับไปให้ LLM เพื่อสรุปคำตอบ

        return response
