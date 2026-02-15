from typing import Any, Dict, List

from langchain_core.utils.function_calling import convert_to_openai_tool

from asr_agent.core.client import QwenAgentClient
from asr_agent.prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from asr_agent.tools.memory import remember_entity, search_memory

# Import Tools เพื่อดึง Schema เท่านั้น
from asr_agent.tools.robot import get_robot_status, move_to_coordinates
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view


class AgentOrchestrator:
    def __init__(self):
        self.client = QwenAgentClient()

        # เตรียม Tools ทั้งหมดเพื่อส่ง Schema ให้ LLM
        raw_tools = [
            move_to_coordinates,
            get_robot_status,
            get_current_view,
            analyze_and_store_objects,
            remember_entity,
            search_memory,
        ]
        # แปลงเป็น Schema อัตโนมัติ (Dynamic Schema)
        self.tools_schema = [convert_to_openai_tool(t) for t in raw_tools]

        # ตั้งค่าพื้นฐานและกฎการทำงาน (SOP)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.extend(FEW_SHOT_EXAMPLES)

    def run_command(self, user_text: str):
        """
        AI วางแผนงานจากคำสั่งของผู้ใช้ แต่ยังไม่ดำเนินการรัน Tool จริง
        """
        self.messages.append({"role": "user", "content": user_text})

        # ให้ LLM สร้างแผนงาน (Thought + Tool Calls)
        response = self.client.generate_response(self.messages, tools=self.tools_schema)

        # เก็บการตัดสินใจไว้ในประวัติ (History)
        self.messages.append(response)

        return response  # คืนค่า AI Response Object กลับไปให้ผู้ใช้ตรวจสอบ
