from typing import Any, Dict, List

from langchain_core.utils.function_calling import convert_to_openai_tool

from asr_agent.core.client import QwenAgentClient
from asr_agent.prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from asr_agent.tools.memory import remember_entity, search_memory

# 1. Import WalkieRobot มาจัดการ Connection ที่นี่ที่เดียว
from asr_agent.tools.robot import WalkieRobot, get_robot_status, move_to_coordinates
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view


class AgentOrchestrator:
    def __init__(self):
        self.client = QwenAgentClient()

        # 2. สร้าง Shared Robot Instance ผ่าน Zenoh 7447
        # การสร้างที่นี่จะทำให้ Session ของ Zenoh ถูกเปิดทิ้งไว้ตลอดอายุของ Orchestrator
        print("🌐 Connecting to Walkie via Zenoh (Port 7447)...")
        self.bot = WalkieRobot(ip="127.0.0.1")

        # 3. เตรียมเครื่องมือ
        raw_tools = [
            move_to_coordinates,
            get_robot_status,
            get_current_view,
            analyze_and_store_objects,
            remember_entity,
            search_memory,
        ]

        self.tools_schema = [convert_to_openai_tool(t) for t in raw_tools]

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.extend(FEW_SHOT_EXAMPLES)

    def run_command(self, user_text: str):
        """
        AI generates a plan.
        """
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.generate_response(self.messages, tools=self.tools_schema)
        self.messages.append(response)
        return response

    def disconnect(self):
        """ปิดการเชื่อมต่อเมื่อจบงาน"""
        if hasattr(self, "bot"):
            self.bot.disconnect()
            print("🔌 Zenoh disconnected.")
