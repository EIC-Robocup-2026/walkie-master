from typing import Any, Dict

from asr_agent.core.client import QwenAgentClient
from asr_agent.graph.state import AgentState
from asr_agent.tools.memory import remember_entity, search_memory
from asr_agent.tools.robot import get_robot_status, move_to_coordinates
from asr_agent.tools.vision import observe_scene
from langchain_core.messages import ToolMessage

# รวบรวม Tools ทั้งหมดเข้าด้วยกัน
tools_list = [
    move_to_coordinates,
    get_robot_status,
    observe_scene,
    remember_entity,
    search_memory,
]
tools_dict = {tool.name: tool for tool in tools_list}

# สร้าง Client สำหรับเรียก Qwen3 8B (FC)
client = QwenAgentClient()


def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Node สำหรับให้ LLM ตัดสินใจจาก State ปัจจุบัน
    """
    # ดึง Schema ของ Tools ทั้งหมดส่งให้โมเดล
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.schema()
                if hasattr(t, "args_schema")
                else {},
            },
        }
        for t in tools_list
    ]

    response = client.generate_response(state["messages"], tools=tools_schema)
    return {"messages": [response]}


def execute_tools(state: AgentState) -> Dict[str, Any]:
    """
    Node สำหรับรันเครื่องมือที่ LLM ร้องขอ
    """
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # รัน Tool จริงผ่านระบบของ Walkie
        result = tools_dict[tool_name].invoke(tool_args)

        # สร้าง ToolMessage เพื่อส่งผลลัพธ์กลับไปให้ LLM
        tool_messages.append(
            ToolMessage(tool_call_id=tool_call["id"], content=str(result))
        )

    return {"messages": tool_messages}
