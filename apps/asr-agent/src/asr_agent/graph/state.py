from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    โครงสร้างสถานะของ Walkie Agent ใน LangGraph
    """

    # เก็บประวัติการสนทนาและ Tool Calls ทั้งหมด
    messages: Annotated[List[BaseMessage], add_messages]

    # ข้อมูลตำแหน่งปัจจุบันของหุ่นยนต์จาก walkie-sdk
    current_pose: Dict[str, float]

    # ผลลัพธ์จากการประมวลผล Vision ล่าสุด
    last_observation: Dict[str, Any]

    # สถานะภารกิจ (เช่น 'searching', 'moving', 'arrived')
    mission_status: str
