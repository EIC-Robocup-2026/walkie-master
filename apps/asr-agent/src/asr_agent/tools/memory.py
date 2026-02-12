from typing import List, Optional

from langchain_core.tools import tool
from walkie_db.agent_integration import AgentIntegration

# สร้างอินสแตนซ์สำหรับจัดการฐานข้อมูลความจำ
memory_manager = AgentIntegration(base_db_path="data/chromadb")


@tool
def remember_entity(object_id: str, label: str, caption: str, xyz: List[float]) -> str:
    """
    Saves an object's information (ID, label, description, and location) into long-term memory.
    Use this after detecting a new important object or when requested to 'remember' something.
    """
    # บันทึกวัตถุที่ผ่านการประมวลผลจาก Vision ลงในฐานข้อมูล
    memory_manager.process_object_detection(
        object_id=object_id,
        xyz=xyz,
        embedding=[0.0] * 512,  # ในงานจริงจะใช้ CLIP embedding จาก vision
        label=label,
        caption=caption,
    )
    return f"Successfully remembered '{label}' (ID: {object_id}) at location {xyz}."


@tool
def search_memory(query: str) -> str:
    """
    Searches for objects or past experiences in the robot's memory using a text query.
    Use this to answer questions like 'Where did you see my mug?' or 'Who is here?'.
    """
    # ค้นหาวัตถุหรือบุคคลจากฐานข้อมูล Vector
    # ในการทำงานจริงจะมีการทำ Embedding ของ query ก่อนค้นหา
    hits = memory_manager.find_objects(query_emb=[0.0] * 512, n=3)
    if not hits:
        return "I couldn't find any matching information in my memory."
    return f"I found the following matches: {str(hits)}"
