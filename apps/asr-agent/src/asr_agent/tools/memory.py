from typing import List, Optional

from langchain_core.tools import tool
from walkie_db.agent_integration import AgentIntegration

# Singleton instance for memory management
_memory_manager = None


def get_memory():
    """Initialize AgentIntegration only when needed."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AgentIntegration(base_db_path="data/chromadb")
    return _memory_manager


@tool
def remember_entity(object_id: str, label: str, caption: str, xyz: List[float]) -> str:
    """
    Saves an object's information (ID, label, description, and location) into long-term memory.
    Use this after detecting a new important object or when requested to 'remember' something.
    """
    memory = get_memory()
    memory.process_object_detection(
        object_id=object_id,
        xyz=xyz,
        embedding=[0.0] * 512,  # Actual CLIP embedding would be used in production
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
    memory = get_memory()
    # In production, this would involve embedding the text query
    hits = memory.find_objects(query_emb=[0.0] * 512, n=3)
    if not hits:
        return "I couldn't find any matching information in my memory."
    return f"I found the following matches: {str(hits)}"
