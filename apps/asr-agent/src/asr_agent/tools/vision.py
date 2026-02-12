from typing import Any, Dict, Optional

from langchain_core.tools import tool
from walkie_vision.detector import ObjectDetector

# โหลดโมเดล Vision ไว้ที่ Global เพื่อประหยัดเวลาโหลดใหม่ในแต่ละรอบ
detector = ObjectDetector()


@tool
def observe_scene(focus_object: Optional[str] = "all") -> Dict[str, Any]:
    """
    Scans the current camera feed to detect objects and generate semantic descriptions.
    Use this when you need to find an object, describe the room, or identify surroundings.
    """
    # เรียกใช้ Perception Unit เพื่อเปลี่ยนภาพสดเป็นข้อมูลเชิงโครงสร้าง
    # ตัวอย่างผลลัพธ์: {'objects': ['cup', 'table'], 'captions': ['a blue mug on a table']}
    results = detector.detect_and_caption(focus_object=focus_object)

    # ส่งข้อมูลแบบ JSON ที่พร้อมให้ LLM (Qwen) ประมวลผลต่อ
    return {
        "detected_objects": results.get("objects", []),
        "semantic_descriptions": results.get("captions", []),
        "confidence_scores": results.get("confidences", []),
    }
