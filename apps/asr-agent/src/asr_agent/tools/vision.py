from typing import Optional, Dict, Any
import numpy as np
from langchain_core.tools import tool
from walkie_vision.detector import VisionDetector  # แก้เป็นชื่อที่ถูกต้อง

# สร้าง Instance โดยใช้ชื่อคลาสเดิมของคุณ
detector = VisionDetector()

@tool
def observe_scene(focus_object: Optional[str] = "all") -> Dict[str, Any]:
    """
    Scans the current camera feed to detect objects.
    """
    # จำลองการรับ Frame (ในงานจริงต้องดึงจาก Camera Bridge)
    # เราจะใช้ frame ว่างๆ สำหรับการทดสอบเบื้องต้น
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # เรียกใช้ Method ที่มีอยู่จริงใน VisionDetector ของคุณ
    detected_items = detector.get_segmented_objects(dummy_frame)

    objects_found = [item["yolo_class"] for item in detected_items]

    # กรองวัตถุตามคำสั่ง
    if focus_object != "all":
        objects_found = [obj for obj in objects_found if focus_object.lower() in obj.lower()]

    return {
        "detected_objects": objects_found,
        "semantic_descriptions": [f"I see a {obj}" for obj in objects_found]
    }
