import cv2
import numpy as np
from langchain_core.tools import tool
from walkie_db.agent_integration import AgentIntegration
from walkie_vision.detector import VisionDetector

from asr_agent.tools.robot import get_bot

# Singleton instances
_v_detector = None
_last_captured_frame = None


def get_detector():
    """Lazy initialization for VisionDetector."""
    global _v_detector
    if _v_detector is None:
        _v_detector = VisionDetector()
    return _v_detector


@tool
def get_current_view() -> str:
    """
    Captures the current frame from the robot's camera via Walkie-SDK.
    """
    global _last_captured_frame
    try:
        bot = get_bot()
        if bot.camera is None:
            return "Error: Camera module is not available."

        # ดึงภาพจาก SDK (BGR numpy array)
        frame = bot.camera.get_frame()

        if frame is not None:
            _last_captured_frame = frame
            return "Successfully captured a new frame from the camera."
        return "Error: Empty frame received from camera."
    except Exception as e:
        return f"Error capturing frame: {str(e)}"


@tool
def analyze_and_store_objects() -> str:
    """
    Analyzes the last captured frame and stores objects in WalkieDB.
    """
    global _last_captured_frame
    if _last_captured_frame is None:
        return "Error: No frame captured. Please call get_current_view first."

    detector = get_detector()
    detections = detector.get_segmented_objects(_last_captured_frame)

    if not detections:
        return "No objects detected in the current view."

    db = AgentIntegration(base_db_path="data/walkie_memory")
    for i, item in enumerate(detections):
        db.process_object_detection(
            object_id=f"{item['yolo_class']}_{i}",
            xyz=[0.0, 0.0, 0.0],
            embedding=[0.0] * 512,
            label=item["yolo_class"],
        )
    return f"Successfully detected and stored {len(detections)} objects in memory."
