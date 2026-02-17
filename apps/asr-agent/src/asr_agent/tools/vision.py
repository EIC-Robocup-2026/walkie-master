import os  # เพิ่ม os
import time

import cv2
import numpy as np
from langchain_core.tools import tool
from walkie_db.agent_integration import AgentIntegration
from walkie_vision.detector import VisionDetector

from asr_agent.tools.robot import get_bot

# Singleton instances
_v_detector = None
_last_captured_frame = None
DEBUG_IMAGE_PATH = "debug_sim_view.jpg"  # กำหนด Path ให้ตรงกับ Test


def get_detector():
    global _v_detector
    if _v_detector is None:
        _v_detector = VisionDetector()
    return _v_detector


@tool
def get_current_view() -> str:
    """
    Captures the current frame from the robot's camera via Walkie-SDK.
    Includes retries for Zenoh discovery.
    """
    global _last_captured_frame
    try:
        bot = get_bot()
        if bot.camera is None:
            return "Error: Camera module is not available."

        # ✅ เพิ่มระบบ Retry สำหรับ Zenoh (ลอง 5 ครั้ง ทุกๆ 0.5 วินาที)
        max_retries = 5
        frame = None

        print(f"📸 Attempting to capture frame via {bot.camera_protocol}...")
        for i in range(max_retries):
            frame = bot.camera.get_frame()
            if frame is not None and frame.size > 0:
                break
            print(f"⏳ Frame is empty, retrying ({i + 1}/{max_retries})...")
            time.sleep(0.5)

        if frame is not None:
            _last_captured_frame = frame
            cv2.imwrite(DEBUG_IMAGE_PATH, frame)
            return f"Successfully captured a new frame and saved to {DEBUG_IMAGE_PATH}"

        return (
            "Error: Empty frame received from camera after retries. Check Zenoh Bridge."
        )
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
