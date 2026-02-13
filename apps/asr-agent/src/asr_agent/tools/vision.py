import cv2
import numpy as np
from langchain_core.tools import tool
from walkie_vision.detector import VisionDetector
from walkie_db.agent_integration import AgentIntegration

# Initialize modules
v_detector = VisionDetector()
# สมมติว่ามีฟังก์ชัน get_frame() อยู่ใน walkie_vision.camera_sim
from walkie_vision.camera_sim import get_frame

# ตัวแปรสำหรับเก็บภาพล่าสุดในหน่วยความจำของ Agent (Shared Buffer)
last_captured_frame = None

@tool
def get_current_view() -> str:
    """
    Captures the current frame from the robot's camera (Gazebo Simulation).
    """
    global last_captured_frame
    try:
        frame = get_frame() # ดึงภาพจาก simulator
        if frame is not None:
            last_captured_frame = frame
            return "Successfully captured a new frame from the camera."
        return "Error: Received empty frame from simulator."
    except Exception as e:
        return f"Error capturing frame: {str(e)}"

@tool
def analyze_and_store_objects() -> str:
    """
    Analyzes the last captured frame, detects objects, and saves their
    embeddings and positions into the WalkieDB database.
    """
    global last_captured_frame
    if last_captured_frame is None:
        return "Error: No frame captured yet. Please call get_current_view first."

    # 1. รัน YOLO Detection
    detections = v_detector.get_segmented_objects(last_captured_frame)
    if not detections:
        return "Scene analyzed: No objects detected in the current view."

    # 2. เชื่อมต่อ Database และบันทึก
    db = AgentIntegration(base_db_path="data/walkie_memory")

    count = 0
    for item in detections:
        # สมมติค่า xyz และ embedding สำหรับทดสอบ (ในงานจริงจะได้จาก Vision Pipeline)
        db.process_object_detection(
            object_id=f"{item['yolo_class']}_{count}",
            xyz=[0.0, 0.0, 0.0], # ค่าตำแหน่งจำลอง
            embedding=[0.0] * 512, # ค่า Featureจำลอง
            label=item['yolo_class']
        )
        count += 1

    return f"Successfully detected and stored {count} objects in the database."
