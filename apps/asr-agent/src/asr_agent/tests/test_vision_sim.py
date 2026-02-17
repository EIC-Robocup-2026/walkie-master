import os
import time

import cv2
import pytest

from asr_agent.core.orchestrator import AgentOrchestrator
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view

DEBUG_IMAGE_PATH = "debug_sim_view.jpg"


def test_vision_end_to_end_with_sim():
    # --- STEP 0: Cleanup ---
    if os.path.exists(DEBUG_IMAGE_PATH):
        os.remove(DEBUG_IMAGE_PATH)

    # 1. สร้าง Orchestrator
    # ตรวจสอบว่าใน __init__ ของ Orchestrator ตั้งค่าเป็น Zenoh 7447 หรือยัง
    orchestrator = AgentOrchestrator()

    # รอให้ Zenoh เชื่อมต่อและเริ่ม Streaming (สำคัญมากสำหรับ Zenoh)
    print("⏳ Waiting for Zenoh discovery...")
    time.sleep(2)

    user_command = "Walkie, look at the table in front of you and tell me what you see."
    print(f"\n🚀 [Command]: {user_command}")

    # --- STEP 1: Planning ---
    response = orchestrator.run_command(user_command)
    tool_calls = orchestrator.client.parse_tool_calls(response)

    assert len(tool_calls) >= 2, (
        "AI failed to plan the vision pipeline (Capture -> Analyze)"
    )

    # --- STEP 2: Capture (Execution) ---
    print("\n📸 Capturing frame from Gazebo via Zenoh (Port 7447)...")
    capture_result = get_current_view.invoke({})

    if "Error" in capture_result:
        pytest.fail(f"Zenoh Camera Capture Failed: {capture_result}")

    # --- STEP 3: Visualization Check ---
    assert os.path.exists(DEBUG_IMAGE_PATH), "Image file was not saved!"
    img = cv2.imread(DEBUG_IMAGE_PATH)
    assert img is not None, "Saved image is corrupted"

    h, w, _ = img.shape
    print(f"🖼️  Real-time Frame captured: {w}x{h}")

    # เพิ่มส่วนนี้ถ้าต้องการ "เห็น" ภาพจริงๆ ระหว่างเทส (จะโชว์ 2 วินาทีแล้วปิด)
    if os.environ.get("DISPLAY"):  # เช็คว่ามีหน้าจอให้โชว์ไหม
        cv2.imshow("Test Vision View", img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    # --- STEP 4: Analysis ---
    print("\n🔍 Running YOLO Analysis (on RTX 5090)...")
    analysis_result = analyze_and_store_objects.invoke({})
    print(f"📊 [Analysis Output]: {analysis_result}")

    assert "stored" in analysis_result.lower()
    print("\n✨ Vision Pipeline is working perfectly with Zenoh & Live Sim!")
