import glob
import os
from unittest.mock import MagicMock, patch

import cv2
import pytest

from asr_agent.core.orchestrator import AgentOrchestrator
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view


def test_vision_pipeline_via_orchestrator():
    """
    ทดสอบการวางแผน (Reasoning) ของ Orchestrator
    ตรวจสอบว่า AI สั่ง Capture และ Analyze พร้อมกันตาม SOP หรือไม่
    """
    # 1. ค้นหารูปภาพ (ใช้เพียงเพื่อเช็คบริบทถ้าต้องการ)
    scene_dir = "data/images/scenes"
    image_list = glob.glob(os.path.join(scene_dir, "*.[jJ][pP][gG]"))
    if not image_list:
        pytest.skip(f"No images found in {scene_dir}")

    # 2. Mock ระบบภายนอก (เพื่อไม่ให้ Robot ต่อจริงตอนเรียก Orchestrator)
    with (
        patch("asr_agent.tools.robot.WalkieRobot"),
        patch("asr_agent.tools.vision.VisionDetector"),
        patch("asr_agent.tools.vision.AgentIntegration"),
    ):
        # 3. สร้าง Orchestrator
        orchestrator = AgentOrchestrator()

        # 4. ส่งคำสั่งผ่าน Orchestrator
        user_text = "Look at the table and tell me what you see."
        response = orchestrator.run_command(user_text)

        # 5. ตรวจสอบการตัดสินใจ (Planning Verification)
        # ตรวจสอบว่า Qwen3-8B เข้าใจ SOP (Capture -> Analyze) ไหม
        tool_calls = orchestrator.client.parse_tool_calls(response)

        print(f"\n[AI Thought]: {response.content}")
        print(f"[Planned Tools]: {[c['name'] for c in tool_calls]}")

        # ตรวจสอบว่ามี 2 ขั้นตอนตามกฎ RoboCup ของเรา
        assert len(tool_calls) >= 2
        assert tool_calls[0]["name"] == "get_current_view"
        assert tool_calls[1]["name"] == "analyze_and_store_objects"

        # ยืนยันว่ายังไม่มีการรันจริง (DB ต้องยังไม่ถูกเรียก)
        # (หมายเหตุ: ถ้าคุณ patch AgentIntegration ไว้ ด้านล่างนี้จะเป็นการเช็คว่าไม่มีการเรียก)
