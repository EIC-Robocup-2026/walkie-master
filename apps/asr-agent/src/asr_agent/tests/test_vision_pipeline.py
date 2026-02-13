import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from asr_agent.tools.vision import get_current_view, analyze_and_store_objects

class TestVisionDatabasePipeline:

    @patch("asr_agent.tools.vision.get_frame")
    def test_full_vision_to_db_flow(self, mock_get_frame):
        """
        ทดสอบ Flow: ถ่ายภาพ -> ตรวจจับ -> บันทึกลง DB
        """
        # 1. Mock ภาพจาก Gazebo (สร้างภาพที่มีสีเข้มๆ ให้ YOLO พอเห็นเป็นโครง)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_get_frame.return_value = dummy_img

        # 2. ขั้นตอนแรก: Agent สั่งถ่ายภาพ
        capture_result = get_current_view.invoke({})
        assert "Successfully captured" in capture_result

        # 3. ขั้นตอนที่สอง: Agent สั่งวิเคราะห์และลง DB
        # เราจะ Mock VisionDetector เพื่อให้คืนค่าว่าเจอ 'mug' เสมอ
        with patch("asr_agent.tools.vision.v_detector.get_segmented_objects") as mock_detect:
            mock_detect.return_value = [{"yolo_class": "mug", "bbox": [0,0,10,10]}]

            with patch("asr_agent.tools.vision.AgentIntegration") as mock_db_class:
                mock_db = MagicMock()
                mock_db_class.return_value = mock_db

                db_result = analyze_and_store_objects.invoke({})

                # ตรวจสอบว่ามีการเรียกบันทึกข้อมูลจริงไหม
                assert "stored 1 objects" in db_result
                assert mock_db.process_object_detection.called

    def test_analyze_without_frame_fails(self):
        """ตรวจสอบว่าถ้ายังไม่ถ่ายภาพ จะสั่งวิเคราะห์ไม่ได้"""
        # รีเซ็ตค่า global (ในเทสจริงต้องระวังเรื่องสถานะค้าง)
        with patch("asr_agent.tools.vision.last_captured_frame", None):
            result = analyze_and_store_objects.invoke({})
            assert "Error: No frame captured" in result
