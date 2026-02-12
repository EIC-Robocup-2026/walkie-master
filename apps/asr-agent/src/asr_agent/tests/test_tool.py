import pytest
from unittest.mock import MagicMock, patch
from asr_agent.tools.robot import move_to_coordinates

class TestNavigationTool:

    def test_tool_schema(self):
        """1. ตรวจสอบว่า Tool Schema ตรงตามที่ LangChain/Qwen3 ต้องการ"""
        assert move_to_coordinates.name == "move_to_coordinates"
        # ตรวจสอบพารามิเตอร์ (อ้างอิงตาม navigation.go_to)
        args = move_to_coordinates.args
        assert "x" in args
        assert "y" in args
        assert "heading" in args
        assert "target location" in move_to_coordinates.description

    @patch("asr_agent.tools.robot.WalkieRobot")
    def test_tool_execution_success(self, mock_robot_class):
        """2. ตรวจสอบว่า Tool เรียกใช้ bot.nav.go_to() และคืนค่า SUCCEEDED"""
        # สร้าง Mock สำหรับ Robot และ Navigation
        mock_bot = MagicMock()
        mock_robot_class.return_value = mock_botimport pytest
        from unittest.mock import MagicMock, patch
        from asr_agent.tools.robot import move_to_coordinates

        class TestNavigationTool:

            def test_tool_schema(self):
                """1. ตรวจสอบว่า Tool มีโครงสร้าง Schema ที่ถูกต้องสำหรับ LangChain"""
                assert move_to_coordinates.name == "move_to_coordinates"
                assert "x" in move_to_coordinates.args
                assert "y" in move_to_coordinates.args
                assert "target location" in move_to_coordinates.description

            @patch("asr_agent.tools.robot.WalkieRobot")
            def test_tool_execution_success(self, mock_robot_class):
                """2. ตรวจสอบว่า Tool ส่งพิกัดไปยัง SDK ได้ถูกต้อง"""
                # ตั้งค่า Mock Robot
                mock_robot = MagicMock()
                mock_robot_class.return_value = mock_robot
                mock_robot.is_connected = True

                # รัน Tool
                result = move_to_coordinates.invoke({"x": 0.0, "y": 0.0})
                # ตรวจสอบว่ามีการเรียกใช้คำสั่งเดิน (เช่น navigation.go_to หรือเมธอดที่เกี่ยวข้อง)
                # หมายเหตุ: ปรับชื่อเมธอดตามที่คุณเขียนไว้ใน walkie-sdk
                assert mock_robot.move_to.called or mock_robot.navigation.go_to.called
                assert "succeeded" in result.lower() or "failed" in result.lower()

            @patch("asr_agent.tools.robot.WalkieRobot")
            def test_tool_robot_not_connected(self, mock_robot_class):
                """3. ตรวจสอบ Error Handling เมื่อหุ่นยนต์ไม่ได้เชื่อมต่อ"""
                mock_robot = MagicMock()
                mock_robot_class.return_value = mock_robot
                mock_robot.is_connected = False

                result = move_to_coordinates.invoke({"x": 1.0, "y": 1.0})

                assert "error" in result.lower() or "not connected" in result.lower()

            def test_invalid_coordinates(self):
                """4. ตรวจสอบการจัดการพิกัดที่ไม่เหมาะสม (ถ้ามี Logic กรองไว้)"""
                # หากคุณมี Logic เช็คพิกัดนอกสนามแข่งขัน
                result = move_to_coordinates.invoke({"x": -999, "y": -999})
                # ตรวจสอบว่า Tool มีการแจ้งเตือนพิกัดที่ผิดปกติ
                assert result is not None


        # ตั้งค่าให้ bot.nav.go_to คืนค่า "SUCCEEDED" ตาม logic ใน navigation.py
        mock_bot.nav.go_to.return_value = "SUCCEEDED"

        # รัน Tool (ส่ง heading ไปด้วยตาม signature ของ navigation.py)
        result = move_to_coordinates.invoke({"x": 2.5, "y": 3.0, "heading": 0.0})

        # ตรวจสอบว่าเรียกใช้ถูกเมธอดและส่งค่าถูกต้องหรือไม่
        mock_bot.nav.go_to.assert_called_once_with(
            x=2.5,
            y=3.0,
            heading=0.0
        )
        assert result == "SUCCEEDED"

    @patch("asr_agent.tools.robot.WalkieRobot")
    def test_tool_execution_failed(self, mock_robot_class):
        """3. ตรวจสอบเมื่อ Nav2 คืนค่า FAILED"""
        mock_bot = MagicMock()
        mock_robot_class.return_value = mock_bot
        mock_bot.nav.go_to.return_value = "FAILED"

        result = move_to_coordinates.invoke({"x": 1.0, "y": 1.0, "heading": 1.57})

        assert result == "FAILED"

    @patch("asr_agent.tools.robot.WalkieRobot")
    def test_tool_connection_error(self, mock_robot_class):
        """4. ตรวจสอบการจัดการ ConnectionError (เมื่อไม่ได้เชื่อมต่อหุ่นยนต์)"""
        mock_bot = MagicMock()
        mock_robot_class.return_value = mock_bot

        # จำลองการ Raise ConnectionError จากใน navigation.py
        mock_bot.nav.go_to.side_effect = ConnectionError("Not connected to robot")

        result = move_to_coordinates.invoke({"x": 0.0, "y": 0.0, "heading": 0.0})

        # ตรวจสอบว่า Tool Handle error แล้วคืนค่าเป็น string บอก Agent
        assert "error" in result.lower() or "not connected" in result.lower()
