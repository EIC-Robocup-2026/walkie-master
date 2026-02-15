import pytest
from walkie_sdk.robot import WalkieRobot


def test_sdk_connection():
    """ตรวจสอบการสื่อสารกับตัวหุ่นยนต์ (หรือ Mock Server)"""
    bot = WalkieRobot(ip="127.0.0.1")
    try:
        assert bot.is_connected, "Cannot connect to Robot Base"
    finally:
        bot.disconnect()


# def test_arm_status():
#     """ตรวจสอบว่าอ่านสถานะแขนกลได้ (ถ้าต่ออยู่)"""
#     bot = WalkieRobot(ip="127.0.0.1")
#     if bot.is_connected:
#         status = bot.arm.get_status()
#         assert status is not None
#         bot.disconnect()
