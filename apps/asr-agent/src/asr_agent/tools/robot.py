from langchain_core.tools import tool
from walkie_sdk.robot import WalkieRobot

# 1. เปลี่ยนจากสร้างทันที เป็นตัวแปรว่างไว้ก่อน
_bot = None


def get_bot():
    """ฟังก์ชันสำหรับดึง instance หุ่นยนต์ (สร้างเมื่อเรียกใช้ครั้งแรก)"""
    global _bot
    if _bot is None:
        # จะเชื่อมต่อก็ต่อเมื่อมีการเรียกใช้ tool จริงๆ เท่านั้น
        _bot = WalkieRobot(ip="127.0.0.1")
    return _bot


@tool
def move_to_coordinates(x: float, y: float, heading: float = 0.0) -> str:
    """...คำอธิบายเดิม..."""
    # 2. เรียกใช้ผ่าน get_bot()
    bot = get_bot()
    status = bot.nav.go_to(x=x, y=y, heading=heading)
    return f"Navigation result: {status}"


@tool
def get_robot_status() -> dict:
    """...คำอธิบายเดิม..."""
    # 2. เรียกใช้ผ่าน get_bot()
    bot = get_bot()
    pose = bot.status.get_pose()
    return {"pose": pose, "is_connected": bot.is_connected}
