from langchain_core.tools import tool
from walkie_sdk.robot import WalkieRobot

# สร้าง Global Instance หรือใช้ผ่าน Dependency Injection
# ในการแข่งขันจริง แนะนำให้รัน singleton เชื่อมต่อไว้ตลอด
bot = WalkieRobot(ip="127.0.0.1")


@tool
def move_to_coordinates(x: float, y: float, heading: float = 0.0) -> str:
    """
    Move the Walkie robot to a specific (x, y) coordinate on the map.
    Use this tool when the user provides a destination or when an object is located.
    """
    # สั่งการผ่านโมดูล Navigation
    status = bot.nav.go_to(x=x, y=y, heading=heading)
    return f"Navigation result: {status}"


@tool
def get_robot_status() -> dict:
    """
    Retrieve the current pose (x, y, heading) and connection status of the robot.
    Use this to verify where the robot is before or after a move.
    """
    # ดึงพิกัดจาก Telemetry
    pose = bot.status.get_pose()
    return {"pose": pose, "is_connected": bot.is_connected}
