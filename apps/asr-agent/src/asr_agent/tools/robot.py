import time

from langchain_core.tools import tool
from walkie_sdk.robot import WalkieRobot

# Singleton instance for the robot connection
_bot = None


def get_bot():
    """
    Initialize and return the WalkieRobot instance using lazy loading.
    Configured specifically for Zenoh protocol on port 7447.
    """
    global _bot
    if _bot is None:
        print("🤖 Initializing Shared Zenoh Robot Instance (Port 7447)...")
        # บังคับใช้ Zenoh ทั้ง ROS และ Camera ตามที่คุณตั้งค่าไว้ใน Simulation
        _bot = WalkieRobot(
            ip="127.0.0.1",
            ros_protocol="zenoh",  # ใช้ Zenoh สำหรับ Nav/Telemetry
            ros_port=7447,
            camera_protocol="zenoh",  # ใช้ Zenoh สำหรับดึงภาพ (สำคัญมาก!)
            camera_port=7447,
        )

        # ให้เวลาระบบ Zenoh ในการทำ Discovery เล็กน้อยเพื่อให้ Module ต่างๆ พร้อมใช้งาน
        time.sleep(1.5)

    return _bot


@tool
def move_to_coordinates(x: float, y: float, heading: float = 0.0) -> str:
    """
    Move the Walkie robot to a specific (x, y) coordinate on the map.
    Use this tool when the user provides a destination or when a specific object's location is known.
    Arguments:
        x (float): Target x-coordinate in meters.
        y (float): Target y-coordinate in meters.
        heading (float): Target orientation in radians (default is 0.0).
    """
    try:
        bot = get_bot()
        # Execute the navigation command through the Walkie-SDK Navigation module
        # status จะคืนค่าเป็น "SUCCEEDED", "FAILED", หรือ "CANCELED"
        status = bot.nav.go_to(x=x, y=y, heading=heading)
        return f"Navigation task result: {status}"
    except Exception as e:
        return f"Error during navigation: {str(e)}"


@tool
def get_robot_status() -> dict:
    """
    Retrieve the current pose (x, y, heading) and the hardware connection status of the robot.
    Use this to verify the robot's current position before starting a mission or after completing a move.
    """
    try:
        bot = get_bot()
        # Fetch current coordinates from the robot's telemetry system
        pose = bot.status.get_pose()
        return {
            "pose": pose,
            "is_connected": bot.is_connected,
            "camera_active": bot.camera is not None and bot.camera.is_streaming,
        }
    except Exception as e:
        return {"error": str(e), "is_connected": False}
