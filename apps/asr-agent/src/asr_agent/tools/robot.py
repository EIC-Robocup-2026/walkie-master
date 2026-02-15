from langchain_core.tools import tool
from walkie_sdk.robot import WalkieRobot

# Singleton instance for the robot connection
_bot = None


def get_bot():
    """
    Initialize and return the WalkieRobot instance using lazy loading.
    The connection to the physical robot or simulation is only established
    when a tool explicitly requests the robot instance for the first time.
    """
    global _bot
    if _bot is None:
        # Connect to the robot base via the specified IP address
        _bot = WalkieRobot(ip="127.0.0.1")
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
    bot = get_bot()
    # Execute the navigation command through the Walkie-SDK Navigation module
    status = bot.nav.go_to(x=x, y=y, heading=heading)
    return f"Navigation task result: {status}"


@tool
def get_robot_status() -> dict:
    """
    Retrieve the current pose (x, y, heading) and the hardware connection status of the robot.
    Use this to verify the robot's current position before starting a mission or after completing a move.
    """
    bot = get_bot()
    # Fetch current coordinates from the robot's telemetry system
    pose = bot.status.get_pose()
    return {"pose": pose, "is_connected": bot.is_connected}
