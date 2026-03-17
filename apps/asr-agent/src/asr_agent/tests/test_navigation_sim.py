import time

from asr_agent.core.orchestrator import AgentOrchestrator
from asr_agent.tools.robot import get_robot_status, move_to_coordinates


def test_navigation_sim_execution():
    """
    Integration Test: Test LLM commands and execution on a real Gazebo Simulation.
    Condition: Gazebo and Walkie-SDK Bridge must be running at 127.0.0.1.
    """

    # 1. Create Orchestrator (No need to patch WalkieRobot to allow real Sim connection)
    orchestrator = AgentOrchestrator()

    # Define test coordinates in Gazebo
    target_x = 1.0
    target_y = 1.0
    user_command = f"Walkie, move to the waypoint at x={target_x} and y={target_y}"

    print(f"\n[Command]: {user_command}")

    # 2. Let the AI plan the task
    response = orchestrator.run_command(user_command)
    tool_calls = orchestrator.client.parse_tool_calls(response)

    assert len(tool_calls) > 0, "AI failed to plan the movement."
    call = tool_calls[0]
    args = call["arguments"]

    print(f"🧠 Agent Thought: {response.content}")
    print(f"🚀 Executing on Gazebo: {call['name']} -> {args}")

    # 3. Execute on the Simulation via .invoke()
    # This function calls bot.nav.go_to in walkie-sdk
    execution_result = move_to_coordinates.invoke(args)
    print(f"📡 SDK Response: {execution_result}")

    # 4. Check robot status after command (Feedback Loop)
    # Note: Navigation takes time; we might need to wait or check status periodically.
    time.sleep(2)  # Wait for the robot to start moving

    status = get_robot_status.invoke({})
    current_pose = status["pose"]

    print(f"📍 Current Robot Pose in Sim: {current_pose}")

    # Verify that the robot is actually connected to the Sim
    assert status["is_connected"] is True, "Robot is not connected to Gazebo/SDK Bridge"
    # assert "Navigation task result" in execution_result
