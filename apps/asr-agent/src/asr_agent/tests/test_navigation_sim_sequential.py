import time

from asr_agent.core.orchestrator import AgentOrchestrator
from asr_agent.tools.robot import get_robot_status, move_to_coordinates


def test_sequential_sim_navigation():
    """Test sequential navigation to 2 points on Simulation"""
    orchestrator = AgentOrchestrator()

    waypoints = [
        {"command": "Go to x=0.5, y=0.5", "target": (0.5, 0.5)},
        {"command": "Then move to x=0.0, y=0.0", "target": (0.0, 0.0)},
    ]

    for wp in waypoints:
        print(f"\n[Step]: {wp['command']}")
        response = orchestrator.run_command(wp["command"])
        tool_calls = orchestrator.client.parse_tool_calls(response)

        args = tool_calls[0]["arguments"]
        res = move_to_coordinates.invoke(args)

        print(f"✅ {res}")

        # Check status after movement (In production, a loop might be needed until target reached)
        status = get_robot_status.invoke({})
        print(f"Robot now at: {status['pose']}")
