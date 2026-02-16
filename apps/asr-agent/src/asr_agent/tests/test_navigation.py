from unittest.mock import MagicMock, patch

import pytest

from asr_agent.core.orchestrator import AgentOrchestrator


def test_navigation_argument_logic():
    """Verify that the Orchestrator extracts correct (x, y) coordinates from a command."""

    # 1. Patch external systems to prevent actual hardware/network connections
    with patch("asr_agent.tools.robot.WalkieRobot"):
        # 2. Initialize Orchestrator (automatically loads SYSTEM_PROMPT and Tools)
        orchestrator = AgentOrchestrator()
        user_command = "Walkie, please go to the kitchen at coordinates x=2.5 and y=4.0"

        print(f"\n[Input]: {user_command}")

        # 3. Execute command through Orchestrator (returns an AI Response Object)
        response = orchestrator.run_command(user_command)

        # 4. Validate planning (Tool Calls)
        tool_calls = orchestrator.client.parse_tool_calls(response)

        assert len(tool_calls) > 0, (
            "The Orchestrator failed to plan a navigation tool call."
        )

        call = tool_calls[0]
        args = call["arguments"]

        print(f"[Agent Thought]: {response.content}")
        print(f"[Planned Call]: {call['name']} -> {args}")

        # 5. Verify argument extraction logic
        assert call["name"] == "move_to_coordinates"
        assert float(args["x"]) == 2.5
        assert float(args["y"]) == 4.0


def test_navigation_complex_command():
    """Verify the Orchestrator's reasoning for navigation based on provided context."""

    with patch("asr_agent.tools.robot.WalkieRobot"):
        orchestrator = AgentOrchestrator()

        # Inject context into the command for LLM reasoning
        user_command = (
            "Context: Sofa is located at x=1.2, y=0.5. Walkie, move to the sofa."
        )

        response = orchestrator.run_command(user_command)
        tool_calls = orchestrator.client.parse_tool_calls(response)

        assert len(tool_calls) > 0
        args = tool_calls[0]["arguments"]

        print(f"\n[Complex Command]: {user_command}")
        print(f"[Model Thought]: {response.content}")
        print(f"[Predicted Args]: {args}")

        # Verify if the AI correctly extracts coordinates from the provided context
        assert float(args["x"]) == 1.2
        assert float(args["y"]) == 0.5


def test_sequential_navigation_logic():
    """
    Test a mission with 3 sequential navigation steps via Orchestrator.
    Verifies if the Agent follows the multi-step planning SOP.
    """
    with patch("asr_agent.tools.robot.WalkieRobot"):
        orchestrator = AgentOrchestrator()

        # Command containing three sequential destination targets
        user_command = (
            "Go to the kitchen at x=2.0, y=3.0. "
            "After that, move to the living room at x=5.0, y=5.0. "
            "Finally, go to the entrance at x=0.0, y=0.0."
        )

        print(f"\n[Sequential Command]: {user_command}")

        # 1. Let the Orchestrator generate the plan
        response = orchestrator.run_command(user_command)

        # 2. Validate the generated task list (Tool Calls)
        tool_calls = orchestrator.client.parse_tool_calls(response)

        print(f"[Model Thought]: {response.content}")
        print(f"[Total Steps Planned]: {len(tool_calls)}")

        # 3. Verify the sequence and coordinates for all 3 steps
        assert len(tool_calls) >= 3, (
            f"Expected at least 3 tool calls, but got {len(tool_calls)}"
        )

        expected_targets = [
            {"x": 2.0, "y": 3.0},
            {"x": 5.0, "y": 5.0},
            {"x": 0.0, "y": 0.0},
        ]

        for i, target in enumerate(expected_targets):
            call = tool_calls[i]
            assert call["name"] == "move_to_coordinates"
            assert float(call["arguments"]["x"]) == target["x"]
            assert float(call["arguments"]["y"]) == target["y"]
            print(f"✅ Verified Step {i + 1}: Target ({target['x']}, {target['y']})")
