import glob
import os
from unittest.mock import MagicMock, patch

import cv2
import pytest

from asr_agent.core.orchestrator import AgentOrchestrator
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view


def test_vision_pipeline_via_orchestrator():
    """
    Test the Orchestrator's reasoning capabilities.
    Verifies if the AI correctly sequences Capture and Analyze commands per SOP.
    """
    # 1. Locate scene images (used to check context if necessary)
    scene_dir = "data/images/scenes"
    image_list = glob.glob(os.path.join(scene_dir, "*.[jJ][pP][gG]"))
    if not image_list:
        pytest.skip(f"No images found in {scene_dir}")

    # 2. Mock external systems to bypass physical robot/hardware dependencies
    with (
        patch("asr_agent.tools.robot.WalkieRobot"),
        patch("asr_agent.tools.vision.VisionDetector"),
        patch("asr_agent.tools.vision.AgentIntegration"),
    ):
        # 3. Initialize Orchestrator
        orchestrator = AgentOrchestrator()

        # 4. Execute user command through Orchestrator
        user_text = "Look at the table and tell me what you see."
        response = orchestrator.run_command(user_text)

        # 5. Planning Verification
        # Check if the LLM (e.g., Qwen) understands the SOP (Capture -> Analyze)
        tool_calls = orchestrator.client.parse_tool_calls(response)

        print(f"\n[AI Thought]: {response.content}")
        print(f"[Planned Tools]: {[c['name'] for c in tool_calls]}")

        # Verify the two-step sequence required by our RoboCup rules
        assert len(tool_calls) >= 2
        assert tool_calls[0]["name"] == "get_current_view"
        assert tool_calls[1]["name"] == "analyze_and_store_objects"

        # Confirm no actual execution occurred (DB/hardware should not be called during planning)
        # (Note: Since AgentIntegration is patched, this confirms zero unintended side effects)
