from typing import Any, Dict, List

from langchain_core.utils.function_calling import convert_to_openai_tool

from asr_agent.core.client import QwenAgentClient
from asr_agent.prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from asr_agent.tools.memory import remember_entity, search_memory

# Import Tools only for extracting their function schemas
from asr_agent.tools.robot import get_robot_status, move_to_coordinates
from asr_agent.tools.vision import analyze_and_store_objects, get_current_view


class AgentOrchestrator:
    def __init__(self):
        self.client = QwenAgentClient()

        # Prepare all tools to provide their schemas to the LLM
        raw_tools = [
            move_to_coordinates,
            get_robot_status,
            get_current_view,
            analyze_and_store_objects,
            remember_entity,
            search_memory,
        ]

        # Convert functions to OpenAI-compatible dynamic schemas
        self.tools_schema = [convert_to_openai_tool(t) for t in raw_tools]

        # Initialize conversation state with System Prompt and Standard Operating Procedures (SOP)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages.extend(FEW_SHOT_EXAMPLES)

    def run_command(self, user_text: str):
        """
        AI generates a plan based on user input without executing actual tool calls.
        """
        self.messages.append({"role": "user", "content": user_text})

        # Generate LLM response containing thought process and proposed tool calls
        response = self.client.generate_response(self.messages, tools=self.tools_schema)

        # Update conversation history with the agent's decision
        self.messages.append(response)

        # Return the AI Response Object for validation or execution
        return response
