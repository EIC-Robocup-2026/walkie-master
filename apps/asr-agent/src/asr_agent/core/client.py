import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from openai import OpenAI
from pydantic import BaseModel


class QwenAgentClient:
    """
    Client for managing communication with Qwen3 8B (FC).
    Optimized for deterministic outputs and robust Function Calling.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-walkie-master",
        model_name: str = "qwen3-8b",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

        # Lock parameters to ensure maximum reproducibility for robot control tasks
        self.default_params = {
            "temperature": 0.0,  # Minimize randomness
            "top_p": 0.1,
            "seed": 42,  # Fixed seed for consistent results during testing
            "max_tokens": 1024,
        }

    def generate_response(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        Send messages to the LLM and handle format conversion between LangChain and OpenAI.
        """
        # --- Convert LangChain Message objects to OpenAI format ---
        formatted_messages = []
        for m in messages:
            # Handle LangChain Message Objects (HumanMessage, AIMessage, etc.)
            if hasattr(m, "type"):
                role = "assistant" if m.type == "ai" else m.type
                # Normalize 'human' role to 'user' for OpenAI API compatibility
                role = "user" if role == "human" else role
                formatted_messages.append({"role": role, "content": m.content})
            # Handle standard dictionaries
            elif isinstance(m, dict):
                formatted_messages.append(m)
            else:
                # Fallback for raw strings or unexpected types
                formatted_messages.append({"role": "user", "content": str(m)})

        try:
            # vLLM/OpenAI may error out if tool_choice="auto" is passed without tools
            actual_tool_choice = tool_choice if tools else None

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                tools=tools,
                tool_choice=actual_tool_choice,
                **self.default_params,
            )
            return response.choices[0].message

        except Exception as e:
            print(f"❌ LLM Client Error: {e}")
            # Return AIMessage to maintain LangGraph state continuity and prevent crashes
            return AIMessage(
                content=f"Error: I encountered an issue with the LLM server: {str(e)}"
            )

    def parse_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        """
        Parse LLM response into a clean, list-based tool call format.
        """
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []

        calls = []
        for tool_call in message.tool_calls:
            try:
                calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    }
                )
            except json.JSONDecodeError:
                print(
                    f"⚠️ Failed to parse arguments for tool: {tool_call.function.name}"
                )
                continue

        return calls
