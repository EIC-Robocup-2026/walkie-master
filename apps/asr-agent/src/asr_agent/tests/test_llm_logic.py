import pytest
from langchain_core.messages import HumanMessage
from asr_agent.graph.workflow import app  # ดึง Graph ที่ประกอบร่างแล้วมาทดสอบ
from asr_agent.graph.state import AgentState

class TestAgentLogic:
    """
    ชุดการทดสอบ Logic การตัดสินใจของ Qwen3 8B (FC) ภายใน LangGraph
    โดยเน้นตรวจสอบการเลือก Function Calling ที่ถูกต้อง
    """

    @pytest.fixture
    def initial_state(self):
        return {
            "messages": [],
            "current_pose": {"x": 0.0, "y": 0.0, "heading": 0.0},
            "last_observation": {},
            "mission_status": "idle"
        }

    def test_navigation_logic(self, initial_state):
        """1. ตรวจสอบว่าสั่งให้เดินแล้วเรียก move_to_coordinates ถูกต้องหรือไม่"""
        user_input = "Walkie, please go to the kitchen at coordinates x=2.5 and y=3.0."
        initial_state["messages"] = [HumanMessage(content=user_input)]

        # รัน Workflow
        output = app.invoke(initial_state)
        last_message = output["messages"][-1]

        # ตรวจสอบว่ามีการเรียก Tool หรือไม่
        assert len(last_message.tool_calls) > 0
        tool_call = last_message.tool_calls[0]
        assert tool_call["name"] == "move_to_coordinates"
        assert tool_call["args"]["x"] == 2.5
        assert tool_call["args"]["y"] == 3.0

    def test_perception_logic(self, initial_state):
        """2. ตรวจสอบว่าถามหาของแล้วเรียก observe_scene ถูกต้องหรือไม่"""
        user_input = "Can you find my blue mug on the table?"
        initial_state["messages"] = [HumanMessage(content=user_input)]

        output = app.invoke(initial_state)
        last_message = output["messages"][-1]

        assert any(tc["name"] == "observe_scene" for tc in last_message.tool_calls)
        # ตรวจสอบว่ามีการระบุ focus_object ที่เกี่ยวข้อง
        args = next(tc["args"] for tc in last_message.tool_calls if tc["name"] == "observe_scene")
        assert "mug" in args["focus_object"].lower()

    def test_memory_retrieval_logic(self, initial_state):
        """3. ตรวจสอบว่าถามหาของเก่าแล้วเรียก search_memory หรือไม่"""
        user_input = "Where did you see my keys last time?"
        initial_state["messages"] = [HumanMessage(content=user_input)]

        output = app.invoke(initial_state)
        last_message = output["messages"][-1]

        assert any(tc["name"] == "search_memory" for tc in last_message.tool_calls)

    def test_deterministic_behavior(self, initial_state):
        """4. ตรวจสอบความนิ่ง (Deterministic) ของคำตอบ"""
        user_input = "Go to (1.0, 1.0)"
        initial_state["messages"] = [HumanMessage(content=user_input)]

        # รัน 2 ครั้งต้องได้ผลเหมือนเดิมเพราะตั้งค่า Seed และ Temp=0 ไว้
        res1 = app.invoke(initial_state)["messages"][-1].tool_calls[0]["args"]
        res2 = app.invoke(initial_state)["messages"][-1].tool_calls[0]["args"]

        assert res1 == res2
