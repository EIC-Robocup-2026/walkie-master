from langgraph.graph import StateGraph, END
from asr_agent.graph.state import AgentState
from asr_agent.graph.nodes import call_model, execute_tools

def route_after_model(state: AgentState):
    last_message = state["messages"][-1]
    # ตรวจสอบว่ามี tool_calls และไม่ใช่ข้อความ Error
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # คืนค่า END ของ LangGraph โดยตรง
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")
workflow.add_edge("tools", "agent")

# ปรับ Mapping ตรงนี้ให้ใช้ END เป็นทั้ง Key และ Value
workflow.add_conditional_edges(
    "agent",
    route_after_model,
    {
        "tools": "tools",
        END: END  # แก้จาก "end": END
    }
)

app = workflow.compile()
