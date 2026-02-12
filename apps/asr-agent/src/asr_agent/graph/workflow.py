from asr_agent.graph.nodes import call_model, execute_tools
from asr_agent.graph.state import AgentState
from langgraph.graph import END, StateGraph


def route_after_model(state: AgentState):
    """
    ตัดสินใจว่าจะไปต่อที่ Tool หรือจบการทำงาน (End)
    """
    last_message = state["messages"][-1]
    # ถ้า LLM ต้องการเรียกใช้ Tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# สร้าง Workflow Graph
workflow = StateGraph(AgentState)

# 1. เพิ่ม Nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

# 2. ตั้งค่าจุดเริ่มต้น
workflow.set_entry_point("agent")

# 3. เพิ่ม Edges (เส้นทางการเดิน)
# หลังจากรันเครื่องมือเสร็จ ให้กลับไปถาม Agent ต่อ (Loop)
workflow.add_edge("tools", "agent")

# ใช้ Conditional Edge เพื่อตัดสินใจว่าจะจบงานหรือไปรัน Tool
workflow.add_conditional_edges(
    "agent", route_after_model, {"tools": "tools", "end": END}
)

# 4. Compile เป็น Application
app = workflow.compile()
