import argparse
from langchain_core.messages import HumanMessage
from asr_agent.graph.workflow import app  # นำเข้า Graph ที่ Compile แล้ว
from asr_agent.asr.model import ASRModel
from asr_agent.asr.processor import AudioProcessor

def run_agent_workflow(user_input: str):
    """
    ส่งคำสั่งเข้าสู่ LangGraph Workflow และจัดการสถานะ (State)
    """
    # กำหนดสถานะเริ่มต้น (Initial State)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "current_pose": {},
        "last_observation": {},
        "mission_status": "started"
    }

    print(f"\n🚀 Starting Workflow for: '{user_input}'")

    # รัน Graph แบบ Streaming เพื่อดูการตัดสินใจในแต่ละ Node
    for event in app.stream(initial_state):
        for node_name, output in event.items():
            print(f"📍 Node: {node_name}")
            # แสดง Thought ของ Agent ถ้ามีข้อความใหม่เกิดขึ้น
            if "messages" in output:
                last_msg = output["messages"][-1]
                if last_msg.content:
                    print(f"🧠 Thought: {last_msg.content}")

def main():
    parser = argparse.ArgumentParser(description="Walkie Master - LangGraph Agent")
    parser.add_argument("--mode", choices=["voice", "text"], default="text")
    args = parser.parse_args()

    if args.mode == "voice":
        asr = ASRModel()
        processor = AudioProcessor()
        print("🎙️ Voice Mode Active. Say 'exit' to stop.")

        while True:
            audio = processor.record_until_silence()
            text = asr.transcribe(audio)
            if text:
                if "exit" in text.lower(): break
                run_agent_workflow(text)
    else:
        print("💬 Text Mode Active. Type your command (or 'exit').")
        while True:
            user_input = input("\n👤 User: ")
            if user_input.lower() == "exit": break
            run_agent_workflow(user_input)

if __name__ == "__main__":
    main()
