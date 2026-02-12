import pytest
import torch
from walkie_sdk.robot import WalkieRobot
from walkie_db.agent_integration import AgentIntegration
from walkie_vision.detector import ObjectDetector
from asr_agent.asr.model import ASRModel
from asr_agent.core.client import QwenAgentClient

class TestWalkieEcosystem:

    @pytest.mark.order(1)
    def test_vision_module(self):
        """1. ตรวจสอบดวงตา: โหลดโมเดล YOLO/PaliGemma และใช้ GPU ได้หรือไม่"""
        print("\n👁️ Testing Vision...")
        detector = ObjectDetector()
        assert detector is not None
        assert torch.cuda.is_available(), "GPU (RTX 5090) should be available for Vision"

    @pytest.mark.order(2)
    def test_sdk_connection(self):
        """2. ตรวจสอบร่างกาย: เชื่อมต่อกับ Robot Base ผ่าน SDK ได้หรือไม่"""
        print("\n🤖 Testing SDK Connection...")
        # ทดสอบเชื่อมต่อแบบ Local Mock หรือ IP จริง
        bot = WalkieRobot(ip="127.0.0.1")
        assert bot.is_connected
        bot.disconnect()

    @pytest.mark.order(3)
    def test_database_persistence(self):
        """3. ตรวจสอบความจำ: เขียนและอ่านข้อมูลจาก WalkieDB ได้หรือไม่"""
        print("\n📦 Testing Memory (DB)...")
        agent_db = AgentIntegration(base_db_path="data/test_db")
        test_id = "test_item_001"
        agent_db.process_object_detection(test_id, [1, 2, 3], [0.1]*512, "test_label")

        # ตรวจสอบว่าพิกัดถูกบันทึกจริง
        coords = agent_db.get_target_coords("object", test_id)
        assert coords == (1.0, 2.0, 3.0)

    @pytest.mark.order(4)
    def test_asr_loading(self):
        """4. ตรวจสอบหู: โหลดโมเดล Faster-Whisper ลง GPU ได้หรือไม่"""
        print("\n🎙️ Testing ASR...")
        asr = ASRModel(model_size="distil-large-v3")
        assert asr.model is not None

    @pytest.mark.order(5)
    def test_llm_server_reachability(self):
        """5. ตรวจสอบสมอง: เชื่อมต่อกับ API ของ Qwen3 ที่พอร์ต 8000 ได้หรือไม่"""
        print("\n🧠 Testing LLM Client...")
        client = QwenAgentClient(base_url="http://localhost:8000/v1")
        # ทดสอบส่งข้อความสั้นๆ เพื่อเช็คสถานะการตอบกลับ
        msg = [{"role": "user", "content": "hello"}]
        response = client.generate_response(msg)
        assert response is not None, "LLM Server (vLLM/Ollama) must be running at port 8000"
