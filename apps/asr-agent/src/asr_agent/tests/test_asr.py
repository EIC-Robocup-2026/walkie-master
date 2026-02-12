import pytest
from asr_agent.asr.model import ASRModel

def test_asr_model_loading():
    """ตรวจสอบว่าโหลดโมเดล distil-large-v3 ลง VRAM ได้"""
    asr = ASRModel(model_size="distil-large-v3")
    assert asr.model is not None
    assert asr.device == "cuda"
