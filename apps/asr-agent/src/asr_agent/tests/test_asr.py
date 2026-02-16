import pytest

from asr_agent.asr.model import ASRModel


def test_asr_model_loading():
    """Verify that the distil-large-v3 model can be loaded into VRAM."""
    asr = ASRModel(model_size="distil-large-v3")
    assert asr.model is not None
    assert asr.device == "cuda"
