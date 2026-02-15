import torch
from faster_whisper import WhisperModel


class ASRModel:
    """
    ตัวจัดการโมเดล Faster-Whisper สำหรับการทำงานแบบ Local
    """

    def __init__(self, model_size: str = "distil-large-v3"):
        # ตรวจสอบการใช้งาน GPU (RTX 5090) เพื่อความเร็วสูงสุด
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"🎙️ ASR: Loading model '{model_size}' on {self.device}...")
        self.model = WhisperModel(
            model_size, device=self.device, compute_type=self.compute_type
        )
        print("✓ ASR Model loaded.")

    def transcribe(self, audio_path_or_ndarray):
        """
        แปลงข้อมูลเสียงเป็น Text
        """
        segments, info = self.model.transcribe(audio_path_or_ndarray, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
