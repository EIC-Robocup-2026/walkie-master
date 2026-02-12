import numpy as np
import pyaudio


class AudioProcessor:
    """
    จัดการการรับสัญญาณเสียงจากไมโครโฟน
    """

    def __init__(self, rate: int = 16000, chunk: int = 1024):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()

    def record_until_silence(self) -> np.ndarray:
        """
        บันทึกเสียงจนกว่าจะไม่มีเสียงพูด (ตัวอย่างแบบง่าย)
        ในงานจริงอาจใช้ VAD (Voice Activity Detection) ช่วย
        """
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        print("🎤 Listening...")

        frames = []
        # จำลองการบันทึกเสียง 3 วินาที (ควรเปลี่ยนเป็น VAD ในภายหลัง)
        for _ in range(0, int(self.rate / self.chunk * 3)):
            data = stream.read(self.chunk)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        return np.concatenate(frames).astype(np.float32) / 32768.0
