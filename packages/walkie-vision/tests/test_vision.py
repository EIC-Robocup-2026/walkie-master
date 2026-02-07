import numpy as np
import pytest


def test_detector_inference(vision_detector):
    """ทดสอบว่า Detector รับภาพแล้วส่งผลลัพธ์กลับมาถูกต้อง"""
    # สร้างภาพสุ่มขนาด 640x480
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    results = vision_detector.detect_and_segment(dummy_img)

    assert isinstance(results, list)
    # แม้ภาพสุ่มจะตรวจไม่เจออะไร แต่ต้องไม่ Raise Exception
    if len(results) > 0:
        assert "label" in results[0]
        assert "bbox" in results[0]


def test_encoder_embedding():
    """เช็คว่า VisionEncoder สร้าง Vector ได้ถูกมิติ"""
    from PIL import Image
    from walkie_vision import VisionEncoder

    encoder = VisionEncoder(device="cpu")  # ใช้ CPU เพื่อความเร็วในการเทส
    dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    emb = encoder.get_image_embedding(dummy_pil)
    assert len(emb) == 512  # CLIP-ViT-B-32 มักจะเป็น 512
