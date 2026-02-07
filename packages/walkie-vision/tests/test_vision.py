import os

import cv2
import numpy as np
import pytest

from walkie_vision.detector import VisionDetector
from walkie_vision.encoder import VisionEncoder


@pytest.fixture
def test_data_path():
    """ชี้ไปยัง Directory ที่เก็บภาพสำหรับ Test"""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def vision_detector():
    """โหลด Detector จริง (ใช้ CPU เพื่อการทดสอบ)"""
    # ตรวจสอบให้แน่ใจว่า path โมเดลถูกต้องตามที่คุณวางไว้ในเครื่อง
    return VisionDetector(
        sam_checkpoint="models/sam2_b.pt",
        yolo_checkpoint="models/yolov8x.pt",
        device="cpu",
    )


@pytest.fixture
def vision_encoder():
    """โหลด Encoder จริง (ใช้ CPU)"""
    return VisionEncoder(device="cpu")


def test_vision_integration_with_real_file(
    vision_detector, vision_encoder, test_data_path
):
    """
    Integration Test:
    1. โหลดภาพจริงจากไฟล์
    2. ทำ Segmentation & Detection
    3. ส่งต่อให้ Encoder สร้าง Caption & Embedding
    """
    img_path = os.path.join(test_data_path, "test_room.jpg")

    # ตรวจสอบว่ามีไฟล์ภาพจริงก่อนรัน
    if not os.path.exists(img_path):
        pytest.skip(f"Test file not found at {img_path}. Please add a sample image.")

    frame = cv2.imread(img_path)

    # 1. รัน Detector จริง
    objects = vision_detector.get_segmented_objects(frame)

    # คราวนี้ len(objects) ควรจะ > 0 เพราะใช้ภาพจริงที่มีวัตถุ
    assert len(objects) > 0, (
        "Detector should find at least one object in the sample image"
    )

    # 2. ทดสอบชิ้นงานแรกที่เจอ
    obj = objects[0]
    assert "image" in obj
    assert "yolo_class" in obj

    # 3. รัน Encoder จริง
    caption, embedding = vision_encoder.encode_object(obj["image"])

    print(f"\n🔍 Found: {obj['yolo_class']}")
    print(f"📝 Caption: {caption}")

    assert len(caption) > 0
    assert len(embedding) == 512
