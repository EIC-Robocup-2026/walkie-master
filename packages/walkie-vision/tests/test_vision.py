import json
import os
from datetime import datetime

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
def output_path():
    """สร้างและชี้ไปยัง Directory สำหรับเก็บผลลัพธ์การ Test"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(__file__), "outputs", timestamp)
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture
def vision_detector():
    """โหลด Detector จริง (รุ่น Standard Detection)"""
    return VisionDetector(
        yolo_checkpoint="models/yolo26x.pt",
        device="cpu",  # ปรับเป็น cuda ถ้ามี GPU และอยากเทสความเร็วเครื่องจริง
    )


@pytest.fixture
def vision_encoder():
    """โหลด Encoder จริง (ใช้ CPU)"""
    return VisionEncoder(device="cpu")


def test_vision_integration_with_real_file(
    vision_detector, vision_encoder, test_data_path, output_path
):
    """
    Integration Test พร้อมระบบเก็บผลลัพธ์ (Artifacts)
    เวอร์ชัน YOLO-only Segmentation
    """
    img_path = os.path.join(test_data_path, "test_room_2.jpg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test file not found at {img_path}. Please add a sample image.")

    frame = cv2.imread(img_path)

    # 1. รัน Detector (ตอนนี้เป็น YOLO-seg ตัวเดียวแล้ว จะเร็วขึ้นมาก)
    objects = vision_detector.get_segmented_objects(frame)

    assert len(objects) > 0, "Detector should find at least one object"

    # เตรียม List สำหรับเก็บ Metadata เพื่อบันทึกเป็น JSON
    test_results_summary = []

    # 2. วนลูปเก็บผลลัพธ์ของทุกวัตถุที่เจอ
    for i, obj in enumerate(objects):
        # รัน Encoder (ตอนนี้ Caption ควรจะออกมาเป็นคำบรรยายจริงแล้ว)
        caption, embedding = vision_encoder.encode_object(obj["image"])

        # บันทึกภาพวัตถุที่ Crop ออกมา
        crop_filename = f"obj_{i}_{obj['yolo_class']}.jpg"
        cv2.imwrite(os.path.join(output_path, crop_filename), obj["image"])

        # เก็บข้อมูลลง Summary
        test_results_summary.append(
            {
                "index": i,
                "class": obj["yolo_class"],
                "caption": caption,
                "crop_path": crop_filename,
                "embedding_sample": embedding[:5],
            }
        )

        # Test assertions สำหรับตัวแรก
        if i == 0:
            assert len(caption) > 0
            # ตรวจสอบขนาด Embedding (ถ้าใช้ clip-ViT-B-32 จะเป็น 512)
            assert len(embedding) == 512

    # 3. บันทึก Metadata ทั้งหมดลงไฟล์ JSON
    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_results_summary, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Test artifacts saved to: {output_path}")
    print(f"📊 Total objects found: {len(objects)}")
