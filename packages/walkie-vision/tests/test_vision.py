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
    """โหลด Detector จริง (อัปเกรดเป็น CUDA สำหรับ 5090)"""
    return VisionDetector(
        yolo_checkpoint="models/yolo26x.pt",  # หรือ yolo26x.pt ตามที่คุณตั้งชื่อไว้
        device="cuda",
    )


@pytest.fixture
def vision_encoder():
    """โหลด Encoder จริง (อัปเกรดเป็น CUDA เพื่อทำ Batch Processing)"""
    return VisionEncoder(device="cuda")


def test_vision_integration_with_real_file(
    vision_detector, vision_encoder, test_data_path, output_path
):
    """
    Integration Test: เวอร์ชัน High-Performance Batch Processing
    """
    img_path = os.path.join(test_data_path, "test_room_3.jpg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test file not found at {img_path}.")

    frame = cv2.imread(img_path)

    # 1. รัน Detector เพื่อหา Bounding Boxes ทั้งหมดในภาพเดียว
    objects = vision_detector.get_segmented_objects(frame)
    assert len(objects) > 0, "Detector should find at least one object"

    # --- 🔥 BATCH PROCESSING START ---

    # รวบรวมภาพทุกลูกที่ YOLO ตัดมาได้เข้าเป็น List เดียว
    images_to_process = [obj["image"] for obj in objects]

    # ส่งเข้า GPU ประมวลผล Caption และ Embedding ทั้งหมดในคำสั่งเดียว
    print(
        f"\n🚀 Processing {len(images_to_process)} objects in parallel on RTX 5090..."
    )
    all_captions, all_embeddings = vision_encoder.encode_batch(images_to_process)

    # --- BATCH PROCESSING END ---

    test_results_summary = []

    # 2. จัดเก็บผลลัพธ์ (ขั้นตอนนี้เป็น I/O เลยต้องวนลูปเซฟไฟล์ปกติ)
    for i, obj in enumerate(objects):
        caption = all_captions[i]
        embedding = all_embeddings[i]

        # บันทึกภาพวัตถุที่ Crop ออกมา
        crop_filename = f"obj_{i}_{obj['yolo_class']}.jpg"
        cv2.imwrite(os.path.join(output_path, crop_filename), obj["image"])

        # รวบรวมข้อมูลลง Summary
        test_results_summary.append(
            {
                "index": i,
                "class": obj["yolo_class"],
                "caption": caption,
                "crop_path": crop_filename,
                "embedding_sample": embedding[:5],  # เก็บ 5 ค่าแรกดูเป็นตัวอย่าง
            }
        )

        # Assertion เช็คความถูกต้องเบื้องต้น
        if i == 0:
            assert len(caption) > 0
            assert len(embedding) == 512

    # 3. บันทึก Metadata ทั้งหมดลงไฟล์ JSON
    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_results_summary, f, indent=4, ensure_ascii=False)

    print(f"✅ Test completed!")
    print(f"📊 Objects: {len(objects)} | Output: {output_path}")
