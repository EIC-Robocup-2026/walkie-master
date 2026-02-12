import pytest
import torch
from walkie_vision.detector import VisionDetector

def test_vision_module_loading():
    """ตรวจสอบว่า VisionDetector โหลดโมเดลสำเร็จและใช้ GPU ได้"""
    detector = VisionDetector()
    assert detector is not None
    assert hasattr(detector, "get_segmented_objects")
    assert torch.cuda.is_available(), "GPU (RTX 5090) should be available"

def test_yolo_inference_dummy():
    """ทดสอบรันโมเดลด้วยภาพเปล่า เพื่อเช็คว่าเครื่องไม่ค้าง"""
    import numpy as np
    detector = VisionDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.get_segmented_objects(dummy_frame)
    assert isinstance(results, list)
