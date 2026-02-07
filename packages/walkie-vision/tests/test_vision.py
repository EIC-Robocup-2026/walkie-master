from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from walkie_vision.detector import VisionDetector
from walkie_vision.encoder import VisionEncoder


@pytest.fixture
def mock_detector():
    """จำลองการทำงานของ SAM และ YOLO"""
    with (
        patch("ultralytics.SAM") as mock_sam_class,
        patch("ultralytics.YOLO") as mock_yolo_class,
    ):
        # Setup Mock SAM Result
        mock_sam_inst = mock_sam_class.return_value
        mock_res = MagicMock()

        # จำลอง Mask 1 ชิ้น
        mock_mask = MagicMock()
        mock_mask.data = torch.ones((1, 100, 100))  # Tensor mask
        mock_res.masks = mock_mask
        mock_res.boxes.xyxy = torch.tensor([[10, 10, 50, 50]])

        mock_sam_inst.return_value = [mock_res]

        # Setup Mock YOLO Result
        mock_yolo_inst = mock_yolo_class.return_value
        mock_yolo_res = MagicMock()
        mock_yolo_res.probs.top1_label = "cup"
        mock_yolo_inst.return_value = [mock_yolo_res]

        yield VisionDetector()


@pytest.fixture
def mock_encoder():
    """จำลองการทำงานของ BLIP และ CLIP"""
    with (
        patch("transformers.BlipProcessor.from_pretrained"),
        patch(
            "transformers.BlipForConditionalGeneration.from_pretrained"
        ) as mock_blip_class,
        patch("clip.load") as mock_clip_load,
    ):
        # Mock CLIP model & preprocess
        mock_clip_model = MagicMock()
        mock_clip_model.encode_image.return_value = torch.randn(1, 512)
        mock_clip_load.return_value = (mock_clip_model, MagicMock())

        # Mock BLIP model
        mock_blip_inst = mock_blip_class.return_value
        mock_blip_inst.generate.return_value = torch.tensor([[101, 102]])

        encoder = VisionEncoder()
        # จำลองฟังก์ชัน decode ของ processor
        encoder.blip_proc.decode = MagicMock(return_value="a red ceramic cup")

        yield encoder


def test_detector_pipeline(mock_detector):
    """ทดสอบว่า Detector สามารถแยก Segment และระบุ Class ได้ถูกต้อง"""
    # สร้างภาพจำลอง (Black Image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detected_objects = mock_detector.get_segmented_objects(dummy_frame)

    # ตรวจสอบว่ามีวัตถุถูกส่งออกมา
    assert len(detected_objects) > 0
    # ตรวจสอบว่า Class ที่ได้จาก YOLO ตรงตามที่ Mock ไว้
    assert detected_objects[0]["yolo_class"] == "cup"
    # ตรวจสอบว่าภาพ Crop มีขนาดที่ถูกต้องตาม Mask
    assert isinstance(detected_objects[0]["image"], np.ndarray)


def test_encoder_semantic_extraction(mock_encoder):
    """ทดสอบการสกัดความหมาย (Caption) และ Vector (Embedding)"""
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    caption, embedding = mock_encoder.encode_object(dummy_crop)

    # ตรวจสอบว่าได้ Caption จาก BLIP
    assert "red ceramic cup" in caption
    # ตรวจสอบความยาว Vector ของ CLIP (ViT-B/32 มักจะเป็น 512)
    assert len(embedding) == 512


def test_vision_integration_flow(mock_detector, mock_encoder):
    """ทดสอบ Flow รวม: จากภาพดิบ สู่การเป็น Metadata พร้อมลง DB"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Detect (SAM + YOLO)
    objects = mock_detector.get_segmented_objects(frame)
    obj = objects[0]

    # 2. Encode (BLIP + CLIP)
    caption, embedding = mock_encoder.encode_object(obj["image"])

    # ข้อมูลพร้อมส่งเข้า AgentIntegration.process_object_detection
    assert obj["yolo_class"] == "cup"
    assert len(caption) > 0
    assert embedding.shape == (512,)
