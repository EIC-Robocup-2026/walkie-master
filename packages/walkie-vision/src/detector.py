import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from ultralytics import YOLO


class VisionDetector:
    def __init__(self, sam_checkpoint: str, yolo_checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. Load YOLO11x for classification/naming
        # อ้างอิงจากแผน Image Encoder Lab ที่ใช้ YOLO11x สำหรับ precise object naming
        self.yolo_model = YOLO(yolo_checkpoint).to(self.device)

        # 2. Load SAM 2.1 for open-world object proposal
        # อ้างอิงจากแผนที่ใช้ SAM 2.1 Large สำหรับการหา candidate วัตถุ
        sam2_model = build_sam2("sam2_h14.yaml", sam_checkpoint, device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    def detect_and_segment(self, image_path: str):
        """
        Inference Pipeline:
        1. SAM 2.1 หาวัตถุทั้งหมดในภาพ (Proposals)
        2. YOLO11x ช่วยยืนยันและระบุชื่อวัตถุเหล่านั้น (Naming)
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Step 1: Generate Masks using SAM 2.1
        masks = self.mask_generator.generate(image_np)

        results = []
        for mask in masks:
            # Crop วัตถุตามขอบเขตที่ SAM ตรวจพบ
            bbox = mask["bbox"]  # [x, y, w, h]
            cropped_img = image.crop(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )

            # Step 2: Classify using YOLO11x
            yolo_results = self.yolo_model.predict(cropped_img, verbose=False)

            # รวบรวมข้อมูลเพื่อส่งต่อให้ DB หรือ Agent
            results.append(
                {
                    "label": yolo_results[0].names[yolo_results[0].probs.top1],
                    "confidence": float(yolo_results[0].probs.top1conf),
                    "bbox": bbox,
                    "segmentation": mask["segmentation"],  # RLE หรือ Binary Mask
                    "crop": cropped_img,
                }
            )

        return results
