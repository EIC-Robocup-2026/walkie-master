import os

import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from ultralytics import YOLO


class VisionDetector:
    def __init__(self, sam_checkpoint: str, yolo_checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. Load YOLO11x
        self.yolo_model = YOLO(yolo_checkpoint).to(self.device)

        # 2. Setup SAM 2.1
        # ต้องใช้ Absolute Path เพื่อป้องกันความผิดพลาดเรื่อง Workdir
        checkpoint_abs = os.path.abspath(sam_checkpoint)

        # [จุดสำคัญ] config_dir ต้องชี้ไปที่โฟลเดอร์ 'แม่' (configs)
        # เพื่อให้เราเรียกไฟล์ลูกเป็น 'sam2.1/sam2.1_h14.yaml' ได้เหมือนโครงสร้างต้นฉบับ
        config_dir_abs = os.path.abspath("data/models/configs")
        config_rel_path = "sam2.1/sam2.1_h14.yaml"

        print(f"🔧 Hydra Search Path: {config_dir_abs}")
        print(f"📦 Loading Config: {config_rel_path}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # เริ่มต้น Hydra โดยชี้ไปที่โฟลเดอร์แม่
        with initialize_config_dir(config_dir=config_dir_abs, version_base="1.2"):
            # build_sam2 จะไปควานหา config_rel_path ภายใต้ config_dir_abs
            self.sam_model = build_sam2(
                config_rel_path, checkpoint_abs, device=self.device
            )

        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam_model)

    # ... (ฟังก์ชัน detect_and_segment คงเดิม)
    def detect_and_segment(self, image_np_or_path):
        """
        Inference Pipeline
        """
        if isinstance(image_np_or_path, str):
            image = Image.open(image_np_or_path).convert("RGB")
            image_np = np.array(image)
        else:
            image_np = image_np_or_path
            image = Image.fromarray(image_np)

        # Step 1: Generate Masks
        masks = self.mask_generator.generate(image_np)

        results = []
        for mask in masks:
            bbox = mask["bbox"]  # [x, y, w, h]
            cropped_img = image.crop(
                (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            )

            # Step 2: Classify using YOLO11x
            yolo_results = self.yolo_model.predict(cropped_img, verbose=False)

            # กรองกรณี YOLO หาไม่เจอ (top1conf อาจไม่มีถ้าโมเดลไม่ใช่ -cls)
            try:
                label = yolo_results[0].names[yolo_results[0].probs.top1]
                conf = float(yolo_results[0].probs.top1conf)
            except AttributeError:
                label = "unknown"
                conf = 0.0

            results.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox,
                    "segmentation": mask["segmentation"],
                    "crop": cropped_img,
                }
            )

        return results
