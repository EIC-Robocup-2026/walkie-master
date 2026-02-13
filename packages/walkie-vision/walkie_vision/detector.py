import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


class VisionDetector:
    def __init__(
        self,
        yolo_checkpoint: str = "data/models/yolo26x.pt",  # ใช้รุ่นมาตรฐาน (ไม่ต้องมี -seg)
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with tqdm(total=1, desc="🤖 Initializing YOLO Detector") as pbar:
            pbar.set_postfix_str(f"Loading {yolo_checkpoint}")
            self.yolo_model = YOLO(yolo_checkpoint).to(self.device)
            pbar.update(1)
            pbar.set_postfix_str("Ready!")

    def get_segmented_objects(self, frame):
        """Pipeline: Pure Object Detection -> BBox Cropping"""
        # รัน Detection ปกติ
        results = self.yolo_model(frame, verbose=False)
        detected_items = []

        for result in results:
            if result.boxes is None:
                continue

            # ดึงข้อมูล Boxes, Classes และ Confidence
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # Crop ภาพตาม Bounding Box ตรงๆ
                cropped_obj = frame[y1:y2, x1:x2]

                if cropped_obj.size == 0:
                    continue

                label = names[int(classes[i])]

                detected_items.append(
                    {
                        "image": cropped_obj,
                        "yolo_class": label,
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        return detected_items
