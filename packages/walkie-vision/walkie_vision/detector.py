import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import SAM, YOLO


class VisionDetector:
    def __init__(
        self,
        sam_checkpoint: str = "models/sam3.pt",
        yolo_checkpoint: str = "models/yolo11x.pt",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with tqdm(total=2, desc="🤖 Initializing Vision Detector") as pbar:
            pbar.set_postfix_str("Loading YOLO")
            self.yolo_model = YOLO(yolo_checkpoint).to(self.device)
            pbar.update(1)

            pbar.set_postfix_str("Loading SAM")
            self.sam_model = SAM(sam_checkpoint).to(self.device)
            pbar.update(1)
            pbar.set_postfix_str("Ready!")

    def get_segmented_objects(self, frame):
        """Pipeline: SAM (Segment) -> YOLO (Classify)"""
        results = self.sam_model(frame)
        detected_items = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                cropped_obj = self._apply_mask(frame, mask_np)

                # ให้ YOLO ระบุวัตถุในส่วนที่ Segment ออกมา
                yolo_res = self.yolo_model.predict(cropped_obj, verbose=False)

                try:
                    if hasattr(yolo_res[0], "probs") and yolo_res[0].probs is not None:
                        label = yolo_res[0].names[yolo_res[0].probs.top1]
                    else:
                        label = yolo_res[0].names[int(yolo_res[0].boxes[0].cls)]
                except (AttributeError, IndexError):
                    label = "unknown"

                detected_items.append(
                    {
                        "image": cropped_obj,  # numpy array
                        "yolo_class": label,
                        "bbox": result.boxes.xyxy[i].cpu().numpy(),
                    }
                )
        return detected_items

    def _apply_mask(self, image, mask):
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return (image * mask_3d).astype(np.uint8)
