import cv2
import numpy as np
from ultralytics import SAM, YOLO  # สมมติใช้ SAM/YOLO จาก Ultralytics


class VisionDetector:
    def __init__(self):
        # โหลด SAM 3 และ YOLO Model
        self.sam_model = SAM("models/sam2_b.pt")  # หรือรุ่นที่ต้องการ
        self.yolo_model = YOLO("models/yolov8x.pt")

    def get_segmented_objects(self, frame):
        """
        1. ใช้ SAM เพื่อหา Mask ของวัตถุทั้งหมด
        2. ใช้ YOLO เพื่อระบุประเภทของวัตถุในแต่ละ Mask
        """
        results = self.sam_model(frame)
        detected_items = []

        for result in results:
            masks = result.masks.data
            for i, mask in enumerate(masks):
                # สร้างภาพ Crop เฉพาะวัตถุโดยใช้ Mask
                mask_np = mask.cpu().numpy()
                cropped_obj = self._apply_mask(frame, mask_np)

                # ให้ YOLO ช่วยระบุว่าใน Mask นี้คืออะไร
                yolo_res = self.yolo_model(cropped_obj)
                label = yolo_res[0].probs.top1_label if yolo_res[0].probs else "unknown"

                detected_items.append(
                    {
                        "image": cropped_obj,
                        "yolo_class": label,
                        "bbox": result.boxes.xyxy[i].cpu().numpy(),
                    }
                )
        return detected_items

    def _apply_mask(self, image, mask):
        # ฟังก์ชันสำหรับตัดภาพเฉพาะส่วนที่มี Mask (Background เป็นสีดำ)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return (image * mask_3d).astype(np.uint8)
