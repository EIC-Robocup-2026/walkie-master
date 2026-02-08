import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionEncoder:
    def __init__(
        self,
        # เปลี่ยนมาใช้รุ่น Large สำหรับงาน Captioning โดยเฉพาะ
        caption_model="Salesforce/blip-image-captioning-large",
        embed_model="clip-ViT-B-32",
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with tqdm(total=3, desc="🧠 Initializing Semantic Encoder (Large)") as pbar:
            pbar.set_postfix_str("Loading BLIP Large Processor")
            self.vqa_processor = BlipProcessor.from_pretrained(caption_model)
            pbar.update(1)

            pbar.set_postfix_str("Loading BLIP Large Model")
            # โมเดลตัวนี้จะใช้ VRAM มากกว่าตัว Base (ประมาณ 1.8GB)
            self.vqa_model = BlipForConditionalGeneration.from_pretrained(
                caption_model
            ).to(self.device)
            pbar.update(1)

            pbar.set_postfix_str("Loading CLIP")
            self.embed_model = SentenceTransformer(embed_model, device=self.device)
            pbar.update(1)
            pbar.set_postfix_str("Ready!")

    def generate_caption(self, image: Image.Image) -> str:
        """
        บรรยายภาพด้วยโมเดล Large เพื่อดึงรายละเอียดสูงสุด
        """
        # สำหรับโมเดล Captioning-Large เราไม่จำเป็นต้องใส่ Prompt เลยก็ได้
        # หรือใส่แค่ "a photography of" เพื่อกระตุ้นให้มันบรรยายแบบสมจริง
        inputs = self.vqa_processor(image, return_tensors="pt").to(self.device)

        # 🛠 ปรับจูนเพื่อรายละเอียด (High Detail Tuning)
        out = self.vqa_model.generate(
            **inputs,
            max_length=80,  # เพิ่มความยาวสูงสุด
            min_length=20,  # บังคับให้บรรยายไม่สั้นจนเกินไป
            num_beams=5,  # ใช้ Beam Search 5 ทาง
            repetition_penalty=1.5,  # เพิ่มการลงโทษคำซ้ำให้หนักขึ้นเพื่อให้ได้คำที่หลากหลาย
            no_repeat_ngram_size=3,  # ป้องกันวลีซ้ำ
            early_stopping=True,
        )

        caption = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    def get_image_embedding(self, image: Image.Image) -> list:
        """สกัดเวกเตอร์ CLIP (512-dim) สำหรับ Vector Search"""
        embedding = self.embed_model.encode(image)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def encode_object(self, image_np: np.ndarray) -> tuple[str, list]:
        """รับภาพ NumPy (BGR) -> คืนค่า Caption และ Embedding"""
        if image_np is None or image_np.size == 0:
            return "invalid_image", [0.0] * 512

        # 1. จัดการสีและแปลงเป็น PIL
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # 2. สร้างคำบรรยาย (ความสวยงามและรายละเอียดจะดีกว่าเดิมมาก)
        caption = self.generate_caption(pil_img)

        # 3. สร้างเวกเตอร์
        embedding = self.get_image_embedding(pil_img)

        return caption, embedding
