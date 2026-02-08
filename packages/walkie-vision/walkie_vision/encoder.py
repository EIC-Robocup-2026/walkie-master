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
        vqa_model="Salesforce/blip-vqa-base",
        embed_model="clip-ViT-B-32",
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with tqdm(total=3, desc="🧠 Initializing Semantic Encoder") as pbar:
            pbar.set_postfix_str("Loading BLIP Processor")
            self.vqa_processor = BlipProcessor.from_pretrained(vqa_model)
            pbar.update(1)

            pbar.set_postfix_str("Loading BLIP Model")
            self.vqa_model = BlipForConditionalGeneration.from_pretrained(vqa_model).to(
                self.device
            )
            pbar.update(1)

            pbar.set_postfix_str("Loading CLIP")
            self.embed_model = SentenceTransformer(embed_model, device=self.device)
            pbar.update(1)
            pbar.set_postfix_str("Ready!")

    def generate_caption(self, image: Image.Image) -> str:
        """โหมดบรรยายภาพ (Image Captioning) - ให้ผลลัพธ์ดีกว่าการถาม describe"""
        # การไม่ใส่ text prompt หรือใส่แค่ "a photo of" จะทำให้โมเดลรันโหมด Captioning
        inputs = self.vqa_processor(image, "a photo of", return_tensors="pt").to(
            self.device
        )

        # ปรับ max_new_tokens เพื่อให้ประโยคยาวขึ้นเล็กน้อย
        out = self.vqa_model.generate(**inputs, max_new_tokens=50)
        return self.vqa_processor.decode(out[0], skip_special_tokens=True)

    def ask_question(self, image: Image.Image, question: str) -> str:
        """Visual Question Answering (VQA) - ใช้เมื่อต้องการเจาะจงข้อมูล"""
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(
            self.device
        )
        out = self.vqa_model.generate(**inputs, max_new_tokens=50)
        return self.vqa_processor.decode(out[0], skip_special_tokens=True)

    def get_image_embedding(self, image: Image.Image) -> list:
        embedding = self.embed_model.encode(image)
        # ตรวจสอบว่าเป็น list หรือยัง (SentenceTransformer มักคืนค่าเป็น numpy/torch)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def encode_object(self, image_np: np.ndarray) -> tuple[str, list]:
        """สกัด Caption และ Vector จากภาพ NumPy"""
        if image_np is None or image_np.size == 0:
            return "empty image", [0.0] * 512

        # 1. แปลง BGR (OpenCV) เป็น RGB (PIL)
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # 2. ใช้ generate_caption แทนการถาม describe
        caption = self.generate_caption(pil_img)

        # 3. สร้าง Embedding
        embedding = self.get_image_embedding(pil_img)

        return caption, embedding
