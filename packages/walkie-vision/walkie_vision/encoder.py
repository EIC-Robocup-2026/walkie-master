import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor


class VisionEncoder:
    def __init__(
        self,
        # แนะนำรุ่น 3b-mix-224 หรือ 3b-mix-448 สำหรับความละเอียดที่สูงขึ้น
        caption_model="google/paligemma-3b-ft-cococap-448",
        embed_model="clip-ViT-B-32",
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with tqdm(total=3, desc="🧠 Initializing PaliGemma Semantic Encoder") as pbar:
            pbar.set_postfix_str("Loading PaliGemma Processor")
            self.vqa_processor = PaliGemmaProcessor.from_pretrained(caption_model)
            pbar.update(1)

            pbar.set_postfix_str("Loading PaliGemma Model")
            # โหลดแบบ bfloat16 เพื่อความเร็วและความประหยัด VRAM (5090 สบายมาก)
            self.vqa_model = (
                PaliGemmaForConditionalGeneration.from_pretrained(
                    caption_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
                )
                .to(self.device)
                .eval()
            )
            pbar.update(1)

            pbar.set_postfix_str("Loading CLIP")
            self.embed_model = SentenceTransformer(embed_model, device=self.device)
            pbar.update(1)
            pbar.set_postfix_str("Ready!")

    def generate_caption(self, image: Image.Image) -> str:
        """บรรยายภาพด้วย PaliGemma โดยใช้ Prompt มาตรฐาน"""
        # PaliGemma จำเป็นต้องมี Prompt เพื่อบอก Task (สำคัญมาก)
        prompt = "caption en\n"

        inputs = self.vqa_processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            output = self.vqa_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # ใช้ Greedy search เพื่อความแม่นยำในงานหุ่นยนต์
            )

        # ตัดส่วนที่เป็น Prompt ออกเพื่อให้เหลือแค่คำบรรยาย
        decoded = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt) :].strip()

    def get_image_embedding(self, image: Image.Image) -> list:
        """สกัดเวกเตอร์ CLIP (512-dim) สำหรับ Semantic Search"""
        embedding = self.embed_model.encode(image)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def encode_object(self, image_np: np.ndarray) -> tuple[str, list]:
        """ประมวลผลวัตถุเดี่ยว (BGR NumPy -> Caption & Embedding)"""
        if image_np is None or image_np.size == 0:
            return "invalid_image", [0.0] * 512

        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        caption = self.generate_caption(pil_img)
        embedding = self.get_image_embedding(pil_img)

        return caption, embedding

    def encode_batch(
        self, images_np: list[np.ndarray]
    ) -> tuple[list[str], list[list[float]]]:
        """ประมวลผลหลายวัตถุพร้อมกัน ดึงพลัง RTX 5090 ออกมาใช้เต็มที่"""
        if not images_np:
            return [], []

        # 1. เตรียมภาพ
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_np
        ]

        # 2. Batch Captioning กับ PaliGemma
        prompt = "caption en\n"
        # สร้าง List ของ Prompt ให้เท่ากับจำนวนภาพ
        prompts = [prompt] * len(pil_images)

        inputs = self.vqa_processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,  # จำเป็นต้องมี Padding เมื่อรันเป็น Batch
        ).to(self.device)

        with torch.no_grad():
            output = self.vqa_model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )

        # Decode และลบ Prompt ออกจากทุกลูกภาพ
        decoded_outputs = self.vqa_processor.batch_decode(
            output, skip_special_tokens=True
        )
        captions = [d[len(prompt) :].strip() for d in decoded_outputs]

        # 3. Batch Embedding กับ CLIP
        embeddings = self.embed_model.encode(
            pil_images,
            batch_size=len(pil_images),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return captions, embeddings.tolist()
