import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionEncoder:
    """
    Module สำหรับการวิเคราะห์ภาพเชิงลึก (VQA) และการสร้าง Image Embedding
    สำหรับวัตถุทั่วไป เพื่อเก็บลงใน ObjectVectorDB.
    """

    def __init__(
        self,
        vqa_model="Salesforce/blip-vqa-large",
        embed_model="clip-ViT-B-32",
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. BLIP VQA สำหรับตอบคำถามจากภาพ เช่น "Is this cup dirty?"
        self.vqa_processor = BlipProcessor.from_pretrained(vqa_model)
        self.vqa_model = BlipForConditionalGeneration.from_pretrained(vqa_model).to(
            self.device
        )

        # 2. Sentence-Transformer (CLIP) สำหรับสร้าง Image Embedding
        # ใช้สร้าง vector เพื่อบันทึกลง objects_image collection
        self.embed_model = SentenceTransformer(embed_model, device=self.device)

    def ask_question(self, image: Image.Image, question: str) -> str:
        """ถามคำถามเกี่ยวกับภาพ (Visual Question Answering)"""
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(
            self.device
        )
        out = self.vqa_model.generate(**inputs)
        return self.vqa_processor.decode(out[0], skip_special_tokens=True)

    def get_image_embedding(self, image: Image.Image) -> list:
        """สร้าง Vector (Embedding) จากภาพวัตถุเพื่อใช้ในการค้นหา (Retrieval)"""
        # คืนค่าเป็น List เพื่อให้พร้อมบันทึกลง ChromaDB
        embedding = self.embed_model.encode(image)
        return embedding.tolist()
