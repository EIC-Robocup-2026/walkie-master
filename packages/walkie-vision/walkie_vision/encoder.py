import cv2
import numpy as np  # เพิ่มเพื่อแก้ NameError: name 'np' is not defined
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionEncoder:
    def __init__(
        self,
        vqa_model="Salesforce/blip-vqa-large",
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

    def ask_question(self, image: Image.Image, question: str) -> str:
        """Visual Question Answering (VQA)"""
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(
            self.device
        )
        out = self.vqa_model.generate(**inputs)
        return self.vqa_processor.decode(out[0], skip_special_tokens=True)

    def get_image_embedding(self, image: Image.Image) -> list:
        embedding = self.embed_model.encode(image)
        return embedding.tolist()

    def encode_object(self, image_np: np.ndarray) -> tuple[str, list]:
        """สกัด Caption และ Vector จากภาพ NumPy"""
        # แปลง BGR เป็น RGB ก่อนเข้า PIL
        pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        caption = self.ask_question(pil_img, "Describe this object in detail")
        embedding = self.get_image_embedding(pil_img)
        return caption, embedding
