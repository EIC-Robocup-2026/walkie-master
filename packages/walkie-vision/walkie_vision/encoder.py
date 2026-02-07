import clip
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class VisionEncoder:
    def __init__(self):
        # โหลด BLIP สำหรับ Captioning
        self.blip_proc = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # โหลด CLIP สำหรับ Vector Embedding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device
        )

    def encode_object(self, cropped_image_np):
        """สร้างทั้ง Caption และ Embedding"""
        pil_img = Image.fromarray(cv2.cvtColor(cropped_image_np, cv2.COLOR_BGR2RGB))

        # 1. Generate Caption ด้วย BLIP
        inputs = self.blip_proc(pil_img, return_tensors="pt")
        out = self.blip_model.generate(**inputs)
        caption = self.blip_proc.decode(out[0], skip_special_tokens=True)

        # 2. Generate Embedding ด้วย CLIP
        image_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = (
                self.clip_model.encode_image(image_input).cpu().numpy().flatten()
            )

        return caption, embedding
