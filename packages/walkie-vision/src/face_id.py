import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceIdentifier:
    """
    Module สำหรับทำ Face Embedding โดยใช้ ArcFace (InsightFace).
    ออกแบบมาเพื่อทำงานร่วมกับ PeopleVectorDB สำหรับการจำเจ้าของบ้าน (Re-ID).
    """

    def __init__(self, model_name="buffalo_l", ctx_id=0):
        # buffalo_l คือโมเดลขนาดใหญ่ที่แม่นยำที่สุด รองรับการรันบน GPU (RTX 5090 Ready)
        # providers จะเรียงลำดับการใช้ CUDA ก่อน ถ้าไม่มีจะสลับไป CPU
        self.app = FaceAnalysis(
            name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # det_size คือขนาดภาพที่ใช้ตรวจจับใบหน้า (640x640 เป็นค่ามาตรฐานที่สมดุล)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def extract_face_data(self, frame):
        """
        รับภาพสดจากหุ่นยนต์ (BGR numpy array จาก walkie-sdk)
        และคืนค่าข้อมูลใบหน้าทั้งหมดที่ตรวจพบ
        """
        faces = self.app.get(frame)
        results = []

        for face in faces:
            # face.normed_embedding คือ vector 512 มิติที่พร้อมบันทึกลง ChromaDB
            results.append(
                {
                    "embedding": face.normed_embedding.tolist(),
                    "bbox": face.bbox.astype(int).tolist(),
                    "landmark": face.kps.astype(int).tolist(),
                    "gender": "M" if face.gender == 1 else "F",
                    "age": face.age,
                    "score": float(face.det_score),
                }
            )

        return results

    @staticmethod
    def get_similarity(feat1, feat2):
        """คำนวณความเหมือน (Cosine Similarity) ระหว่าง 2 ใบหน้า (0.0 - 1.0)"""
        feat1 = np.array(feat1)
        feat2 = np.array(feat2)
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
