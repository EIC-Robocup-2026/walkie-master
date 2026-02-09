import numpy as np
from insightface.app import FaceAnalysis


class FaceIdentifier:
    """
    Module สำหรับทำ Face Embedding โดยใช้ ArcFace (InsightFace).
    ออกแบบมาเพื่อทำงานร่วมกับ PeopleVectorDB สำหรับการจำเจ้าของบ้าน (Re-ID).
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        # buffalo_l คือโมเดลที่แม่นยำสูง รองรับ CUDA สำหรับ RTX 5090
        self.app = FaceAnalysis(
            name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def extract_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        ตรวจจับและสกัดฟีเจอร์ใบหน้า คืนค่าข้อมูลที่พร้อมบันทึกลง PeopleVectorDB
        """
        faces = self.app.get(frame)
        results = []

        for face in faces:
            # ตรวจสอบคุณภาพเบื้องต้น (Detection Score)
            if face.det_score < 0.5:
                continue

            results.append(
                {
                    "embedding": face.normed_embedding.tolist(),  # พร้อมสำหรับ ChromaDB
                    "bbox": face.bbox.astype(int).tolist(),
                    "kps": face.kps.astype(int).tolist(),
                    "gender": "M" if face.gender == 1 else "F",
                    "age": int(face.age),
                    "score": float(face.det_score),
                    "crop": self._crop_face(frame, face.bbox),
                }
            )

        return results

    def _crop_face(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """ตัดภาพใบหน้าเพื่อใช้เก็บเป็น Thumbnail หรือแสดงผล"""
        x1, y1, x2, y2 = bbox.astype(int)
        # ป้องกันพิกัดออกนอกขอบภาพ
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def get_similarity(feat1: List[float], feat2: List[float]) -> float:
        """คำนวณความเหมือนระหว่าง 2 ใบหน้า"""
        f1 = np.array(feat1)
        f2 = np.array(feat2)
        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)))
