from __future__ import annotations
import math
import chromadb
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class SceneRecord:
    scene_id: str
    scene_xyz: Sequence[float]
    scene_name: str
    description: Optional[str] = None  # เพิ่ม: คำอธิบายเชิง Semantic เช่น "ห้องครัวที่มีโต๊ะไม้"
    scene_embedding: Optional[Sequence[float]] = None

class SceneVectorDB:
    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="scenes_location",
            metadata={"hnsw:space": "l2"} # ใช้ L2 สำหรับพิกัดทางกายภาพ (Euclidean)
        )

    def add_scene(self, record: SceneRecord):
            """บันทึกสถานที่พร้อมคำอธิบายเชิงความหมาย"""
            x, y, z = record.scene_xyz
            spatial_emb = [float(x), float(y), float(z)]

            self.collection.upsert(
                ids=[record.scene_id],
                embeddings=[record.scene_embedding if record.scene_embedding else spatial_emb],
                metadatas={
                    "name": record.scene_name,
                    "description": record.description or "", # เก็บคำอธิบายลง Metadata
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                }
            )

    def query_scenes_by_text(self, query_text: str, n_results: int = 3):
        """ค้นหาสถานที่ด้วยภาษาธรรมชาติ เช่น 'where is the messy room?'"""
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas"]
        )

    def find_scenes_by_slam_coords(self, x: float, y: float, z: float = 0.0, radius: float = 1.0):
        """ค้นหาสถานที่ใกล้เคียงพิกัด SLAM"""
        results = self.collection.query(
            query_embeddings=[[float(x), float(y), float(z)]],
            n_results=5,
            include=["metadatas", "distances"]
        )

        filtered = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = math.sqrt(results["distances"][0][i])
                if dist <= radius:
                    item = dict(results["metadatas"][0][i])
                    item["scene_id"] = results["ids"][0][i]
                    item["distance"] = dist
                    filtered.append(item)
        return filtered
