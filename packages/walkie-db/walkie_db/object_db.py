from __future__ import annotations
import chromadb
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class ObjectRecord:
    object_id: str
    object_xyz: Sequence[float]
    object_embedding: Sequence[float]
    label: str = "unknown"
    scene_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ObjectVectorDB:
    def __init__(self, persist_directory: Optional[str] = None):
        # เลือกว่าจะเซฟลง Disk หรือรันใน RAM (สำหรับ Test)
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="objects_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def add_object(self, record: ObjectRecord):
        """บันทึกข้อมูลวัตถุ (Schema: ID, XYZ, Embedding, Label)"""
        x, y, z = record.object_xyz

        meta = {
            "label": record.label,
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }
        if record.scene_id:
            meta["scene_id"] = record.scene_id
        if record.metadata:
            meta.update(record.metadata)

        self.collection.upsert(
            ids=[record.object_id],
            embeddings=[list(record.object_embedding)],
            metadatas=[meta]
        )

    def query_objects_by_image(self, query_embedding: Sequence[float], n_results: int = 5):
        """ค้นหาวัตถุที่ใกล้เคียงจาก Image Vector"""
        results = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"]
        )

        # จัด Format ให้ใช้ง่าย
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                item = dict(results["metadatas"][0][i])
                item["object_id"] = results["ids"][0][i]
                item["distance"] = float(results["distances"][0][i])
                formatted.append(item)
        return formatted
