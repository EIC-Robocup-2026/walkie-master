from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb


@dataclass
class ObjectRecord:
    object_id: str
    object_xyz: Sequence[float]
    object_embedding: Sequence[float]  # จาก CLIP
    label: str = "unknown"  # ชื่อทั่วไป
    yolo_class: Optional[str] = None  # Class ที่ได้จาก YOLO
    caption: Optional[str] = None  # คำอธิบายจาก BLIP
    scene_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ObjectVectorDB:
    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="objects_memory", metadata={"hnsw:space": "cosine"}
        )

    def add_object(self, record: ObjectRecord):
        """บันทึกข้อมูลวัตถุพร้อม Metadata ใหม่ (Caption/YOLO)"""
        x, y, z = record.object_xyz

        meta = {
            "label": record.label,
            "yolo_class": record.yolo_class or "unknown",
            "caption": record.caption or "",
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
            metadatas=[meta],
        )

    # ฟังก์ชันเสริม: ค้นหาด้วย Keyword จาก Caption
    def query_objects_by_text(self, query_text: str, n_results: int = 5):
        """ค้นหาวัตถุจากคำอธิบาย (Metadata Filter/Search)"""
        return self.collection.query(
            query_texts=[query_text], n_results=n_results, include=["metadatas"]
        )
