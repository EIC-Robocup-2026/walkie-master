from __future__ import annotations
import chromadb
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

@dataclass
class PersonRecord:
    person_id: str
    face_embedding: Sequence[float]  # Vector จาก Face Recognition (เช่น InsightFace)
    person_name: str
    person_info: str = ""           # ข้อมูลเพิ่มเติม เช่น "ชอบกินกาแฟ", "เป็นเจ้าของบ้าน"
    metadata: Optional[Dict[str, Any]] = None

class PeopleVectorDB:
    """
    ระบบฐานข้อมูลความจำเกี่ยวกับบุคคล (Long-term Person Memory)
    เน้นการระบุตัวตนด้วยใบหน้า (Face ID)
    """

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        # เลือกว่าจะบันทึกข้อมูลง Disk หรือใช้ Memory (สำหรับ Test)
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name="people_memory",
            metadata={"hnsw:space": "cosine"} # ใช้ Cosine Similarity สำหรับ Face Vector
        )

    def add_person(self, record: PersonRecord) -> None:
        """บันทึกข้อมูลบุคคลใหม่ลงในฐานข้อมูล"""

        # เตรียม Metadata
        base_meta = {
            "name": record.person_name,
            "info": record.person_info,
        }

        # รวม metadata เพิ่มเติมถ้ามี
        if record.metadata:
            base_meta.update(record.metadata)

        self._collection.upsert(
            ids=[record.person_id],
            embeddings=[list(record.face_embedding)],
            metadatas=[base_meta]
        )

    def query_by_face(self, query_embedding: Sequence[float], n_results: int = 1) -> List[Dict[str, Any]]:
        """ค้นหาบุคคลจาก Vector ใบหน้า"""
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"]
        )

        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                hit = dict(results["metadatas"][0][i])
                hit["person_id"] = results["ids"][0][i]
                hit["distance"] = float(results["distances"][0][i])
                formatted.append(hit)

        return formatted

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """ดึงข้อมูลบุคคลจาก ID"""
        result = self._collection.get(
            ids=[person_id],
            include=["metadatas"]
        )

        if not result or not result["metadatas"]:
            return None

        meta = dict(result["metadatas"][0])
        meta["person_id"] = person_id
        return meta

    def delete_person(self, person_id: str) -> None:
        """ลบข้อมูลบุคคล"""
        self._collection.delete(ids=[person_id])

    def get_all_people(self) -> List[Dict[str, Any]]:
        """ดึงรายชื่อบุคคลทั้งหมดในฐานข้อมูล"""
        results = self._collection.get(include=["metadatas"])

        output = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                item = dict(results["metadatas"][i])
                item["person_id"] = results["ids"][i]
                output.append(item)
        return output

__all__ = ["PersonRecord", "PeopleVectorDB"]
