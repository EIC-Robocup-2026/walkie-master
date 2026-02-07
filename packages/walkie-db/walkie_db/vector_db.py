import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings


class BaseVectorDB:
    """
    คลาสฐานสำหรับจัดการ Vector Database ในโปรเจค walkie-master
    ทำหน้าที่เชื่อมต่อและจัดการ Collection พื้นฐาน
    """

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        # กำหนด Path ให้ชี้ไปที่ root/data/chromadb เสมอ
        if persist_directory is None:
            # ใช้พิกัดปัจจุบันเป็นหลักแล้วถอยกลับไปหาโฟลเดอร์ data
            self.persist_directory = os.path.abspath(
                os.path.join(os.getcwd(), "data", "chromadb")
            )
        else:
            self.persist_directory = persist_directory

        # ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(self.persist_directory, exist_ok=True)

        # เชื่อมต่อกับ ChromaDB แบบ Persistent
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None,
    ):
        """เพิ่มข้อมูลลงใน Collection"""
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 1,
        where: Optional[Dict] = None,
    ):
        """ค้นหาข้อมูลที่ใกล้เคียงที่สุด"""
        return self.collection.query(
            query_embeddings=query_embeddings, n_results=n_results, where=where
        )

    def delete(self, ids: List[str]):
        """ลบข้อมูลตาม ID"""
        self.collection.delete(ids=ids)

    def get_all(self):
        """ดึงข้อมูลทั้งหมดใน Collection"""
        return self.collection.get()
