from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import จาก walkie_db ที่เราปรับปรุงล่าสุด
from walkie_db import (
    ObjectRecord,
    ObjectVectorDB,
    PeopleVectorDB,
    PersonRecord,
    SceneRecord,
    SceneVectorDB,
)


class AgentIntegration:
    """
    คลาสรวมศูนย์สำหรับ Agent เพื่อจัดการความจำ (Memory Integration)
    ทำหน้าที่ประสานงานระหว่าง Vision, Database และ Control
    """

    def __init__(self, base_db_path: str = "data/chromadb"):
        # แยก Directory ย่อยสำหรับแต่ละ DB เพื่อความระเบียบ
        self.object_db = ObjectVectorDB(persist_directory=f"{base_db_path}/objects")
        self.scene_db = SceneVectorDB(persist_directory=f"{base_db_path}/scenes")
        self.people_db = PeopleVectorDB(persist_directory=f"{base_db_path}/people")

    # --- 1. Vision -> Database (Storage) ---

    # แก้ไขส่วนฟังก์ชัน process_object_detection ใน AgentIntegration
    def process_object_detection(
        self,
        object_id: str,
        xyz: Sequence[float],
        embedding: List[float],
        label: str,
        yolo_class: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> None:
        """บันทึกวัตถุที่ผ่านการประมวลผลจาก SAM -> YOLO -> BLIP -> CLIP"""
        record = ObjectRecord(
            object_id=object_id,
            object_xyz=xyz,
            object_embedding=embedding,
            label=label,
            yolo_class=yolo_class,
            caption=caption,
        )
        self.object_db.add_object(record)
        print(f"📦 Agent: Remembered '{label}' ({yolo_class}) at {xyz}")
        if caption:
            print(f"📝 Caption: {caption}")

    def process_scene_detection(
        self, scene_id: str, xyz: Sequence[float], name: str
    ) -> None:
        """บันทึกสถานที่ที่หุ่นยนต์อยู่"""
        record = SceneRecord(scene_id=scene_id, scene_xyz=xyz, scene_name=name)
        self.scene_db.add_scene(record)
        print(f"🏠 Agent: Marked scene '{name}' at {xyz}")

    def process_people_detection(
        self, person_id: str, name: str, face_embedding: List[float], info: str = ""
    ) -> None:
        """บันทึกคนที่มีปฏิสัมพันธ์ด้วย"""
        record = PersonRecord(
            person_id=person_id,
            person_name=name,
            face_embedding=face_embedding,
            person_info=info,
        )
        self.people_db.add_person(record)
        print(f"👤 Agent: Recognized {name} - {info}")

    # --- 2. Database -> Agent (Retrieval) ---

    def find_objects(self, query_emb: List[float], n: int = 3):
        """ค้นหาวัตถุด้วยภาพ (Similarity Search)"""
        return self.object_db.query_objects_by_image(query_emb, n_results=n)

    def identify_current_room(self, current_xyz: Sequence[float]) -> Optional[str]:
        """เช็คว่าพิกัดปัจจุบันคือห้องอะไร"""
        scenes = self.scene_db.find_scenes_by_slam_coords(
            current_xyz[0], current_xyz[1], current_xyz[2], radius=1.0
        )
        return scenes[0]["name"] if scenes else "Unknown Room"

    def identify_person(self, face_emb: List[float]) -> Optional[Dict]:
        """เช็คว่าหน้าที่เห็นคือใคร"""
        hits = self.people_db.query_by_face(face_emb, n_results=1)
        return hits[0] if hits else None

    # --- 3. Retrieval -> Navigation (Coordinates) ---

    def get_target_coords(
        self, target_type: str, target_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """ดึงพิกัด XYZ สำหรับสั่งหุ่นยนต์เคลื่อนที่"""
        if target_type == "object":
            # ค้นหา Metadata ของวัตถุ
            all_objs = self.object_db.collection.get(
                ids=[target_id], include=["metadatas"]
            )
            if all_objs["metadatas"]:
                m = all_objs["metadatas"][0]
                return (m["x"], m["y"], m["z"])

        elif target_type == "scene":
            res = self.scene_db.find_scene_by_name(target_id)
            if res:
                return (res["x"], res["y"], res["z"])

        return None


# --- 🧪 Example Usage ---


def example_flow():
    agent = AgentIntegration()

    # จำลอง: Vision เห็นถ้วยกาแฟ
    agent.process_object_detection(
        "cup_01", [1.5, 0.8, 0.7], [0.1] * 512, "Starbucks Cup"
    )

    # จำลอง: หุ่นยนต์เดินเข้าห้องครัว
    agent.process_scene_detection("kitchen_zone", [2.0, 3.0, 0.0], "Kitchen")

    # จำลอง: การถามหาพิกัดเพื่อเดินไปหาของ
    target = agent.get_target_coords("object", "cup_01")
    if target:
        print(f"\n🚀 Navigation Command: Move base to {target}")


if __name__ == "__main__":
    example_flow()
