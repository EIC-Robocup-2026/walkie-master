import pytest
# Import มาให้ครบทั้ง Records และ DB classes
from walkie_db import (
    ObjectVectorDB, ObjectRecord,
    SceneVectorDB, SceneRecord,
    PeopleVectorDB, PersonRecord
)

def test_object_persistence(test_db_path):
    """ทดสอบ Schema ใหม่: YOLO + BLIP + CLIP"""
    db = ObjectVectorDB(persist_directory=test_db_path)

    record = ObjectRecord(
        object_id="cup_001",
        object_xyz=[1.0, 2.0, 0.5],
        object_embedding=[0.1] * 512,
        label="starbucks_cup",
        yolo_class="cup",            # ฟิลด์ใหม่
        caption="a red coffee cup with a white lid" # ฟิลด์ใหม่จาก BLIP
    )

    db.add_object(record)

    # ทดสอบการค้นหาด้วยภาพ (Vector)
    results = db.query_objects_by_image([0.1] * 512, n_results=1)
    assert results[0]["yolo_class"] == "cup"
    assert "red coffee cup" in results[0]["caption"]

def test_scene_semantic_search(test_db_path):
    """ทดสอบการค้นหาสถานที่ด้วยความหมาย (Semantic Scene)"""
    db = SceneVectorDB(persist_directory=test_db_path)

    scene = SceneRecord(
        scene_id="kitchen_01",
        scene_xyz=[10.5, 5.0, 0.0],
        scene_name="kitchen",
        description="a modern kitchen with a large fridge" # ฟิลด์ใหม่
    )

    db.add_scene(scene)

    # ค้นหาด้วยชื่อหรือคำอธิบาย
    res = db.query_scenes_by_text("Where is the fridge?")
    assert len(res["ids"]) > 0
    assert res["metadatas"][0][0]["name"] == "kitchen"


def test_people_memory(test_db_path):
    """เช็คการจำหน้าคนและข้อมูลบุคคล"""
    # คราวนี้ NameError จะหายไปเพราะเรา Import PeopleVectorDB มาแล้วด้านบน
    db = PeopleVectorDB(persist_directory=test_db_path)

    person = PersonRecord(
        person_id="user_vince",
        person_name="Vince",
        face_embedding=[0.5] * 128,
        person_info="Tutor for Computer Olympiad"
    )

    db.add_person(person)

    # ค้นหาด้วยใบหน้า
    hits = db.query_by_face([0.5] * 128)
    assert len(hits) > 0
    assert hits[0]["person_id"] == "user_vince"
    assert hits[0]["name"] == "Vince"
    assert "Olympiad" in hits[0]["info"]
