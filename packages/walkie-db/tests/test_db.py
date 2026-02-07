import pytest
# Import มาให้ครบทั้ง Records และ DB classes
from walkie_db import (
    ObjectVectorDB, ObjectRecord,
    SceneVectorDB, SceneRecord,
    PeopleVectorDB, PersonRecord
)

def test_object_persistence(test_db_path):
    """เช็คการบันทึกและดึงข้อมูลวัตถุ (Lean Schema)"""
    db = ObjectVectorDB(persist_directory=test_db_path)

    record = ObjectRecord(
        object_id="cup_001",
        object_xyz=[1.0, 2.0, 0.5],
        object_embedding=[0.1] * 512,
        label="red_cup"
    )

    db.add_object(record)
    results = db.query_objects_by_image([0.1] * 512, n_results=1)

    assert len(results) > 0
    assert results[0]["object_id"] == "cup_001"
    assert results[0]["label"] == "red_cup"


def test_scene_location(test_db_path):
    """เช็คความแม่นยำของพิกัด SLAM ใน SceneDB"""
    db = SceneVectorDB(persist_directory=test_db_path)

    scene = SceneRecord(
        scene_id="kitchen_01",
        scene_xyz=[10.5, 5.0, 0.0],
        scene_name="kitchen"
    )

    db.add_scene(scene)
    # ค้นหาด้วยพิกัด SLAM
    res = db.find_scenes_by_slam_coords(10.4, 4.9, 0.0, radius=0.5)

    assert len(res) > 0
    assert res[0]["name"] == "kitchen"


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
