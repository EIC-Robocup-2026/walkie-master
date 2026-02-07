import pytest

from walkie_db import ObjectVectorDB, SceneVectorDB


def test_object_persistence(test_db_path):
    """เช็คการบันทึกและดึงข้อมูลวัตถุ"""
    db = ObjectVectorDB(persist_directory=test_db_path)

    test_id = "cup_001"
    test_vector = [0.1] * 512

    db.add_object(test_id, test_vector, {"label": "cup"})

    results = db.query_objects_by_image(test_vector, n_results=1)
    assert results["ids"][0][0] == test_id


def test_scene_location(test_db_path):
    """เช็คความแม่นยำของพิกัด SLAM ใน SceneDB"""
    db = SceneVectorDB()
    # เปลี่ยน persist_directory ชั่วคราวผ่านตัวแปรภายในถ้าจำเป็น
    # (หรือปรับ SceneVectorDB ให้รับ path ใน __init__)

    db.add_scene("loc_1", {"name": "kitchen", "x": 10.5, "y": 5.0, "heading": 1.5})

    # ค้นหาด้วยพิกัดที่ใกล้เคียง
    res = db.find_scenes_by_slam_coords(10.4, 4.9, radius=0.5)
    assert len(res) > 0
    assert res[0]["name"] == "kitchen"
