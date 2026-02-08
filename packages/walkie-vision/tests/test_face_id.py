import json
import os
from datetime import datetime

import cv2
import pytest
from walkie_db.people_db import PeopleVectorDB, PersonRecord

from walkie_vision.face_id import FaceIdentifier


@pytest.fixture
def face_identifier():
    """โหลดโมเดล Face Recognition โดยรองรับ CUDA สำหรับ RTX 5090"""
    return FaceIdentifier(ctx_id=0)


@pytest.fixture
def people_db(tmp_path):
    """จำลองฐานข้อมูลบุคคล (ChromaDB) สำหรับการทดสอบ"""
    # ใช้ tmp_path เพื่อให้ database ถูกล้างใหม่ทุกครั้งที่รัน test case
    return PeopleVectorDB(persist_directory=str(tmp_path))


@pytest.fixture
def test_data_dir():
    """ระบุตำแหน่งโฟลเดอร์ที่เก็บรูปภาพทดสอบ"""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def output_root():
    """สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์การทดสอบแยกตาม timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(__file__), "outputs", f"face_batch_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


# ใช้ Parametrize เพื่อรัน Test Case แยกกันสำหรับทั้ง 5 รูป
@pytest.mark.parametrize(
    "image_filename",
    [
        "test_people_1.jpg",
        "test_people_2.jpg",
        "test_people_3.jpg",
        "test_people_4.jpg",
        "test_people_5.jpg",
    ],
)
def test_face_to_db_integration_batch(
    face_identifier, people_db, test_data_dir, output_root, image_filename
):
    """
    Integration Test: ทดสอบ Pipeline ตั้งแต่การสกัดใบหน้าไปจนถึงการบันทึกและค้นหาใน DB
    """
    img_path = os.path.join(test_data_dir, image_filename)

    # ตรวจสอบว่ามีไฟล์อยู่จริงเพื่อป้องกันการ Skipped โดยไม่ทราบสาเหตุ
    if not os.path.exists(img_path):
        pytest.skip(f"ไม่พบไฟล์ทดสอบ: {img_path} (กรุณาตรวจสอบว่าวางไฟล์ในโฟลเดอร์ data แล้ว)")

    frame = cv2.imread(img_path)
    if frame is None:
        pytest.fail(f"อ่านไฟล์ภาพไม่ได้: {img_path}")

    # 1. ตรวจจับและสกัดฟีเจอร์ใบหน้า
    face_results = face_identifier.extract_faces(frame)

    # หากรูปนั้นไม่มีคนอยู่เลย ให้แจ้งเตือนแต่ไม่ต้องให้ Test พัง
    if len(face_results) == 0:
        pytest.xfail(f"ไม่พบใบหน้าใน {image_filename} (ตรวจสอบภาพตัวอย่าง)")

    # สร้างโครงสร้างโฟลเดอร์เก็บผลลัพธ์แยกตามชื่อไฟล์รูปภาพ
    case_name = os.path.splitext(image_filename)[0]
    case_output_dir = os.path.join(output_root, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    summary = []
    for i, face_data in enumerate(face_results):
        # สร้าง ID เฉพาะสำหรับแต่ละใบหน้าในแต่ละไฟล์
        person_id = f"{case_name}_p{i}"

        # 2. เตรียม Record ตาม Schema ของ People DB
        record = PersonRecord(
            person_id=person_id,
            face_embedding=face_data["embedding"],
            person_name=f"Subject_{person_id}",
            person_info=f"Detected from {image_filename} at index {i}",
            metadata={
                "age": face_data["age"],
                "gender": face_data["gender"],
                "det_score": face_data["score"],
            },
        )

        # 3. บันทึกลงฐานข้อมูล
        people_db.add_person(record)

        # 4. บันทึกรูป Crop ใบหน้าเพื่อตรวจสอบด้วยตา (Manual Verification)
        crop_filename = f"face_idx_{i}.jpg"
        cv2.imwrite(os.path.join(case_output_dir, crop_filename), face_data["crop"])

        # 5. ทดสอบการค้นหา (Vector Search) เพื่อยืนยันว่าข้อมูลถูกบันทึกและดึงกลับมาได้ถูกต้อง
        search_results = people_db.query_by_face(face_data["embedding"], n_results=1)

        assert len(search_results) > 0, "Query ไม่พบข้อมูลที่เพิ่งบันทึก"
        assert search_results[0]["person_id"] == person_id, (
            f"ID ไม่ตรงกัน: คาดหวัง {person_id} แต่พบ {search_results[0]['person_id']}"
        )

        summary.append(
            {
                "person_id": person_id,
                "similarity_distance": search_results[0]["distance"],
                "metadata": record.metadata,
            }
        )

    # บันทึก Metadata ทั้งหมดเป็น JSON สรุปผลของภาพนั้นๆ
    with open(os.path.join(case_output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"✅ ทดสอบรูป {image_filename} สำเร็จ: พบทั้งหมด {len(face_results)} ใบหน้า")
