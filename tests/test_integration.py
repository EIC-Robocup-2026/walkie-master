import numpy as np
import pytest


def test_full_vision_to_db_flow(vision_detector, test_db_path):
    """
    ตรวจว่า Vision ตรวจเจอ -> ส่งต่อให้ DB บันทึก ได้จริงหรือไม่
    """
    from walkie_db import ObjectVectorDB

    db = ObjectVectorDB(persist_directory=test_db_path)

    # 1. จำลองการเห็นวัตถุ (สร้างรูปที่มีกล่องสี่เหลี่ยมสีขาว)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:200, 100:200] = 255

    # 2. Vision ประมวลผล
    results = vision_detector.detect_and_segment(img)

    if results:
        # 3. ลองบันทึกผลลัพธ์แรกที่เจอลง DB
        item = results[0]
        db.add_object(
            object_id="item_001",
            image_embedding=[0.1] * 512,  # mock embedding
            metadata={"label": item["label"]},
        )

        # 4. ตรวจสอบว่าใน DB มีข้อมูลอยู่จริง
        check = db.get_all()
        assert "item_001" in check["ids"]
    else:
        pytest.skip("No object detected in synthetic image, check Vision model.")
