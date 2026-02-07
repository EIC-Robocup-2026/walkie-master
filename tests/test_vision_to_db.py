import os

import cv2
import numpy as np
from PIL import Image
from walkie_db.object_db import ObjectVectorDB
from walkie_db.people_db import PeopleVectorDB

# Import จาก Workspace Packages
from walkie_sdk import WalkieRobot
from walkie_vision.detector import VisionDetector
from walkie_vision.encoder import VisionEncoder
from walkie_vision.face_id import FaceIdentifier


def main():
    print("🔍 เริ่มต้นการทดสอบ Vision to DB Integration...")

    # 1. Setup Paths (อ้างอิงจากโครงสร้าง data/models ที่เราวางแผนไว้)
    MODEL_DIR = "data/models"
    SAM_CHECKPOINT = os.path.join(MODEL_DIR, "sam2.1_h14.pt")
    YOLO_CHECKPOINT = os.path.join(MODEL_DIR, "yolo11x-cls.pt")

    # 2. Initialize Modules
    print("⚙️ กำลังโหลดโมเดล AI...")
    detector = VisionDetector(
        sam_checkpoint=SAM_CHECKPOINT, yolo_checkpoint=YOLO_CHECKPOINT
    )
    face_id = FaceIdentifier()
    encoder = VisionEncoder()

    obj_db = ObjectVectorDB()  # บันทึกลง data/chromadb โดยอัตโนมัติ
    people_db = PeopleVectorDB()

    # 3. Get Image (ลองดึงจากหุ่นยนต์ ถ้าไม่พบให้ใช้รูปทดสอบ)
    print("📸 กำลังดึงภาพจากกล้อง...")
    # อ้างอิงการใช้งาน camera.get_frame() จาก walkie-sdk
    # สมมติ IP หุ่นยนต์เป็น 127.0.0.1 สำหรับการทดสอบ Local หรือใส่ IP จริง
    try:
        with WalkieRobot(ip="192.168.1.100") as bot:
            frame = bot.camera.get_frame()
    except Exception:
        print("⚠️ ไม่พบหุ่นยนต์ ใช้ภาพจำลองสำหรับการทดสอบ")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Test Person & Cup",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    if frame is None:
        print("❌ ไม่สามารถดึงภาพได้")
        return

    # 4. Vision Processing & DB Storage
    # ขั้นตอน Detection และ Segmentation
    print("🤖 AI กำลังวิเคราะห์ภาพ...")
    cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_rgb)

    # 4.1 ตรวจจับวัตถุและบันทึกลง Object DB
    detections = detector.detect_and_segment(frame)
    for det in detections:
        img_embedding = encoder.get_image_embedding(det["crop"])
        # บันทึกลง objects_image และ objects_location
        obj_db.add_object(
            object_id=f"obj_{det['label']}_{np.random.randint(1000)}",
            image_embedding=img_embedding,
            metadata={"label": det["label"], "confidence": det["confidence"]},
        )
        print(f"✅ บันทึกวัตถุ: {det['label']} ลง DB แล้ว")

    # 4.2 ตรวจจับใบหน้าและบันทึกลง People DB
    faces = face_id.extract_face_data(frame)
    for face in faces:
        # บันทึกลง people_face collection
        people_db.add_person(
            person_id="person_test_01",
            name="Unknown",
            face_embedding=face["embedding"],
            metadata={"gender": face["gender"], "age": face["age"]},
        )
        print(f"✅ บันทึกใบหน้า (Age: {face['age']}) ลง DB แล้ว")

    # 5. Verification (Query ข้อมูลกลับมาเช็ค)
    print("\n🔎 กำลังตรวจสอบข้อมูลใน Database...")
    results = obj_db.query_objects_by_image(img_embedding, n_results=1)
    if results:
        print(f"🎯 ค้นหาพบวัตถุที่ใกล้เคียงที่สุด: {results['metadatas'][0][0]['label']}")

    print("\n✨ การทดสอบ Integration สำเร็จ!")


if __name__ == "__main__":
    main()
