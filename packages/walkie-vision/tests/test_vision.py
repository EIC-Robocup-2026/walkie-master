def test_vision_integration_with_real_file(
    vision_detector, vision_encoder, test_data_path, output_path
):
    img_path = os.path.join(test_data_path, "test_room_2.jpg")
    if not os.path.exists(img_path):
        pytest.skip(f"Test file not found")

    frame = cv2.imread(img_path)

    # 1. Detection (ยังเป็น YOLO-only)
    objects = vision_detector.get_segmented_objects(frame)
    assert len(objects) > 0

    # --- 🔥 จุดที่เปลี่ยน: BATCH PROCESSING START ---

    # ดึงเฉพาะภาพทุกลูกออกมาเป็น List
    images_to_process = [obj["image"] for obj in objects]

    # ส่งประมวลผลใน GPU ทีเดียว (Batch Inference)
    print(f"\n🚀 Batch processing {len(images_to_process)} objects on GPU...")
    all_captions, all_embeddings = vision_encoder.encode_batch(images_to_process)

    # --- BATCH PROCESSING END ---

    test_results_summary = []

    # 2. วนลูปเพื่อเซฟไฟล์และจัดระเบียบข้อมูล (ตอนนี้ไม่ต้องรัน AI ในลูปแล้ว)
    for i, obj in enumerate(objects):
        caption = all_captions[i]
        embedding = all_embeddings[i]

        crop_filename = f"obj_{i}_{obj['yolo_class']}.jpg"
        cv2.imwrite(os.path.join(output_path, crop_filename), obj["image"])

        test_results_summary.append(
            {
                "index": i,
                "class": obj["yolo_class"],
                "caption": caption,
                "crop_path": crop_filename,
                "embedding_sample": embedding[:5],
            }
        )

        if i == 0:
            assert len(caption) > 0
            assert len(embedding) == 512

    # 3. บันทึก JSON
    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_results_summary, f, indent=4, ensure_ascii=False)

    print(f"✅ Batch Test Completed. Saved to: {output_path}")
