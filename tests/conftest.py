import os
import sys
from pathlib import Path

import pytest

# 1. หาตำแหน่ง Root ของโปรเจค (walkie-master) เพื่อทำ Absolute Path
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 2. เพิ่ม path ของ packages เข้าไปเพื่อให้ pytest หาเจอโดยไม่ต้องติดตั้งก่อน
sys.path.append(str(ROOT_DIR / "packages/walkie-vision"))
sys.path.append(str(ROOT_DIR / "packages/walkie-db"))
sys.path.append(str(ROOT_DIR / "packages/walkie-sdk"))


@pytest.fixture(scope="session")
def project_root():
    return ROOT_DIR


@pytest.fixture(scope="session")
def model_paths(project_root):
    """ระบุตำแหน่งไฟล์โมเดลแบบเป๊ะๆ"""
    return {
        "sam_checkpoint": str(project_root / "data/models/sam2.1_h14.pt"),
        "yolo_checkpoint": str(project_root / "data/models/yolo11x-cls.pt"),
        "config_dir": str(project_root / "data/models/configs"),
    }


@pytest.fixture(scope="session")
def vision_detector(model_paths):
    """โหลด VisionDetector แค่ครั้งเดียวเพื่อประหยัด VRAM"""
    from walkie_vision import VisionDetector

    # ตรวจสอบไฟล์ก่อนโหลด
    if not os.path.exists(model_paths["sam_checkpoint"]):
        pytest.skip(f"Missing weight: {model_paths['sam_checkpoint']}")

    return VisionDetector(
        sam_checkpoint=model_paths["sam_checkpoint"],
        yolo_checkpoint=model_paths["yolo_checkpoint"],
    )


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """สร้างถังข้อมูลแยกสำหรับการ Test เท่านั้น (ไม่ทับ data/chromadb จริง)"""
    return str(tmp_path_factory.mktemp("test_db"))
