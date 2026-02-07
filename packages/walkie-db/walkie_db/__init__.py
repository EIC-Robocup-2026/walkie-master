from .object_db import ObjectRecord, ObjectVectorDB
from .people_db import PersonRecord, PeopleVectorDB
from .scene_db import SceneRecord, SceneVectorDB

# นำ BaseVectorDB ออก เพราะเราไม่ต้องการให้คนภายนอกเรียกใช้ (หรือเราลบไฟล์ทิ้งไปแล้ว)

__all__ = [
    "ObjectRecord",
    "ObjectVectorDB",
    "PersonRecord",
    "PeopleVectorDB",
    "SceneRecord",
    "SceneVectorDB",
]
