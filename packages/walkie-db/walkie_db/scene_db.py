import math
from typing import List, Optional, TypedDict

from .vector_db import BaseVectorDB


class SceneRecord(TypedDict):
    """Schema สำหรับบันทึกข้อมูลสถานที่"""

    name: str
    x: float
    y: float
    heading: float


class SceneVectorDB(BaseVectorDB):
    """
    ระบบจัดการความจำเชิงสถานที่ (Spatial Memory)
    ใช้สำหรับจดจำห้องและพิกัดบนแผนที่ SLAM
    """

    def __init__(self):
        super().__init__(collection_name="scenes_location")

    def add_scene(self, scene_id: str, scene_data: SceneRecord):
        """บันทึกสถานที่ใหม่ลงในฐานข้อมูล"""
        # เราใช้พิกัด [x, y] เป็น embedding แบบง่ายเพื่อให้ค้นหาด้วยระยะทางได้
        self.add(
            ids=[scene_id],
            embeddings=[[scene_data["x"], scene_data["y"]]],
            metadatas=[{"name": scene_data["name"], "heading": scene_data["heading"]}],
            documents=[scene_data["name"]],
        )

    def find_scene_by_name(self, name: str):
        """ค้นหาพิกัดสถานที่จากชื่อห้อง"""
        results = self.collection.get(where={"name": name}, limit=1)
        if results and results["ids"]:
            return {
                "id": results["ids"][0],
                "x": results["embeddings"][0][0] if results["embeddings"] else None,
                "y": results["embeddings"][0][1] if results["embeddings"] else None,
                "heading": results["metadatas"][0]["heading"],
            }
        return None

    def find_scenes_by_slam_coords(self, x: float, y: float, radius: float = 1.0):
        """
        ค้นหาว่าพิกัดปัจจุบัน (SLAM) อยู่ในห้องไหน
        โดยใช้รัศมีวงกลม (Radius) ในการระบุขอบเขต
        """
        # ค้นหาจุดที่ใกล้พิกัด [x, y] ที่สุด
        results = self.query(query_embeddings=[[x, y]], n_results=5)

        filtered_results = []
        if results and results["ids"]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                # กรองเฉพาะจุดที่อยู่ในระยะรัศมีที่กำหนด
                if math.sqrt(dist) <= radius:
                    filtered_results.append(
                        {
                            "id": results["ids"][0][i],
                            "name": results["metadatas"][0][i]["name"],
                            "distance": math.sqrt(dist),
                        }
                    )

        return filtered_results
