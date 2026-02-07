from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb


@dataclass
class ObjectRecord:
    object_id: str
    object_xyz: Sequence[float]
    object_image_ref: str
    object_embedding: Sequence[float]
    location_embedding: Sequence[float]
    scene_id: Optional[str] = None


@dataclass
class SceneRecord:
    scene_id: str
    scene_xyz: Sequence[float]
    scene_image_ref: str
    scene_embedding: Sequence[float]


class ObjectVectorDB:
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)

        self._image_col = self._client.get_or_create_collection(
            name="objects_image",
            metadata={"hnsw:space": "cosine"},
        )
        self._location_col = self._client.get_or_create_collection(
            name="objects_location",
            metadata={"hnsw:space": "cosine"},
        )
        self._scene_col = self._client.get_or_create_collection(
            name="scenes",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, record: ObjectRecord) -> None:
        xyz = list(record.object_xyz)
        if len(xyz) != 3:
            raise ValueError("object_xyz must have length 3")

        base_meta = {
            "object_id": record.object_id,
            "object_x": float(xyz[0]),
            "object_y": float(xyz[1]),
            "object_z": float(xyz[2]),
            "object_image_ref": record.object_image_ref,
        }
        if record.scene_id:
            base_meta["scene_id"] = record.scene_id

        self._image_col.upsert(
            ids=[f"{record.object_id}::image"],
            embeddings=[list(record.object_embedding)],
            metadatas=[{**base_meta, "embedding_type": "image"}],
            documents=[""],
        )

        self._location_col.upsert(
            ids=[f"{record.object_id}::location"],
            embeddings=[list(record.location_embedding)],
            metadatas=[{**base_meta, "embedding_type": "location"}],
            documents=[""],
        )

    def delete_object(self, object_id: str) -> None:
        self._image_col.delete(ids=[f"{object_id}::image"])
        self._location_col.delete(ids=[f"{object_id}::location"])

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        result = self._image_col.get(
            ids=[f"{object_id}::image"],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            return None

        meta = dict(result["metadatas"][0])
        meta["object_xyz"] = [
            meta.get("object_x"),
            meta.get("object_y"),
            meta.get("object_z"),
        ]
        return meta

    def _query(
        self,
        query_embedding: Sequence[float],
        embedding_type: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        col = self._image_col if embedding_type == "image" else self._location_col

        result = col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where={"embedding_type": embedding_type},
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            hit = dict(meta)
            hit["object_xyz"] = [
                hit.get("object_x"),
                hit.get("object_y"),
                hit.get("object_z"),
            ]
            hit["distance"] = float(dist)
            hits.append(hit)

        return hits

    def query_by_image_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._query(
            query_embedding=query_embedding,
            embedding_type="image",
            n_results=n_results,
        )

    def query_by_location_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._query(
            query_embedding=query_embedding,
            embedding_type="location",
            n_results=n_results,
        )

    def upsert_scene(self, record: SceneRecord) -> None:
        """Store a scene"""
        xyz = list(record.scene_xyz)
        if len(xyz) != 3:
            raise ValueError("scene_xyz must have length 3")

        base_meta = {
            "scene_id": record.scene_id,
            "scene_x": float(xyz[0]),
            "scene_y": float(xyz[1]),
            "scene_z": float(xyz[2]),
            "scene_image_ref": record.scene_image_ref,
            "record_type": "scene",
        }

        self._scene_col.upsert(
            ids=[f"{record.scene_id}::scene"],
            embeddings=[list(record.scene_embedding)],
            metadatas=[base_meta],
            documents=[""],
        )

    def delete_scene(self, scene_id: str) -> None:
        """Delete a scene"""
        self._scene_col.delete(ids=[f"{scene_id}::scene"])

    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get scene by ID"""
        result = self._scene_col.get(
            ids=[f"{scene_id}::scene"],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            return None

        meta = dict(result["metadatas"][0])
        meta["scene_xyz"] = [
            meta.get("scene_x"),
            meta.get("scene_y"),
            meta.get("scene_z"),
        ]
        return meta

    def query_by_scene_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query scenes by embedding similarity"""
        result = self._scene_col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            hit = dict(meta)
            hit["scene_xyz"] = [
                hit.get("scene_x"),
                hit.get("scene_y"),
                hit.get("scene_z"),
            ]
            hit["distance"] = float(dist)
            hits.append(hit)

        return hits

    def find_scenes_by_slam_coords(
        self,
        query_xyz: Sequence[float],
        radius: float = 1.0,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find scenes near SLAM coordinates"""
        if len(query_xyz) != 3:
            raise ValueError("query_xyz must have length 3")

        all_scenes = self._scene_col.get(include=["metadatas"])

        if not all_scenes or not all_scenes.get("metadatas"):
            return []

        results: List[Dict[str, Any]] = []
        for meta in all_scenes["metadatas"]:
            scene_x = meta.get("scene_x", 0.0)
            scene_y = meta.get("scene_y", 0.0)
            scene_z = meta.get("scene_z", 0.0)

            dist = (
                (scene_x - query_xyz[0]) ** 2
                + (scene_y - query_xyz[1]) ** 2
                + (scene_z - query_xyz[2]) ** 2
            ) ** 0.5

            if dist <= radius:
                hit = dict(meta)
                hit["scene_xyz"] = [scene_x, scene_y, scene_z]
                hit["distance"] = dist
                results.append(hit)

        results.sort(key=lambda x: x["distance"])
        return results[:n_results]

    def get_objects_by_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        """Get all objects in a scene"""
        all_objects = []

        image_results = self._image_col.get(
            where={"scene_id": scene_id}, include=["metadatas"]
        )
        if image_results and image_results.get("metadatas"):
            for meta in image_results["metadatas"]:
                hit = dict(meta)
                hit["object_xyz"] = [
                    hit.get("object_x"),
                    hit.get("object_y"),
                    hit.get("object_z"),
                ]
                all_objects.append(hit)

        seen_ids = set()
        unique_objects = []
        for obj in all_objects:
            obj_id = obj.get("object_id")
            if obj_id and obj_id not in seen_ids:
                seen_ids.add(obj_id)
                unique_objects.append(obj)

        return unique_objects


__all__ = ["ObjectRecord", "SceneRecord", "ObjectVectorDB"]
