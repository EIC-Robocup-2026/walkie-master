from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb


@dataclass
class PersonRecord:
    person_id: str
    people_xyz: Sequence[float]
    face_embedding: Optional[Sequence[float]] = None
    pose_embedding: Optional[Sequence[float]] = None
    timeframe: Optional[str] = None
    chat_history_ref: Optional[str] = None


class PeopleVectorDB:
    """Database for people storage"""

    def __init__(self, persist_directory: str = "chroma_people_db") -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)

        self._face_col = self._client.get_or_create_collection(
            name="people_face",
            metadata={"hnsw:space": "cosine"},
        )

        self._pose_col = self._client.get_or_create_collection(
            name="people_pose",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, record: PersonRecord) -> None:
        """Store a person"""
        xyz = list(record.people_xyz)
        if len(xyz) != 3:
            raise ValueError("people_xyz must have length 3")

        base_meta = {
            "person_id": record.person_id,
            "people_x": float(xyz[0]),
            "people_y": float(xyz[1]),
            "people_z": float(xyz[2]),
        }

        if record.timeframe:
            base_meta["timeframe"] = record.timeframe
        if record.chat_history_ref:
            base_meta["chat_history_ref"] = record.chat_history_ref

        if record.face_embedding:
            self._face_col.upsert(
                ids=[f"{record.person_id}::face"],
                embeddings=[list(record.face_embedding)],
                metadatas=[{**base_meta, "embedding_type": "face"}],
                documents=[""],
            )

        if record.pose_embedding:
            self._pose_col.upsert(
                ids=[f"{record.person_id}::pose"],
                embeddings=[list(record.pose_embedding)],
                metadatas=[{**base_meta, "embedding_type": "pose"}],
                documents=[""],
            )

    def delete_person(self, person_id: str) -> None:
        """Delete a person"""
        self._face_col.delete(ids=[f"{person_id}::face"])
        self._pose_col.delete(ids=[f"{person_id}::pose"])

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get person by ID"""
        result = self._face_col.get(
            ids=[f"{person_id}::face"],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            result = self._pose_col.get(
                ids=[f"{person_id}::pose"],
                include=["metadatas"],
            )

        if not result or not result.get("metadatas"):
            return None

        meta = dict(result["metadatas"][0])
        meta["people_xyz"] = [
            meta.get("people_x"),
            meta.get("people_y"),
            meta.get("people_z"),
        ]
        return meta

    def query_by_face_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query people by face embedding"""
        result = self._face_col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            hit = dict(meta)
            hit["people_xyz"] = [
                hit.get("people_x"),
                hit.get("people_y"),
                hit.get("people_z"),
            ]
            hit["distance"] = float(dist)
            hits.append(hit)

        return hits

    def query_by_pose_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query people by pose embedding"""
        result = self._pose_col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            hit = dict(meta)
            hit["people_xyz"] = [
                hit.get("people_x"),
                hit.get("people_y"),
                hit.get("people_z"),
            ]
            hit["distance"] = float(dist)
            hits.append(hit)

        return hits

    def get_person_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get person by FaceID"""
        return self.get_person(face_id)


__all__ = ["PersonRecord", "PeopleVectorDB"]
