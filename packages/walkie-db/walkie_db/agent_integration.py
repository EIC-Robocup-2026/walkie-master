"""
Example integration with Agent and Robot system

Shows how databases work with agent flow:
- Vision detects objects, scenes, people
- Object&SceneFinding queries object database
- PeopleFinding queries people database
- ControlRobot uses coordinates for navigation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from object_db import ObjectRecord, ObjectVectorDB, SceneRecord
from people_db import PeopleVectorDB, PersonRecord


class AgentIntegration:
    """Example integration class for agent system"""

    def __init__(
        self,
        object_db_path: str = "chroma_db",
        people_db_path: str = "chroma_people_db",
    ):
        self.object_db = ObjectVectorDB(persist_directory=object_db_path)
        self.people_db = PeopleVectorDB(persist_directory=people_db_path)

    def process_object_detection(
        self,
        object_id: str,
        object_xyz: tuple[float, float, float],
        object_image_path: str,
        object_embedding: List[float],
        location_embedding: List[float],
    ) -> None:
        """Store detected objects"""
        obj = ObjectRecord(
            object_id=object_id,
            object_xyz=object_xyz,
            object_image_ref=object_image_path,
            object_embedding=object_embedding,
            location_embedding=location_embedding,
        )
        self.object_db.upsert(obj)
        print(f"Stored object: {object_id} at {object_xyz}")

    def process_scene_detection(
        self,
        scene_id: str,
        slam_xyz: tuple[float, float, float],
        scene_image_path: str,
        scene_embedding: List[float],
    ) -> None:
        """Store detected scenes"""
        scene = SceneRecord(
            scene_id=scene_id,
            scene_xyz=slam_xyz,
            scene_image_ref=scene_image_path,
            scene_embedding=scene_embedding,
        )
        self.object_db.upsert_scene(scene)
        print(f"Stored scene: {scene_id} at SLAM {slam_xyz}")

    def process_people_detection(
        self,
        person_id: str,
        people_xyz: tuple[float, float, float],
        face_embedding: Optional[List[float]] = None,
        pose_embedding: Optional[List[float]] = None,
        timeframe: Optional[str] = None,
        chat_history_ref: Optional[str] = None,
    ) -> None:
        """Store detected people"""
        person = PersonRecord(
            person_id=person_id,
            people_xyz=people_xyz,
            face_embedding=face_embedding,
            pose_embedding=pose_embedding,
            timeframe=timeframe,
            chat_history_ref=chat_history_ref,
        )
        self.people_db.upsert(person)
        print(f"Stored person: {person_id} at {people_xyz}")

    def find_objects(
        self,
        query_embedding: List[float],
        search_type: str = "image",
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find objects by image or location similarity"""
        if search_type == "image":
            return self.object_db.query_by_image_embedding(
                query_embedding, n_results=n_results
            )
        else:
            return self.object_db.query_by_location_embedding(
                query_embedding, n_results=n_results
            )

    def find_scenes(
        self,
        query_embedding: Optional[List[float]] = None,
        slam_coords: Optional[tuple[float, float, float]] = None,
        radius: float = 1.0,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find scenes by embedding similarity or SLAM coordinates"""
        if slam_coords is not None:
            return self.object_db.find_scenes_by_slam_coords(
                query_xyz=slam_coords, radius=radius, n_results=n_results
            )
        elif query_embedding is not None:
            return self.object_db.query_by_scene_embedding(
                query_embedding, n_results=n_results
            )
        else:
            raise ValueError("Must provide either query_embedding or slam_coords")

    def find_people(
        self,
        face_embedding: Optional[List[float]] = None,
        pose_embedding: Optional[List[float]] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find people by face or pose embedding"""
        if face_embedding is not None:
            return self.people_db.query_by_face_embedding(
                face_embedding, n_results=n_results
            )
        elif pose_embedding is not None:
            return self.people_db.query_by_pose_embedding(
                pose_embedding, n_results=n_results
            )
        else:
            raise ValueError("Must provide either face_embedding or pose_embedding")

    def get_person_by_face_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get person by FaceID"""
        return self.people_db.get_person_by_face_id(face_id)

    def get_object_coordinates(
        self, object_id: str
    ) -> Optional[tuple[float, float, float]]:
        """Get object coordinates for robot navigation"""
        obj = self.object_db.get_object(object_id)
        if obj:
            xyz = obj["object_xyz"]
            return (xyz[0], xyz[1], xyz[2])
        return None

    def get_scene_coordinates(
        self, scene_id: str
    ) -> Optional[tuple[float, float, float]]:
        """Get scene SLAM coordinates for robot navigation"""
        scene = self.object_db.get_scene(scene_id)
        if scene:
            xyz = scene["scene_xyz"]
            return (xyz[0], xyz[1], xyz[2])
        return None

    def get_person_coordinates(
        self, person_id: str
    ) -> Optional[tuple[float, float, float]]:
        """Get person coordinates for robot navigation"""
        person = self.people_db.get_person(person_id)
        if person:
            xyz = person["people_xyz"]
            return (xyz[0], xyz[1], xyz[2])
        return None


def example_agent_flow():
    """Simulates database usage in agent flow"""
    print("=" * 60)
    print("Agent & Robot Integration Example")
    print("=" * 60)
    print()

    agent = AgentIntegration()

    print("1. Vision detects objects")
    agent.process_object_detection(
        object_id="mug_01",
        object_xyz=(1.0, 2.0, 0.0),
        object_image_path="images/mug_01.png",
        object_embedding=np.random.randn(128).tolist(),
        location_embedding=np.random.randn(64).tolist(),
    )

    print("\n2. Vision detects scenes")
    agent.process_scene_detection(
        scene_id="kitchen_01",
        slam_xyz=(2.5, 3.1, 0.0),
        scene_image_path="images/kitchen_01.png",
        scene_embedding=np.random.randn(256).tolist(),
    )

    print("\n3. Vision detects people")
    agent.process_people_detection(
        person_id="person_01",
        people_xyz=(4.0, 5.0, 0.0),
        face_embedding=np.random.randn(512).tolist(),
        pose_embedding=np.random.randn(256).tolist(),
        timeframe="2024-01-01T10:30:00",
        chat_history_ref="chat/person_01.json",
    )

    print("\n4. Object&SceneFinding queries database")
    objects = agent.find_objects(
        query_embedding=np.random.randn(128).tolist(), search_type="image", n_results=3
    )
    print(f"   Found {len(objects)} objects")

    scenes = agent.find_scenes(slam_coords=(2.4, 3.0, 0.0), radius=1.0, n_results=3)
    print(f"   Found {len(scenes)} scenes")

    print("\n5. PeopleFinding queries database")
    people_by_face = agent.find_people(
        face_embedding=np.random.randn(512).tolist(), n_results=3
    )
    print(f"   Found {len(people_by_face)} people by face")

    people_by_pose = agent.find_people(
        pose_embedding=np.random.randn(256).tolist(), n_results=3
    )
    print(f"   Found {len(people_by_pose)} people by pose")

    person = agent.get_person_by_face_id("person_01")
    if person:
        print(f"   Found person by FaceID: {person['person_id']}")

    print("\n6. ControlRobot gets coordinates for navigation")
    coords = agent.get_object_coordinates("mug_01")
    if coords:
        print(f"   Object 'mug_01' is at: {coords}")
        print(f"   Robot can navigate to: Go to object ({coords[0]}, {coords[1]})")

    scene_coords = agent.get_scene_coordinates("kitchen_01")
    if scene_coords:
        print(f"   Scene 'kitchen_01' is at SLAM: {scene_coords}")

    person_coords = agent.get_person_coordinates("person_01")
    if person_coords:
        print(f"   Person 'person_01' is at: {person_coords}")

    print("\n" + "=" * 60)
    print("Integration example complete")
    print("=" * 60)


if __name__ == "__main__":
    example_agent_flow()
