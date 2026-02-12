import pytest
from walkie_db.agent_integration import AgentIntegration

def test_db_read_write():
    """ทดสอบการบันทึกตำแหน่งวัตถุลง Memory"""
    db = AgentIntegration(base_db_path="data/test_db")
    item_id = "test_mug_001"

    # Write
    db.process_object_detection(
        object_id=item_id,
        xyz=[1.0, 2.5, 0.5],
        embedding=[0.1]*512,
        label="blue_mug"
    )

    # Read
    coords = db.get_target_coords("object", item_id)
    assert coords == (1.0, 2.5, 0.5)
