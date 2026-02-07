try:
    from walkie_sdk import WalkieRobot

    print("✓ Successfully imported WalkieRobot from local workspace!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
