import time

import pytest
from walkie_sdk.robot import WalkieRobot


# 1. สร้าง Fixture สำหรับจัดการ Robot Connection
@pytest.fixture(scope="module")
def robot():
    """เชื่อมต่อหุ่นยนต์เพียงครั้งเดียวต่อการรัน test module นี้"""
    # ในการเทสจริง คุณอาจดึง IP จาก Environment Variable ได้
    # เช่น os.getenv("ROBOT_IP", "127.0.0.1")
    r = WalkieRobot(ros_protocol="rosbridge", ip="127.0.0.1", ros_port=9090)

    # รอสักครู่ให้การเชื่อมต่อเสถียร (หรือใช้ logic เช็ค r.is_connected())
    time.sleep(1)

    yield r  # ส่ง robot ไปให้ test functions ใช้งาน

    # หลังจบทุกการเทส ให้ตัดการเชื่อมต่ออัตโนมัติ (Teardown)
    r.disconnect()
    print("\n[Teardown] Robot disconnected.")


# 2. Test Case: สั่งพับแขน (Go Home)
def test_arm_go_to_home(robot):
    print("\nTesting 'Go To Home' command...")
    # สมมติว่า SDK คืนค่า True/False หรือ Result Object
    result = robot.arm.go_to_home(group_name="left_arm")

    # Assert ว่าคำสั่งต้องถูกส่งสำเร็จ
    assert result is not None, "Failed to send Go To Home command"


# 3. Test Case: สั่งเปิด/ปิด Gripper
def test_control_gripper(robot):
    print("\nTesting 'Control Gripper' command...")
    result = robot.arm.control_gripper(group_name="left_gripper", position=0.7)

    assert result is not None, "Failed to send Control Gripper command"


# 4. Test Case (Optional): สั่งเคลื่อนที่แบบสัมพัทธ์ (Relative Movement)
def test_arm_relative_movement(robot):
    """ทดสอบการขยับแขนเล็กน้อย"""
    # ใช้การขยับเพียงครั้งเดียวเพื่อความปลอดภัยในการเทส
    result = robot.arm.go_to_pose_relative(
        group_name="left_arm",
        x=0.01,
        z=-0.01,
        blocking=True,  # ในการเทสแนะนำให้ใช้ True เพื่อรอเช็คผล
    )
    assert result is not None


# 5. Test Case: สั่งไปที่พิกัด Pose (ถ้าต้องการเทสความแม่นยำ)
@pytest.mark.skip(reason="Safety: avoid hitting environment during auto-test")
def test_arm_go_to_pose(robot):
    result = robot.arm.go_to_pose(
        x=0.38, y=0.19, z=0.58, roll=-1.57, pitch=0.0, yaw=1.57, group_name="left_arm"
    )
    assert result is not None
