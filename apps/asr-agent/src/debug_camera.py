import os
import time

import cv2
import numpy as np

from asr_agent.tools.robot import WalkieRobot


def debug_camera_zenoh():
    output_file = "camera_zenoh_result.jpg"
    if os.path.exists(output_file):
        os.remove(output_file)

    print("🚀 Walkie SDK Camera Debugger: [ZENOH MODE]")
    print("📡 Target: 127.0.0.1 | Port: 7447")

    try:
        # 1. เชื่อมต่อโดยระบุโปรโตคอล Zenoh ทั้งระบบควบคุมและกล้อง
        bot = WalkieRobot(
            ip="127.0.0.1",
            ros_protocol="zenoh",  # ใช้ Zenoh สำหรับ Telemetry/Nav
            ros_port=7447,
            camera_protocol="zenoh",  # ใช้ Zenoh สำหรับ Video Stream
            camera_port=7447,
        )

        # 2. รอการค้นหาคู่สถานี (Zenoh Discovery)
        # Zenoh มักจะใช้เวลาหา Peer สักครู่ในครั้งแรก
        print("⏳ Waiting for Zenoh discovery and stream...")
        timeout = 10
        start_time = time.time()

        while time.time() - start_time < timeout:
            if bot.camera and bot.camera.is_streaming:
                print("✅ Zenoh Camera stream is active!")
                break
            time.sleep(1)
        else:
            print("⚠️ Timeout: Camera is not streaming.")
            print("💡 Tip: Check if 'zenoh-bridge-ros2dds' is running in your Sim.")

        # 3. ตรวจสอบการรับเฟรม
        if bot.camera:
            print("📸 Attempting to grab a frame via Zenoh...")
            frame = bot.camera.get_frame()

            if frame is not None and isinstance(frame, np.ndarray):
                h, w = frame.shape[:2]
                print(f"✅ Success! Received frame: {w}x{h}")

                cv2.imwrite(output_file, frame)
                print(f"📂 Saved to: {os.path.abspath(output_file)}")

                # แสดงผลสด
                try:
                    cv2.imshow("Zenoh Live View", frame)
                    print("📺 Displaying... Press any key to close.")
                    cv2.waitKey(0)
                except Exception:
                    print("ℹ️ Headless mode active.")
            else:
                print(
                    "❌ Failed: get_frame() returned None. Zenoh might be connected but no data published."
                )
        else:
            print("❌ Error: Camera interface not initialized.")

    except Exception as e:
        print(f"💥 Exception: {str(e)}")
    finally:
        print("🔌 Disconnecting Zenoh...")
        if "bot" in locals():
            bot.disconnect()


if __name__ == "__main__":
    debug_camera_zenoh()
