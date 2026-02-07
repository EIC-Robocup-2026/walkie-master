#!/usr/bin/env python3
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ตรวจสอบ Dependency
try:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCSessionDescription
except ImportError:
    print("❌ Missing dependencies. Run: uv add aiortc aiohttp av")
    sys.exit(1)


class WebRTCCameraClient:
    """
    Advanced WebRTC Client สำหรับทดสอบการเชื่อมต่อกล้องหุ่นยนต์
    รองรับ Async Context Manager (async with) เพื่อการจัดการ Resource ที่สะอาด
    """

    def __init__(self, host: str, port: int = 8554):
        self.host = host
        self.port = port
        self.signaling_url = f"http://{host}:{port}/offer"
        self.pc: Optional[RTCPeerConnection] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self._stop_event = asyncio.Event()
        self._frame_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self) -> bool:
        """สร้างการเชื่อมต่อ WebRTC และทำ Signaling"""
        print(f"📡 Signaling: {self.signaling_url}")
        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self._frame_task = asyncio.create_task(self._receive_frames(track))

        @self.pc.on("connectionstatechange")
        async def on_state_change():
            print(f"🌐 Connection State: {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                self._stop_event.set()

        # WebRTC Offer
        self.pc.addTransceiver("video", direction="recvonly")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.signaling_url,
                    json={
                        "sdp": self.pc.localDescription.sdp,
                        "type": self.pc.localDescription.type,
                    },
                    timeout=5.0,
                ) as resp:
                    if resp.status != 200:
                        print(f"❌ Server Error: {resp.status}")
                        return False

                    answer = await resp.json()
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
                    )
            return True
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            return False

    async def _receive_frames(self, track):
        """Task สำหรับถอดรหัสภาพวิดีโอ"""
        while not self._stop_event.is_set():
            try:
                frame = await track.recv()
                # แปลงจาก AVFrame เป็น RGB numpy array
                self.current_frame = frame.to_ndarray(format="bgr24")
                self.frame_count += 1
            except Exception:
                break

    async def disconnect(self):
        """ล้าง Resource และหยุดการทำงาน"""
        self._stop_event.set()
        if self._frame_task:
            self._frame_task.cancel()
        if self.pc:
            await self.pc.close()
        print("🔌 Disconnected and cleaned up.")


# --- UI / Runner Logic ---


async def main_loop(args):
    """ฟังก์ชันหลักสำหรับรันการทดสอบภาพ"""
    async with WebRTCCameraClient(args.host, args.port) as camera:
        print("⌛ Waiting for stream...")

        # รอจนกว่าเฟรมแรกจะมา
        timeout = 10
        start_wait = time.time()
        while camera.current_frame is None:
            if time.time() - start_wait > timeout:
                print("❌ Timeout: No video stream received.")
                return
            await asyncio.sleep(0.1)

        print(f"✅ Stream Connected: {camera.current_frame.shape}")

        if args.headless:
            print(f"🚀 Running in Headless mode. Received {camera.frame_count} frames.")
            await asyncio.sleep(5)  # รันเทส 5 วินาทีแล้วจบ
            return

        # GUI Mode
        window_name = f"WebRTC: {args.host}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while not camera._stop_event.is_set():
                frame = camera.current_frame
                if frame is not None:
                    # ใส่ Overlay เล็กน้อย
                    cv2.putText(
                        frame,
                        f"Frames: {camera.frame_count} | host: {args.host}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                await asyncio.sleep(0.01)  # ปล่อยให้ Event Loop ทำงาน
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1")
    parser.add_argument("port", nargs="?", type=int, default=8554)
    parser.add_argument("--headless", action="store_true", help="รันโดยไม่เปิดหน้าต่าง GUI")
    args = parser.parse_args()

    try:
        asyncio.run(main_loop(args))
    except KeyboardInterrupt:
        pass
