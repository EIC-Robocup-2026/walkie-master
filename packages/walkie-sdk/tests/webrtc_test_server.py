#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import time
from typing import Optional, Set

import cv2
import numpy as np
from aiohttp import web

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame
except ImportError:
    print("❌ Error: Missing dependencies. Run: uv add aiortc aiohttp av opencv-python")
    exit(1)

# ตั้งค่า Logging ให้เห็นสถานะการเชื่อมต่อชัดเจน
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebRTC-Server")

# ใช้ Relay เพื่อให้ดึงภาพจากกล้องตัวเดียวไปส่งให้หลาย Client พร้อมกันได้
relay = MediaRelay()


class WebcamVideoTrack(VideoStreamTrack):
    """ดึงภาพจากกล้อง OpenCV และส่งผ่าน WebRTC"""

    kind = "video"

    def __init__(self, device: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            logger.error(f"Cannot open device {device}")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()

        if not ret:  # ถ้าดึงภาพไม่ได้ ให้ส่งเฟรมสีเทาแทน
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        # แปลงเป็น RGB สำหรับ PyAV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release()


class TestPatternTrack(VideoStreamTrack):
    """สร้างภาพทดสอบแบบเคลื่อนไหว (ไม่ต้องใช้กล้องจริง)"""

    kind = "video"

    def __init__(self):
        super().__init__()
        self.counter = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        # สร้างภาพสีดำที่มีวงกลมวิ่งไปมา
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"TEST PATTERN - FRAME {self.counter}",
            (150, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # วงกลมเคลื่อนที่
        center_x = int(320 + 200 * np.sin(self.counter / 10))
        cv2.circle(frame, (center_x, 300), 40, (0, 255, 0), -1)

        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        self.counter += 1
        return video_frame


class WebRTCServer:
    def __init__(self, source: str, port: int):
        self.source = source
        self.port = port
        self.pcs: Set[RTCPeerConnection] = set()
        self.master_track = self._setup_source()

    def _setup_source(self):
        if self.source == "test":
            return TestPatternTrack()
        return WebcamVideoTrack(device=int(self.source) if self.source.isdigit() else 0)

    async def handle_offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            logger.info(f"Connection State: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)

        # ใช้ relay.subscribe เพื่อแชร์ track เดียวให้หลายคน
        pc.addTrack(relay.subscribe(self.master_track))

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def handle_index(self, request):
        return web.Response(
            text="<h1>Walkie WebRTC Test Server</h1><p>Status: Running</p>",
            content_type="text/html",
        )

    async def cleanup(self, app):
        logger.info("Closing all peer connections...")
        await asyncio.gather(*[pc.close() for pc in self.pcs])
        self.master_track.stop()

    def run(self):
        app = web.Application()
        app.router.add_post("/offer", self.handle_offer)
        app.router.add_get("/", self.handle_index)
        app.on_shutdown.append(self.cleanup)

        print(f"🚀 WebRTC Server started at http://localhost:{self.port}")
        print(f"📹 Source: {self.source}")
        web.run_app(app, port=self.port, access_log=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument(
        "--source", default="webcam", help="webcam or test or device_index"
    )
    args = parser.parse_args()

    server = WebRTCServer(args.source, args.port)
    server.run()
