"""RealSense D435 -> WebSocket bridge for station-viewer.

Reads color + depth (aligned to color) from a RealSense camera and broadcasts the
latest frame to every connected WebSocket client on 127.0.0.1:8890.

Run with sudo on macOS (librealsense needs root to detach the UVC driver):
    sudo .venv/bin/python bridge.py

Wire format, one binary message per frame:
    uint32 LE headerLen | UTF-8 JSON header | JPEG color bytes | depth bytes (uint16 LE, mm)
Plus text JSON status messages: {"type": "status", "state": ..., ...}
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import websockets
from PIL import Image

log = logging.getLogger("realsense-bridge")

COLOR_W, COLOR_H = 640, 480
# Camera pose + arm roll offsets from the station-viewer calibration (Level to table / Fit to arm
# shape). The viewer sends them here; the robot's MCP server reads this file to use the depth camera.
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"


def load_calibration() -> dict | None:
    try:
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_calibration(msg: dict) -> None:
    pose = msg.get("camera_pose") or {}
    if not all(isinstance(pose.get(k), (int, float)) for k in ("x", "y", "z", "yaw", "pitch", "roll")):
        log.warning("ignoring calibration without a complete camera_pose")
        return
    data = load_calibration() or {}
    data["camera_pose"] = {k: float(pose[k]) for k in ("x", "y", "z", "yaw", "pitch", "roll")}
    if msg.get("camera_serial"):
        data["camera_serial"] = str(msg["camera_serial"])
    if isinstance(msg.get("joint_offsets"), dict):
        data.setdefault("joint_offsets", {}).update(msg["joint_offsets"])
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = CALIBRATION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(CALIBRATION_FILE)
    log.info("calibration saved: %s", data["camera_pose"])


@dataclass
class Packet:
    seq: int
    data: bytes


class Camera:
    """Runs the RealSense pipeline in a background thread and keeps the latest packet."""

    def __init__(self, fps: int, max_depth_m: float, jpeg_quality: int, decimate: int):
        self.fps = fps
        self.max_depth_m = max_depth_m
        self.jpeg_quality = jpeg_quality
        self.decimate = decimate
        self.status: dict = {"type": "status", "state": "no_device", "message": "starting"}
        self.latest: Packet | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._on_change = None  # set by server: callable() thread-safe notifier

    # -- status ----------------------------------------------------------
    def _set_status(self, state: str, message: str = "", **extra) -> None:
        status = {"type": "status", "state": state, "message": message, **extra}
        if status != self.status:
            self.status = status
            log.info("status: %s %s", state, message)
            if self._on_change:
                self._on_change()

    # -- capture loop ----------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except Exception as exc:  # camera unplugged, permission error, ...
                msg = str(exc)
                hint = " (did you run with sudo?)" if "access" in msg.lower() else ""
                self._set_status("no_device" if "no device" in msg.lower() else "error", msg + hint)
            self._stop.wait(2.0)

    def _stream(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device connected")
        dev = devices[0]
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)
        name = dev.get_info(rs.camera_info.name)

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16, 30)
        pipeline = rs.pipeline(ctx)
        profile = pipeline.start(config)
        try:
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            d = self.decimate
            intrinsics = {
                # Intrinsics for the decimated depth grid (pixel centers of each d×d block).
                "fx": intr.fx / d,
                "fy": intr.fy / d,
                "ppx": (intr.ppx + 0.5) / d - 0.5,
                "ppy": (intr.ppy + 0.5) / d - 0.5,
            }
            self._set_status("streaming", "", serial=serial, firmware=firmware, name=name)
            period = 1.0 / self.fps
            next_send = 0.0
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(5000)
                now = time.monotonic()
                if now < next_send:
                    continue
                next_send = max(next_send + period, now)
                aligned = align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                self._publish(color, depth, depth_scale, intrinsics, serial)
        finally:
            pipeline.stop()

    def _publish(self, color, depth, depth_scale: float, intrinsics: dict, serial: str) -> None:
        rgb = np.asanyarray(color.get_data())
        z = np.asanyarray(depth.get_data()).astype(np.float32) * (depth_scale * 1000.0)  # -> mm
        d = self.decimate
        h, w = z.shape[0] // d, z.shape[1] // d
        # Min-pool over valid (non-zero) pixels: keeps edges sharp, ignores holes.
        blocks = z[: h * d, : w * d].reshape(h, d, w, d)
        blocks = np.where(blocks > 0, blocks, np.inf)
        z_small = blocks.min(axis=(1, 3))
        z_small[~np.isfinite(z_small)] = 0
        z_small[z_small > self.max_depth_m * 1000.0] = 0
        depth_bytes = z_small.astype("<u2").tobytes()

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=self.jpeg_quality)
        jpeg = buf.getvalue()

        self._seq += 1
        header = json.dumps(
            {
                "type": "frame",
                "seq": self._seq,
                "ts": time.time(),
                "color": {"w": rgb.shape[1], "h": rgb.shape[0], "bytes": len(jpeg)},
                "depth": {"w": w, "h": h, "bytes": len(depth_bytes), "unit_m": 0.001},
                "intrinsics": intrinsics,
                "serial": serial,
            }
        ).encode()
        data = struct.pack("<I", len(header)) + header + jpeg + depth_bytes
        with self._lock:
            self.latest = Packet(self._seq, data)
        if self._on_change:
            self._on_change()

    def stop(self) -> None:
        self._stop.set()


async def serve(cam: Camera, host: str, port: int) -> None:
    loop = asyncio.get_running_loop()
    changed = asyncio.Event()
    cam._on_change = lambda: loop.call_soon_threadsafe(changed.set)
    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        log.info("client connected (%d)", len(clients))
        try:
            await ws.send(json.dumps(cam.status))
            cal = load_calibration()
            if cal:
                await ws.send(json.dumps({"type": "calibration", **cal}))
            async for msg in ws:
                if isinstance(msg, str):
                    try:
                        data = json.loads(msg)
                    except ValueError:
                        continue
                    if data.get("type") == "calibration":
                        save_calibration(data)
        except websockets.ConnectionClosed:
            pass
        finally:
            clients.discard(ws)
            log.info("client disconnected (%d)", len(clients))

    async def pump():
        # Each client gets only the newest frame; a slow client skips frames
        # instead of queueing them (one in-flight send per client).
        last_status = None
        busy: set = set()
        while True:
            await changed.wait()
            changed.clear()
            status = cam.status
            if status is not last_status:
                last_status = status
                websockets.broadcast(clients, json.dumps(status))
            pkt = cam.latest
            if pkt is None or cam.status.get("state") != "streaming":
                continue
            for ws in list(clients):
                if ws in busy:
                    continue
                busy.add(ws)

                async def send(ws=ws, data=pkt.data):
                    try:
                        await ws.send(data)
                    except websockets.ConnectionClosed:
                        pass
                    finally:
                        busy.discard(ws)

                asyncio.create_task(send())

    async with websockets.serve(handler, host, port, max_size=None, compression=None):
        log.info("serving on ws://%s:%d", host, port)
        await pump()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="bind address (default localhost only)")
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--fps", type=int, default=15, help="frames sent per second")
    p.add_argument("--max-depth", type=float, default=1.5, help="drop depth beyond this (m)")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--decimate", type=int, default=2, help="depth downsampling factor")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cam = Camera(args.fps, args.max_depth, args.jpeg_quality, args.decimate)
    threading.Thread(target=cam.run, daemon=True).start()
    try:
        asyncio.run(serve(cam, args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
