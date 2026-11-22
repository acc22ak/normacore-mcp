"""Gather the camera views Claude looks at in goal mode.

Sources (each optional; whatever is live is used):
  gripper  Station camera with role 'gripper' (on the follower's wrist)
  desk     Station camera with role 'desk' (or the saved photo with --photo)
  rgbd     RealSense colour + aligned depth (colourised) from the depthcam daemon
           (~/normacore-mcp/depthcam.py serve, http://127.0.0.1:8791); depth_at() measures a pixel
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image

DEPTHCAM_URL = os.environ.get("DEPTHCAM_URL", "http://127.0.0.1:8791")
MAX_AGE_S = 2.0
LONG_SIDE = 640          # ~410 image tokens per 640x480 view


@dataclass
class View:
    name: str            # gripper | desk | rgbd | rgbd_depth
    jpeg: bytes          # downscaled, what Claude sees
    size: tuple[int, int]        # size Claude sees (pixel coords it uses)
    orig_size: tuple[int, int]   # camera's own size (for depth_at)
    label: str


def shrink(jpeg: bytes, long_side: int = LONG_SIDE, quality: int = 80) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    orig = img.size
    s = min(1.0, long_side / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return out.getvalue(), img.size, orig


def _http(path: str, timeout: float = 2.0) -> bytes:
    with urllib.request.urlopen(DEPTHCAM_URL + path, timeout=timeout) as r:
        return r.read()


class Cameras:
    def __init__(self, io_, roles: tuple[str, ...] = ("gripper", "desk"), use_rgbd: bool = True):
        self.io, self.roles, self.use_rgbd = io_, roles, use_rgbd
        self.rgbd_orig: tuple[int, int] | None = None
        self.rgbd_sent: tuple[int, int] | None = None

    async def grab(self) -> tuple[list[View], list[str]]:
        """All live views, plus notes on the ones that weren't available."""
        views, notes = [], []
        for role in self.roles:
            try:
                jpeg, age = await self.io.read_frame(role)
                if age > MAX_AGE_S:
                    notes.append(f"{role} camera frame is {age:.1f} s old: skipped")
                    continue
                small, size, orig = shrink(jpeg)
                where = {"gripper": "on the wrist, looking past the gripper fingers",
                         "desk": "fixed, looking down the table from the far end; the arm is at the top"}.get(role, "")
                views.append(View(role, small, size, orig, f"{role} camera ({where}), {size[0]}x{size[1]}"))
            except Exception as e:
                notes.append(f"{role} camera unavailable ({str(e)[:80]})")
        if self.use_rgbd:
            try:
                views += await asyncio.to_thread(self._rgbd)
            except Exception as e:
                notes.append(f"rgbd camera unavailable ({str(e)[:80]})")
        return views, notes

    def _rgbd(self) -> list[View]:
        st = json.loads(_http("/status"))
        age = st.get("age_s")
        if age is None or age > MAX_AGE_S:
            raise RuntimeError(f"frame is stale ({age})")
        color, size, orig = shrink(_http("/color.jpg"))
        self.rgbd_orig, self.rgbd_sent = orig, size
        views = [View("rgbd", color, size, orig, f"rgbd colour camera (fixed, beside the robot), {size[0]}x{size[1]}; "
                      "use 'measure' / 'distance' with pixel coordinates in THIS image")]
        try:   # height above the fitted table plane (depthcam_cloud); falls back to plain depth
            hmap, hsize, _ = shrink(_http("/height.jpg"))
            views.append(View("rgbd_height", hmap, hsize, orig,
                              "rgbd HEIGHT ABOVE THE TABLE, aligned pixel-for-pixel to the rgbd colour image: black = "
                              "table level, blue = 0.5-2 cm (paper pad, pens lying down), green ~5 cm, yellow ~7 cm, "
                              "red = 10 cm or more; dark grey = no depth reading"))
        except (urllib.error.URLError, OSError):
            depth, dsize, _ = shrink(_http("/depth.jpg"))
            views.append(View("rgbd_depth", depth, dsize, orig, "rgbd depth, colourised and aligned pixel-for-pixel to "
                              "the rgbd colour image (near = warm, far = cool, black = no reading)"))
        return views

    def _full(self, x: int, y: int) -> tuple[int, int]:
        sx = self.rgbd_orig[0] / self.rgbd_sent[0]
        sy = self.rgbd_orig[1] / self.rgbd_sent[1]
        return round(x * sx), round(y * sy)

    async def measure(self, x: int, y: int) -> str:
        """Physical height at a pixel of the rgbd colour image as Claude saw it."""
        if not self.rgbd_orig:
            return "no rgbd camera"
        X, Y = self._full(x, y)
        try:
            d = await asyncio.to_thread(lambda: json.loads(_http(f"/height_at?x={X}&y={Y}")))
        except urllib.error.HTTPError:      # older daemon without the table fit
            return await self._depth_only(x, y, X, Y)
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"measure failed ({e})"
        if d.get("error"):
            return await self._depth_only(x, y, X, Y)
        if d.get("height_m") is None:
            return f"({x}, {y}): no depth reading there (too close, shiny, dark or very thin)"
        cm = lambda v: f"{v * 100:.1f} cm"
        out = f"({x}, {y}): top of what is there is {cm(d['height_m'])} above the table"
        if d.get("above_surface_m") is not None:
            out += (f", {cm(d['above_surface_m'])} above the surface around it (that surface is "
                    f"{cm(d['surface_m'])} above the table)")
        return out + f"; {d['camera_dist_m']:.2f} m from the camera"

    async def _depth_only(self, x, y, X, Y) -> str:
        try:
            d = await asyncio.to_thread(lambda: json.loads(_http(f"/depth_at?x={X}&y={Y}&r=3")))
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"measure failed ({e})"
        if d.get("depth_m") is None:
            return f"no depth reading at ({x}, {y}) (too close, shiny or dark)"
        return f"({x}, {y}): {d['depth_m']:.3f} m from the camera (no table plane found, so no height)"

    async def distance(self, x1: int, y1: int, x2: int, y2: int) -> str:
        """Straight-line distance between the surfaces at two rgbd colour pixels."""
        if not self.rgbd_orig:
            return "no rgbd camera"
        X1, Y1 = self._full(x1, y1)
        X2, Y2 = self._full(x2, y2)
        try:
            d = await asyncio.to_thread(lambda: json.loads(_http(f"/distance?x1={X1}&y1={Y1}&x2={X2}&y2={Y2}")))
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"distance failed ({e})"
        if d.get("error"):
            return f"distance ({x1},{y1})-({x2},{y2}): {d['error']}"
        return f"distance ({x1},{y1})-({x2},{y2}): {d['distance_m'] * 100:.1f} cm"
