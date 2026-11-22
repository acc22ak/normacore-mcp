"""RealSense depth for the MCP server: frames from realsense-bridge, the calibrated camera pose, and
object detection on the table, all in the station-viewer 3D frame (metres, robot base at origin, Y up).

Pure functions plus one small async client; no MCP / Station imports so it can be unit-tested.
The camera pose and the arm's roll offsets come from the calibration file the bridge writes when
the station-viewer page calibrates (Level to table / Fit to arm shape).
"""

from __future__ import annotations

import asyncio
import colorsys
import io
import json
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ------------------------------------------------------------------ frames
@dataclass
class Frame:
    header: dict
    jpeg: bytes
    depth_mm: np.ndarray      # (h, w) uint16, aligned to the color image, 0 = no data

    @property
    def intrinsics(self) -> dict:
        return self.header["intrinsics"]


def parse_frame(msg: bytes) -> Frame:
    if len(msg) < 4:
        raise ValueError("message too short")
    hlen = struct.unpack_from("<I", msg, 0)[0]
    header = json.loads(msg[4:4 + hlen])
    if header.get("type") != "frame":
        raise ValueError(f"unexpected message type {header.get('type')!r}")
    c, d = header["color"], header["depth"]
    j0 = 4 + hlen
    d0 = j0 + c["bytes"]
    d1 = d0 + d["bytes"]
    if d1 > len(msg) or d["bytes"] != d["w"] * d["h"] * 2:
        raise ValueError("truncated or inconsistent frame")
    depth = np.frombuffer(msg[d0:d1], dtype="<u2").reshape(d["h"], d["w"]).copy()
    return Frame(header=header, jpeg=bytes(msg[j0:d0]), depth_mm=depth)


def encode_frame(frame: Frame) -> bytes:
    """A frame in the bridge's wire format (parse_frame reads it back): for saving frames to disk."""
    header = json.dumps(frame.header).encode()
    return struct.pack("<I", len(header)) + header + frame.jpeg + frame.depth_mm.astype("<u2").tobytes()


def merge_frames(frames: list[Frame]) -> Frame:
    """One frame from several: the newest color image, and per pixel the median depth of the frames that
    have depth there (pixels with depth in fewer than half the frames stay empty). One frame's depth noise
    on the table is ~2.2 mm (measured 2026-11-22); the median of 5 is about half that."""
    if len(frames) == 1:
        return frames[0]
    stack = np.stack([f.depth_mm for f in frames]).astype(float)
    valid = stack > 0
    stack[~valid] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    med[valid.sum(axis=0) * 2 < len(frames)] = 0
    last = frames[-1]
    return Frame(header=last.header, jpeg=last.jpeg, depth_mm=np.nan_to_num(med).round().astype(np.uint16))


async def fetch_frame(url: str, timeout_s: float = 3.0, frames: int = 1) -> Frame:
    """Connect to realsense-bridge, return the next frame (or the merge of the next `frames`), disconnect."""
    import websockets  # local import: optional dependency for tests that don't need it

    async def _get() -> Frame:
        got: list[Frame] = []
        async with websockets.connect(url, max_size=None, open_timeout=timeout_s) as ws:
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    got.append(parse_frame(bytes(msg)))
                    if len(got) >= max(1, frames):
                        return merge_frames(got)
                    continue
                status = json.loads(msg)
                if status.get("type") == "status" and status.get("state") != "streaming":
                    raise RuntimeError(f"RealSense bridge: {status.get('state')} {status.get('message', '')}".strip())

    return await asyncio.wait_for(_get(), timeout_s + 0.2 * max(0, frames - 1))


# ---------------------------------------------------------------- geometry
def pose_matrix(pose: dict) -> np.ndarray:
    """Camera rig -> world. three.js Euler 'YXZ': R = Ry(yaw) · Rx(pitch) · Rz(roll)."""
    cy, sy = np.cos(pose["yaw"]), np.sin(pose["yaw"])
    cx, sx = np.cos(pose["pitch"]), np.sin(pose["pitch"])
    cz, sz = np.cos(pose["roll"]), np.sin(pose["roll"])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Ry @ Rx @ Rz
    T[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    return T


def world_points(frame: Frame, pose: dict) -> np.ndarray:
    """(h, w, 3) world coordinates per depth pixel; NaN where there's no depth."""
    h, w = frame.depth_mm.shape
    intr = frame.intrinsics
    z = frame.depth_mm.astype(float) * frame.header["depth"].get("unit_m", 0.001)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    # RealSense optical (x right, y down, z forward) -> rig (x right, y up, -z forward).
    local = np.stack([(u - intr["ppx"]) * z / intr["fx"], -(v - intr["ppy"]) * z / intr["fy"], -z], axis=-1)
    T = pose_matrix(pose)
    pts = local @ T[:3, :3].T + T[:3, 3]
    pts[z == 0] = np.nan
    return pts


def pixel_to_world(frame: Frame, pose: dict, fx: float, fy: float, radius: int = 2) -> np.ndarray | None:
    """World point under an image position given as fractions (0..1) of width/height (median depth)."""
    h, w = frame.depth_mm.shape
    u, v = int(round(fx * (w - 1))), int(round(fy * (h - 1)))
    win = frame.depth_mm[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
    valid = win[win > 0]
    if valid.size == 0:
        return None
    z = float(np.median(valid)) * frame.header["depth"].get("unit_m", 0.001)
    intr = frame.intrinsics
    local = np.array([(u - intr["ppx"]) * z / intr["fx"], -(v - intr["ppy"]) * z / intr["fy"], -z, 1.0])
    return (pose_matrix(pose) @ local)[:3]


# ----------------------------------------------------------------- objects
@dataclass
class DetectConfig:
    min_height_m: float = 0.005        # point height above the fitted table plane
    min_top_m: float = 0.009           # an object must stand at least this tall
    max_height_m: float = 0.30
    max_reach_m: float = 0.60          # from the robot base, horizontally
    min_x_m: float = 0.10              # in front of the base (the arm faces +x); skips the table edge
    cell_m: float = 0.007              # footprint grid for clustering (4-connected)
    min_points: int = 15
    arm_margin_m: float = 0.05         # ignore depth points this close to the 3D arm (model can be 1-2 cm off)
    touch_table_m: float = 0.04        # an object's lowest point must be this close to the table


def color_name(rgb) -> str:
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.22:
        return "black"
    if s < 0.22:
        return "white" if v > 0.7 else "gray"
    deg = h * 360
    for limit, name in ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"),
                        (255, "blue"), (290, "purple"), (340, "pink"), (361, "red")):
        if deg < limit:
            return name
    return "red"


def _voxel_keys(pts: np.ndarray, cell: float) -> set[tuple[int, int, int]]:
    keys = np.floor(pts / cell).astype(int)
    out = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.update(map(tuple, keys + [dx, dy, dz]))
    return out


def fit_table(pts: np.ndarray, reach_m: float = 0.6) -> np.ndarray:
    """Least-squares plane y = a*x + b*z + c through the table (points near y = 0), 2 rounds with
    outlier rejection. Returns (a, b, c). The calibration puts the table at y ~ 0, but tables sag
    and calibrations drift by a few mm, which is the height of a lying object."""
    p = pts.reshape(-1, 3)
    p = p[~np.isnan(p[:, 1]) & (np.abs(p[:, 1]) < 0.02) & (np.hypot(p[:, 0], p[:, 2]) < reach_m)]
    if len(p) < 100:
        return np.zeros(3)
    if len(p) > 20000:
        p = p[:: len(p) // 20000]
    coef = np.zeros(3)
    for _ in range(3):
        A = np.c_[p[:, 0], p[:, 2], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 1], rcond=None)
        res = p[:, 1] - A @ coef
        keep = np.abs(res) < 0.004
        if keep.sum() < 100:
            break
        p = p[keep]
    return coef


def find_objects(frame: Frame, pose: dict, arm_points: np.ndarray | None = None,
                 cfg: DetectConfig = DetectConfig()) -> list[dict]:
    """Things standing on the table (not the table, not the arm), largest first.

    Each object: id, color, center [x, y, z] (y = half its height), top_m, size_cm [length, width,
    height], axis_deg (long-axis direction in the x-z plane), ends (for elongated objects: the two
    ends of the long axis, each with brightness), distance_m from the base, bbox_px in the color image.
    """
    pts = world_points(frame, pose)
    h, w = frame.depth_mm.shape
    rgb = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB").resize((w, h)), dtype=float)

    a, b, c0 = fit_table(pts, cfg.max_reach_m)
    y = pts[..., 1] - (a * pts[..., 0] + b * pts[..., 2] + c0)     # height above the table
    reach = np.hypot(pts[..., 0], pts[..., 2])
    mask = (~np.isnan(y)) & (y > cfg.min_height_m) & (y < cfg.max_height_m) \
        & (reach < cfg.max_reach_m) & (pts[..., 0] > cfg.min_x_m)
    if arm_points is not None and len(arm_points):
        cell = cfg.arm_margin_m
        arm = _voxel_keys(arm_points, cell)
        idx = np.argwhere(mask)
        keys = np.floor(pts[mask] / cell).astype(int)
        near_arm = np.fromiter((tuple(k) in arm for k in keys), dtype=bool, count=len(keys))
        mask[idx[near_arm, 0], idx[near_arm, 1]] = False

    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return []
    P = pts[vs, us].copy()
    P[:, 1] = y[vs, us]                                           # heights above the table from here on
    C = rgb[vs, us]
    cells = np.floor(P[:, [0, 2]] / cfg.cell_m).astype(int)
    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        by_cell.setdefault(c, []).append(i)

    seen: set[tuple[int, int]] = set()
    objects = []
    for start in by_cell:
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            c = queue.popleft()
            comp.extend(by_cell[c])
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dz)
                if n in by_cell and n not in seen:
                    seen.add(n)
                    queue.append(n)
        if len(comp) < cfg.min_points:
            continue
        sel = np.array(comp)
        p = P[sel]
        if np.percentile(p[:, 1], 5) > cfg.touch_table_m:
            continue                      # floating: an arm part or a hand, not something on the table
        top = float(np.percentile(p[:, 1], 95))
        if top < cfg.min_top_m:
            continue
        xz = p[:, [0, 2]]
        # Centre from the object's upper half only. Depth pixels that mix the object with the table behind
        # it trail away from the camera at low heights and stretch the footprint: a standing minifigure
        # came out 7.8 cm long and its box centre 2.5 cm behind the figure (2026-11-22).
        body = xz[p[:, 1] >= 0.5 * top] if top > 0.015 else xz
        if len(body) < 5:
            body = xz
        center_xz = (body.min(axis=0) + body.max(axis=0)) / 2
        cov = np.cov((xz - xz.mean(axis=0)).T) if len(xz) > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        proj = (xz - xz.mean(axis=0)) @ axis
        perp = (xz - xz.mean(axis=0)) @ np.array([-axis[1], axis[0]])
        length = float(proj.max() - proj.min())
        width = float(perp.max() - perp.min())
        mean_rgb = C[sel].mean(axis=0)
        obj = {
            "color": color_name(mean_rgb),
            "center": [round(float(center_xz[0]), 4), round(top / 2, 4), round(float(center_xz[1]), 4)],
            "top_m": round(top, 4),
            "size_cm": [round(length * 100, 1), round(width * 100, 1), round(top * 100, 1)],
            "axis_deg": round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1),
            "distance_m": round(float(np.hypot(*center_xz)), 3),
            "points": int(len(sel)),
            "bbox_px": _bbox_px(us[sel], vs[sel], frame),
        }
        if length > 2.5 * max(width, 0.005):
            ends = []
            for end_proj in (proj.min(), proj.max()):
                k = np.abs(proj - end_proj) < 0.02          # points within 2 cm of this end
                e = xz.mean(axis=0) + axis * end_proj
                ends.append({"xz": [round(float(e[0]), 4), round(float(e[1]), 4)],
                             "brightness": round(float(C[sel][k].mean() / 255), 2) if k.any() else None})
            obj["ends"] = ends
        objects.append(obj)

    objects.sort(key=lambda o: -o["points"])
    for i, o in enumerate(objects, 1):
        o["id"] = i
    return objects


def _bbox_px(us: np.ndarray, vs: np.ndarray, frame: Frame) -> list[int]:
    sx = frame.header["color"]["w"] / frame.depth_mm.shape[1]
    sy = frame.header["color"]["h"] / frame.depth_mm.shape[0]
    return [int(us.min() * sx), int(vs.min() * sy), int((us.max() + 1) * sx), int((vs.max() + 1) * sy)]


def annotate(frame: Frame, objects: list[dict]) -> bytes:
    """Color image with numbered boxes around the detected objects, as JPEG."""
    img = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for o in objects:
        x0, y0, x1, y1 = o["bbox_px"]
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(240, 168, 48), width=2)
        draw.rectangle([x0 - 2, y0 - 16, x0 + 14, y0 - 2], fill=(240, 168, 48))
        draw.text((x0 + 2, y0 - 15), str(o["id"]), fill=(0, 0, 0))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


# ------------------------------------------------------------- calibration
def load_calibration(path: Path) -> dict:
    """{"camera_pose": {...}, "joint_offsets": {bus serial: {joint: rad}}} written by the bridge."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pose = data.get("camera_pose")
    if not pose or not all(k in pose for k in ("x", "y", "z", "yaw", "pitch", "roll")):
        raise ValueError(f"{path} has no camera_pose; calibrate in the station-viewer page first")
    return data


# ----------------------------------------------------------------- markers
def color_intrinsics(frame: Frame) -> dict:
    """Full-resolution color intrinsics, from the header's (depth-grid) intrinsics."""
    d = frame.header["color"]["w"] / frame.depth_mm.shape[1]
    i = frame.intrinsics
    return {"fx": i["fx"] * d, "fy": i["fy"] * d, "ppx": (i["ppx"] + 0.5) * d - 0.5, "ppy": (i["ppy"] + 0.5) * d - 0.5}


OPTICAL_TO_RIG = np.diag([1.0, -1.0, -1.0])   # RealSense optical (x right, y down, z fwd) -> rig (y up, -z fwd)


def detect_markers(frame: Frame, pose: dict, marker_size_m: float, dictionary: str = "DICT_4X4_50") -> list[dict]:
    """ArUco markers in the color image, in the robot frame.

    position: marker centre (m) from the depth image where it has depth (flat paper: good depth),
    else from the marker's corners (solvePnP). normal: unit vector out of the printed face, x_axis: along
    the marker's top edge. pnp_depth_gap_mm says how far the two estimates disagree (sanity check).
    """
    import cv2

    img = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return []
    ci = color_intrinsics(frame)
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]])
    s = marker_size_m / 2
    objp = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)   # TL, TR, BR, BL
    T = pose_matrix(pose)
    scale = frame.depth_mm.shape[1] / frame.header["color"]["w"]
    out = []
    for c, mid in zip(corners, ids.flatten()):
        px = c[0].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(objp, px, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R_cv, _ = cv2.Rodrigues(rvec)                     # marker -> optical
        p_pnp = tvec.flatten()
        # Depth: median of the depth pixels inside the marker's quad (shrunk a bit to avoid edges).
        centre = px.mean(axis=0)
        quad = (centre + (px - centre) * 0.6) * scale
        u0, v0 = np.floor(quad.min(axis=0)).astype(int)
        u1, v1 = np.ceil(quad.max(axis=0)).astype(int)
        win = frame.depth_mm[max(0, v0):v1 + 1, max(0, u0):u1 + 1]
        valid = win[win > 0]
        if valid.size >= 3:
            z = float(np.median(valid)) * frame.header["depth"].get("unit_m", 0.001)
            p_opt = np.array([(centre[0] - ci["ppx"]) * z / ci["fx"], (centre[1] - ci["ppy"]) * z / ci["fy"], z])
            source = "depth"
        else:
            p_opt, source = p_pnp, "pnp"
        R_world = T[:3, :3] @ OPTICAL_TO_RIG @ R_cv
        out.append({
            "id": int(mid),
            "position": (T[:3, :3] @ (OPTICAL_TO_RIG @ p_opt) + T[:3, 3]).round(4).tolist(),
            "normal": (R_world @ [0, 0, 1]).round(3).tolist(),
            "x_axis": (R_world @ [1, 0, 0]).round(3).tolist(),
            "source": source,
            "pnp_depth_gap_mm": round(float(np.linalg.norm(p_pnp - p_opt)) * 1000, 1),
            "size_px": round(float(np.linalg.norm(px[0] - px[1])), 1),
            "center_px": centre.round(1).tolist(),
        })
    return out


# ---------------------------------------------------------------- jaw tips
def find_jaw_tips(frame: Frame, pose: dict, predicted_jaw_end, *, dark_max: int = 75, window_px: int = 70,
                  table: np.ndarray | None = None, plane_y: float | None = None) -> dict | None:
    """The gripper's two jaw tips as the depth camera really sees them (world, m), or None.

    Why: the arm model can put the point between the jaws 2 cm from where it really is (measured
    2026-11-22: the model's centre sat on one real jaw tip in every frame). Objects come from this same
    camera, so aiming the *seen* jaw midpoint at the *seen* object cancels arm-model and camera errors.

    How: the jaws are black spikes pointing down at a lighter table. Around the pixel where the model
    expects the jaw ends, take the dark pixels, find the lowest point of each dark column run, and keep
    the two lowest separate peaks. Each tip's distance comes from the depth pixels on the jaw just above
    it (the thin tip itself rarely has depth); without depth there, from the table plane under it.
    Needs the jaws pointing down (>= ~45 deg below horizontal) and a table lighter than the jaws.
    plane_y: skip the depth lookup and place both tips where their pixel rays meet the horizontal plane
    at this height. The result is then exact across the camera's view and off by (height error / tan of
    the camera's look-down angle) along it, which calibrate_reach exploits by looking from the side.
    Returns {"tips": [[x,y,z],[x,y,z]], "midpoint": [x,y,z], "gap_m", "source": "depth"|"plane"}.
    """
    img = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB")).astype(int)
    H, W = img.shape[:2]
    ci = color_intrinsics(frame)
    Tc = pose_matrix(pose)
    Tinv = np.linalg.inv(Tc)
    l = Tinv[:3, :3] @ np.asarray(predicted_jaw_end, dtype=float) + Tinv[:3, 3]
    if l[2] >= -0.05:
        return None
    u0 = int(round(ci["fx"] * l[0] / -l[2] + ci["ppx"]))
    v0 = int(round(-ci["fy"] * l[1] / -l[2] + ci["ppy"]))
    ua, ub = max(0, u0 - window_px), min(W, u0 + window_px)
    va, vb = max(0, v0 - window_px), min(H, v0 + int(window_px * 0.65))
    if ub - ua < 20 or vb - va < 20:
        return None
    # "Dark" is relative to the table around the jaws: in direct sunlight the black jaws read brighter
    # than the fixed limit and went undetected (2026-11-22).
    window = img[va:vb, ua:ub].max(axis=2)
    dark = np.zeros((H, W), dtype=bool)
    dark[va:vb, ua:ub] = window < max(dark_max, 0.5 * float(np.median(window)))
    cols = []
    for u in range(ua, ub):
        rows = np.nonzero(dark[:, u])[0]
        if len(rows) < 12:
            continue
        v = int(rows.max())
        if v < vb - 1 and dark[max(0, v - 14):v + 1, u].sum() >= 12:   # a solid run reaching down to v
            cols.append((u, v))
    if len(cols) < 4:
        return None
    cols = np.array(cols)
    low = cols[cols[:, 1] > cols[:, 1].max() - 25]
    groups = [g for g in np.split(low, np.nonzero(np.diff(low[:, 0]) > 4)[0] + 1) if len(g) >= 2]
    peaks = sorted((g[np.argmax(g[:, 1])] for g in groups), key=lambda p: -p[1])[:2]
    if len(peaks) < 2:
        return None

    unit = frame.header["depth"].get("unit_m", 0.001)
    s = frame.depth_mm.shape[1] / W
    tips, sources = [], []
    for u, v in peaks:
        # Depth on the jaw just above the tip: dark pixels in a narrow strip, in the depth grid.
        vs, us = np.nonzero(dark[max(0, v - 30):max(0, v - 3), max(0, u - 4):u + 5])
        z = None
        if len(vs):
            dv = np.clip(((vs + max(0, v - 30)) * s).astype(int), 0, frame.depth_mm.shape[0] - 1)
            du = np.clip(((us + max(0, u - 4)) * s).astype(int), 0, frame.depth_mm.shape[1] - 1)
            d = frame.depth_mm[dv, du]
            d = d[d > 0]
            if len(d) >= 3:
                z = float(np.percentile(d, 25)) * unit     # nearer quartile: the rest is table seen past the jaw
        ray = np.array([(u - ci["ppx"]) / ci["fx"], -(v - ci["ppy"]) / ci["fy"], -1.0])
        if plane_y is not None:
            dw = Tc[:3, :3] @ ray
            if abs(dw[1]) < 1e-6:
                return None
            tips.append(Tc[:3, 3] + dw * ((plane_y - Tc[1, 3]) / dw[1]))
            sources.append("plane")
            continue
        if z is not None:
            p = Tc[:3, :3] @ (ray * z) + Tc[:3, 3]
            a, b, c = table if table is not None else (0.0, 0.0, 0.0)
            if p[1] - (a * p[0] + b * p[2] + c) > -0.01:   # a jaw tip can't be under the table
                tips.append(p)
                sources.append("depth")
                continue
        # Fallback: where the pixel's ray meets the horizontal plane at the model's jaw-end height.
        dw = Tc[:3, :3] @ ray
        if abs(dw[1]) < 1e-6:
            return None
        tips.append(Tc[:3, 3] + dw * ((float(predicted_jaw_end[1]) - Tc[1, 3]) / dw[1]))
        sources.append("plane")
    gap = float(np.linalg.norm(tips[0] - tips[1]))
    if not 0.008 < gap < 0.075:          # the jaws open to ~5 cm: anything else isn't the two tips
        return None
    return {"tips": [t.tolist() for t in tips], "midpoint": ((tips[0] + tips[1]) / 2).tolist(),
            "gap_m": gap, "source": "depth" if all(x == "depth" for x in sources) else "plane"}


# ------------------------------------------------- objects from the color image
@dataclass
class RgbDetectConfig:
    background_px: int = 61            # median window that models the bare table (bigger than any object)
    contrast: int = 34                 # color difference from that background that counts as "not table"
    min_area_px: int = 45
    max_area_px: int = 9000
    max_reach_m: float = 0.46          # the arm reaches ~0.42 m; further out is table edge, tape and cables
    min_x_m: float = 0.04              # skip the strip right at the robot's base
    arm_margin_px: int = 12
    crowd_gap_m: float = 0.035         # another object closer than this is in the jaws' way


def _table_point(frame: Frame, Tc: np.ndarray, plane, u: float, v: float) -> np.ndarray | None:
    """Where the ray through color pixel (u, v) meets the table plane y = a*x + b*z + c (world, m)."""
    ci = color_intrinsics(frame)
    a, b, c = plane
    d = Tc[:3, :3] @ np.array([(u - ci["ppx"]) / ci["fx"], -(v - ci["ppy"]) / ci["fy"], -1.0])
    o = Tc[:3, 3]
    denom = d[1] - a * d[0] - b * d[2]
    if abs(denom) < 1e-9:
        return None
    s = (a * o[0] + b * o[2] + c - o[1]) / denom
    return o + s * d if s > 0 else None


def find_objects_rgb(frame: Frame, pose: dict, arm_points: np.ndarray | None = None,
                     cfg: RgbDetectConfig = RgbDetectConfig()) -> list[dict]:
    """Objects on the table, found in the COLOR image and placed with the table plane.

    Why not depth alone (the old find_objects): depth is ~2 mm noisy per frame on a 320x240 grid, so it
    missed a lying minifigure and the pen, reported sun patches as objects, and its mixed pixels behind an
    object stretched the box by centimetres (2026-11-22). Here:
      1. An object is a patch of the color image that differs from the table around it. The table is
         modelled by a wide median of the image itself, so sunlit and shaded areas each look like table.
         A patch that is only darker, with the table's own color balance, is a shadow and is dropped.
      2. Its position is where the ray through its lowest (camera-side) pixels meets the table plane:
         a calibration-only measurement with no depth noise. The centre is half its width further back.
      3. Depth is used only for the height (and the table plane fit).
    Each object: id, color, center [x, y, z], top_m, size_cm [length, width, height], axis_deg, ends (long
    flat objects), distance_m, bbox_px, height_from ("depth" or "image").
    """
    import cv2

    img = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)        # BGR
    H, W = img.shape[:2]
    Tc = pose_matrix(pose)
    pts = world_points(frame, pose)
    plane = fit_table(pts, cfg.max_reach_m)
    a, b, c0 = plane
    ci = color_intrinsics(frame)

    # --- where in the image is reachable table?
    # The table ends where the depth stops being table: without this, the tape and whatever lies beyond
    # the edge would be placed on an imaginary extension of the table plane.
    us, vs = np.meshgrid(np.arange(0, W, 4), np.arange(0, H, 4))
    rays = np.stack([(us - ci["ppx"]) / ci["fx"], -(vs - ci["ppy"]) / ci["fy"], -np.ones_like(us, dtype=float)], -1) @ Tc[:3, :3].T
    o = Tc[:3, 3]
    denom = rays[..., 1] - a * rays[..., 0] - b * rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (a * o[0] + b * o[2] + c0 - o[1]) / denom
    hit = o + s[..., None] * rays
    on_table = (s > 0) & (np.hypot(hit[..., 0], hit[..., 2]) < cfg.max_reach_m) & (hit[..., 0] > cfg.min_x_m)
    region = cv2.resize(on_table.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    outside = cv2.dilate((~region).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    # --- not-table pixels: differ from the wide-median background, and aren't just its shadow
    bg = cv2.medianBlur(img, cfg.background_px).astype(np.int16)
    im = img.astype(np.int16)
    diff = np.abs(im - bg).max(axis=2)
    ratio = (im.astype(float) + 4.0) / (bg.astype(float) + 4.0)
    # A shadow is the table's own color, only darker: every channel dimmed by about the same factor.
    shadow = (ratio.max(axis=2) < 0.97) & (ratio.min(axis=2) > 0.5) & ((ratio.max(axis=2) - ratio.min(axis=2)) < 0.11)
    mask = (diff > cfg.contrast) & ~shadow & region
    arm_zone = None
    if arm_points is not None and len(arm_points):
        Tinv = np.linalg.inv(Tc)
        local = arm_points @ Tinv[:3, :3].T + Tinv[:3, 3]
        front = local[:, 2] < -0.05
        au = (ci["fx"] * local[front, 0] / -local[front, 2] + ci["ppx"]).astype(int)
        av = (-ci["fy"] * local[front, 1] / -local[front, 2] + ci["ppy"]).astype(int)
        ok = (au >= 0) & (au < W) & (av >= 0) & (av < H)
        arm = np.zeros((H, W), np.uint8)
        arm[av[ok], au[ok]] = 1
        k = 2 * cfg.arm_margin_px + 1
        arm_zone = cv2.dilate(arm, np.ones((k, k), np.uint8)) > 0
    m8 = mask.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)

    scale = frame.depth_mm.shape[1] / W
    height_map = pts[..., 1] - (a * pts[..., 0] + b * pts[..., 2] + c0)      # depth grid
    elev = float(np.arcsin(min(1.0, abs((Tc[:3, :3] @ np.array([0.0, 0.0, -1.0]))[1]))))   # camera look-down angle
    objects = []
    for i in range(1, n):
        x0, y0, w, h, area = (int(v) for v in stats[i])
        if not cfg.min_area_px <= area <= cfg.max_area_px:
            continue
        ys, xs = np.nonzero(labels[y0:y0 + h, x0:x0 + w] == i)
        ys, xs = ys + y0, xs + x0
        # Anything joined to the arm in the image is the arm: its cable loop and the jaw tips aren't in
        # the model, so a margin alone left them as "objects" whenever the arm was low.
        if arm_zone is not None and arm_zone[ys, xs].any():
            continue
        # A patch cut off by the edge of the working area is the table's edge, tape or a cable, not an object.
        if outside[ys, xs].any():
            continue
        # Camera-side edge: the lowest rows of the blob. Their rays meet the table where the object stands.
        low = ys >= np.percentile(ys, 88)
        base = _table_point(frame, Tc, plane, float(xs[low].mean()), float(ys[low].max()))
        left = _table_point(frame, Tc, plane, float(xs.min()), float(ys[low].max()))
        right = _table_point(frame, Tc, plane, float(xs.max()), float(ys[low].max()))
        top_ray = _table_point(frame, Tc, plane, float(xs[ys <= np.percentile(ys, 5)].mean()), float(ys.min()))
        if base is None or left is None or right is None:
            continue
        width = float(np.linalg.norm(left - right))
        # Height: depth inside the blob when there is enough of it, else from how far up the image it rises.
        dv = np.clip((ys * scale).astype(int), 0, height_map.shape[0] - 1)
        du = np.clip((xs * scale).astype(int), 0, height_map.shape[1] - 1)
        hs = height_map[dv, du]
        hs = hs[np.isfinite(hs) & (hs > -0.01) & (hs < 0.3)]
        image_top = float(np.linalg.norm((top_ray - base)[[0, 2]]) * np.tan(elev)) if top_ray is not None else 0.0
        if len(hs) >= 8:
            top, height_from = max(0.003, float(np.percentile(hs, 92))), "depth"
        else:
            # No depth on it: the image alone can't tell a tall thing from a flat one lying away from the
            # camera, so this height is only an upper bound.
            top, height_from = image_top, "image (upper bound)"
        away = base[[0, 2]] - Tc[[0, 2], 3]
        away /= max(float(np.linalg.norm(away)), 1e-9)                       # on the table, away from the camera
        flat = top < 0.018
        if flat:
            # A flat thing shows its whole top: every blob pixel lies (nearly) on the table.
            step = max(1, len(xs) // 400)
            foot = np.array([p[[0, 2]] for p in (_table_point(frame, Tc, plane, float(u), float(v))
                                                  for u, v in zip(xs[::step], ys[::step])) if p is not None])
            centre = (foot.min(axis=0) + foot.max(axis=0)) / 2
            cov = np.cov((foot - foot.mean(axis=0)).T) if len(foot) > 2 else np.eye(2)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, int(np.argmax(evals))]
            proj = (foot - foot.mean(axis=0)) @ axis
            perp = (foot - foot.mean(axis=0)) @ np.array([-axis[1], axis[0]])
            length, width2 = float(proj.max() - proj.min()), float(perp.max() - perp.min())
        else:
            # A standing thing hides its footprint: take it as deep as it is wide, behind its front edge.
            centre = base[[0, 2]] + away * min(width, 0.06) / 2
            axis = np.array([-away[1], away[0]])
            length, width2, proj, foot = width, min(width, 0.06), None, None
        bgr = img[ys, xs].mean(axis=0)
        obj = {
            "color": color_name((bgr[2], bgr[1], bgr[0])),
            "center": [round(float(centre[0]), 4), round(top / 2, 4), round(float(centre[1]), 4)],
            "top_m": round(top, 4),
            "size_cm": [round(max(length, width2) * 100, 1), round(min(length, width2) * 100, 1), round(top * 100, 1)],
            "axis_deg": round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1),
            "distance_m": round(float(np.hypot(*centre)), 3),
            "points": area,
            "bbox_px": [x0, y0, x0 + w, y0 + h],
            "height_from": height_from,
            "standing": not flat,
        }
        if flat and proj is not None and length > 2.5 * max(width2, 0.005):
            ends = []
            for end_proj in (proj.min(), proj.max()):
                e = foot.mean(axis=0) + axis * end_proj
                ends.append({"xz": [round(float(e[0]), 4), round(float(e[1]), 4)]})
            obj["ends"] = ends
        objects.append(obj)

    objects.sort(key=lambda ob: ob["distance_m"])
    for i, ob in enumerate(objects, 1):
        ob["id"] = i
    # Who is in whose way: edge-to-edge gap to the nearest other object.
    for ob in objects:
        best = None
        for other in objects:
            if other is ob:
                continue
            gap = float(np.hypot(ob["center"][0] - other["center"][0], ob["center"][2] - other["center"][2])) \
                - (ob["size_cm"][1] + other["size_cm"][1]) / 200
            if best is None or gap < best[0]:
                best = (gap, other["id"])
        if best is not None and best[0] < cfg.crowd_gap_m:
            ob["crowded_by"] = {"id": best[1], "gap_cm": round(max(best[0], 0.0) * 100, 1)}
    return objects
