"""Fit the depth camera pose and the arm model's joint corrections together, from depth frames of the arm.

Why: the station-viewer's mapping squeezes each servo's calibrated range into the URDF joint limits, which
under-reports real angles by 2-5% (arm_model.ArmModel.spans). Fixing the scale moves where the model thinks
the arm is, so the camera pose (fitted against the old model) and the joint zeros must be fitted again.

Parameters: camera pose (x, y, z, yaw, pitch, roll) + zero offsets of the pitch joints (shoulder, elbow,
wrist pitch). Residuals: depth points on the arm -> nearest model surface point (every sample pose), and
table points -> the plane y = 0 (keeps the camera from tilting to hide arm errors). Robust (Huber) loss.

CLI:
    python arm_fit.py capture <dir> <name> '<norm json>'   save the current RealSense frame + joint norms
    python arm_fit.py fit <dir> [--save]                   fit; --save writes `arm_fit` into calibration.json
    python arm_fit.py check <dir> <out dir>                draw the model over each sample's photo + jaw offsets

Always run `check` after a fit and look at the pictures. The fit's own arm_error_mm cannot show a model
that is consistently beside the real arm: on 2026-11-22 it reported 6.5-11 mm while the pictures showed the
model's jaw centre sitting on one real jaw tip, 2 cm off.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import arm_model as AM
import depth as D

HERE = Path(__file__).resolve().parent
URDF = Path(os.environ.get("NORMA_CORE_REPO", str(HERE.parent / "norma-core"))).expanduser() / "software/station/clients/station-viewer/public/devices/elrobot/elrobot_follower.urdf"
CALIBRATION = HERE / "realsense-bridge" / "calibration.json"
FIT_JOINTS = ("rev_motor_02", "rev_motor_04", "rev_motor_06")
POSE_KEYS = ("x", "y", "z", "yaw", "pitch", "roll")
ARM_BAND_M = 0.06          # depth points this close to the (current) model count as arm points
MAX_ARM_POINTS = 1500      # per sample
TABLE_POINTS = 600         # per sample


def servo_spans(ranges: dict) -> dict[str, float]:
    """{motor id: (range_min, range_max)} -> {URDF joint: true rotation (rad)} (ranges may wrap past 4095)."""
    out = {}
    for mid, (lo, hi) in ranges.items():
        size = hi - lo if hi >= lo else (4096 - lo) + hi
        out[AM.ARM_JOINTS[int(mid) - 1]] = size * 2 * math.pi / 4096
    return out


def _local_points(frame: D.Frame) -> np.ndarray:
    """Depth pixels in the camera rig frame (x right, y up, -z forward), (N, 3), no NaNs."""
    zero = {k: 0.0 for k in POSE_KEYS}
    p = D.world_points(frame, zero).reshape(-1, 3)
    return p[~np.isnan(p[:, 0])]


def _to_world(params: np.ndarray, local: np.ndarray) -> np.ndarray:
    T = D.pose_matrix(dict(zip(POSE_KEYS, params[:6])))
    return local @ T[:3, :3].T + T[:3, 3]


class Fit:
    def __init__(self, model: AM.ArmModel, samples: list[tuple[np.ndarray, dict]], pose: dict,
                 offsets: dict[str, float]):
        from scipy.spatial import cKDTree  # noqa: F401  (fail early if scipy is missing)

        self.model = model
        self.samples = samples                       # [(local points, norm)]
        self.base_offsets = dict(offsets)
        self.x0 = np.array([pose[k] for k in POSE_KEYS] + [offsets.get(j, 0.0) for j in FIT_JOINTS])
        self.arm_sel: list[np.ndarray] = []
        self.table_sel: list[np.ndarray] = []

    def _model_points(self, params: np.ndarray, norm: dict) -> np.ndarray:
        self.model.offsets = {**self.base_offsets, **dict(zip(FIT_JOINTS, params[6:]))}
        return self.model.surface_points(norm)

    def select(self, params: np.ndarray) -> None:
        """Pick which depth points are arm / table at the current estimate (re-done between rounds, ICP-style)."""
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(0)
        self.arm_sel, self.table_sel = [], []
        for local, norm in self.samples:
            w = _to_world(params, local)
            near = cKDTree(self._model_points(params, norm)).query(w, distance_upper_bound=ARM_BAND_M)[0]
            arm = local[np.isfinite(near) & (w[:, 1] > 0.012)]
            if len(arm) > MAX_ARM_POINTS:
                arm = arm[rng.choice(len(arm), MAX_ARM_POINTS, replace=False)]
            r = np.hypot(w[:, 0], w[:, 2])
            tab = local[~np.isfinite(near) & (np.abs(w[:, 1]) < 0.02) & (r > 0.12) & (r < 0.55)]
            if len(tab) > TABLE_POINTS:
                tab = tab[rng.choice(len(tab), TABLE_POINTS, replace=False)]
            self.arm_sel.append(arm)
            self.table_sel.append(tab)

    def residuals(self, params: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        res = []
        for (local, norm), arm, tab in zip(self.samples, self.arm_sel, self.table_sel):
            if len(arm):
                res.append(cKDTree(self._model_points(params, norm)).query(_to_world(params, arm))[0])
            if len(tab):
                res.append(_to_world(params, tab)[:, 1])
        return np.concatenate(res)

    def arm_error_mm(self, params: np.ndarray) -> list[float]:
        """Median arm-point distance to the model, per sample (mm): the number that says if the fit is good."""
        from scipy.spatial import cKDTree

        out = []
        for (local, norm), arm in zip(self.samples, self.arm_sel):
            d = cKDTree(self._model_points(params, norm)).query(_to_world(params, arm))[0] if len(arm) else [np.nan]
            out.append(round(float(np.median(d)) * 1000, 1))
        return out

    def run(self, rounds: int = 4) -> np.ndarray:
        from scipy.optimize import least_squares

        x = self.x0.copy()
        for _ in range(rounds):
            self.select(x)
            x = least_squares(self.residuals, x, loss="huber", f_scale=0.008, x_scale=0.02,
                              diff_step=1e-4, max_nfev=200).x
        self.select(x)
        return x


def load_samples(folder: Path) -> list[tuple[str, D.Frame, dict]]:
    out = []
    for meta in sorted(folder.glob("*.json")):
        info = json.loads(meta.read_text())
        frame = D.parse_frame(meta.with_suffix(".bin").read_bytes())
        out.append((meta.stem, frame, {int(k): float(v) for k, v in info["norm"].items()}))
    return out


def fit_folder(folder: Path, save: bool = False) -> dict:
    cal = json.loads(CALIBRATION.read_text())
    serial = next(iter(cal.get("joint_offsets") or {"5B8E115950": {}}))
    info = json.loads(next(iter(sorted(folder.glob("*.json")))).read_text())
    spans = servo_spans({int(k): v for k, v in info["ranges"].items()})
    raw = load_samples(folder)
    samples = [(_local_points(f), n) for _, f, n in raw]
    offsets = dict((cal.get("joint_offsets") or {}).get(serial, {}))

    model = AM.ArmModel.load(URDF, max_points_per_link=1500)
    report = {"samples": [name for name, _, _ in raw]}

    # Before: the viewer's mapping with the viewer's calibration.
    old = Fit(model, samples, cal["camera_pose"], offsets)
    model.spans = {}
    old.select(old.x0)
    report["before_arm_error_mm"] = old.arm_error_mm(old.x0)

    # Same, but with true joint scales and the joint zeros + camera fitted.
    model.spans = spans
    fit = Fit(model, samples, cal["camera_pose"], offsets)
    t0 = time.time()
    x = fit.run()
    report["after_arm_error_mm"] = fit.arm_error_mm(x)
    report["fit_s"] = round(time.time() - t0, 1)
    pose = dict(zip(POSE_KEYS, map(float, x[:6])))
    fitted = {**offsets, **dict(zip(FIT_JOINTS, map(float, x[6:])))}
    report["camera_pose"] = {k: round(v, 4) for k, v in pose.items()}
    report["camera_moved_cm"] = round(float(np.linalg.norm(x[:3] - old.x0[:3])) * 100, 1)
    report["joint_zero_offsets_deg"] = {j: round(math.degrees(fitted[j]), 2) for j in FIT_JOINTS}
    if save:
        cal["arm_fit"] = {"serial": serial, "camera_pose": pose, "joint_offsets": fitted, "joint_spans": spans,
                          "samples": report["samples"], "arm_error_mm": report["after_arm_error_mm"],
                          "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = CALIBRATION.with_suffix(".tmp")
        tmp.write_text(json.dumps(cal, indent=2))
        tmp.replace(CALIBRATION)
        report["saved"] = str(CALIBRATION)
    return report


def check_folder(folder: Path, out: Path) -> dict:
    """Draw the calibrated model over each sample's color image (yellow arm, cyan gripper, magenta jaws, red
    ring = model jaw-end centre, green rings = jaw tips found in the image) and report, for the samples with
    the jaws pointing down, where the real jaw midpoint is in the gripper frame (the model says x = 0)."""
    import io

    from PIL import Image, ImageDraw

    cal = json.loads(CALIBRATION.read_text())
    fit = cal.get("arm_fit") or {"camera_pose": cal["camera_pose"], "joint_offsets": {}, "joint_spans": {}}
    model = AM.ArmModel.load(URDF, max_points_per_link=3000)
    model.offsets, model.spans = fit["joint_offsets"], fit.get("joint_spans", {})
    pose = fit["camera_pose"]
    Tc = D.pose_matrix(pose)
    Tinv = np.linalg.inv(Tc)
    end = model.joints[AM.END_LINK_JOINT].child
    gripper = set(model._subtree_links(end))
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, frame, norm in load_samples(folder):
        img = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
        ci = D.color_intrinsics(frame)
        draw = ImageDraw.Draw(img)

        def project(points: np.ndarray) -> np.ndarray:
            local = points @ Tinv[:3, :3].T + Tinv[:3, 3]
            return np.c_[ci["fx"] * local[:, 0] / -local[:, 2] + ci["ppx"],
                         -ci["fy"] * local[:, 1] / -local[:, 2] + ci["ppy"]]

        tfs = model.link_transforms(norm)
        for link, pts in model.link_points.items():
            if not len(pts):
                continue
            color = (255, 0, 255) if "Jaw" in link else (0, 255, 255) if link in gripper else (255, 230, 0)
            for u, v in project(pts @ tfs[link][:3, :3].T + tfs[link][:3, 3])[::3]:
                draw.point((u, v), fill=color)
        T = tfs[end]
        jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
        u, v = project(jaw_end[None])[0]
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 0, 0), width=2)
        entry: dict = {"jaws_point_down": bool(model.approach_dir(norm)[1] < -0.7)}
        found = D.find_jaw_tips(frame, pose, jaw_end) if entry["jaws_point_down"] else None
        if found:
            for tip in found["tips"]:
                tu, tv = project(np.array(tip)[None])[0]
                draw.ellipse((tu - 5, tv - 5, tu + 5, tv + 5), outline=(0, 255, 0), width=2)
            local = T[:3, :3].T @ (np.array(found["midpoint"]) - T[:3, 3])
            entry["real_jaw_midpoint_sideways_mm"] = round(float(local[0]) * 1000, 1)
            entry["jaw_gap_mm"] = round(found["gap_m"] * 1000, 1)
        img.save(out / f"{name}.png")
        report[name] = entry
    return {"pictures": str(out), "samples": report}


def capture(folder: Path, name: str, norm: dict, ranges: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    import websockets

    async def grab() -> bytes:
        async with websockets.connect("ws://127.0.0.1:8890", max_size=None) as ws:
            while True:
                msg = await asyncio.wait_for(ws.recv(), 3.0)
                if isinstance(msg, bytes):
                    return msg

    data = asyncio.run(grab())
    D.parse_frame(data)  # validate
    (folder / f"{name}.bin").write_bytes(data)
    (folder / f"{name}.json").write_text(json.dumps({"norm": norm, "ranges": ranges}))
    return {"saved": name}


if __name__ == "__main__":
    cmd, folder = sys.argv[1], Path(sys.argv[2])
    if cmd == "capture":
        args = json.loads(sys.argv[4])
        print(capture(folder, sys.argv[3], args["norm"], args["ranges"]))
    elif cmd == "check":
        print(json.dumps(check_folder(folder, Path(sys.argv[3])), indent=2))
    elif cmd == "fit":
        print(json.dumps(fit_folder(folder, save="--save" in sys.argv), indent=2))
