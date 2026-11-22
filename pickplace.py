"""Pure decision logic for camera-guided pick and place (no I/O, unit-testable).

The server's pick_and_place tool loops: grab a gripper-camera frame -> detect -> one of the
step functions below says which small joint move to make next (or that the arm is aligned).
Image coordinates are normalised to 0..1 (x right, y down) so the logic doesn't depend on
the camera's resolution.

Joint effects measured on the ElRobot follower (2026-11-21, see memory notes):
  base_yaw +        -> scene moves right in the gripper image (x increases), ~1000 px/norm at 299 px.
  "reach" (shoulder +, elbow -) -> the arm extends forward; objects ahead slide down the image.
  shoulder + alone  -> the gripper goes forward/down (final descent onto the motor).
These are starting gains: AdaptiveGain refines them from the observed response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULTS: dict = {
    # poses (normalised joint positions)
    "home": {"base_yaw": 0.512, "shoulder": 0.01, "upper_arm_roll": 0.502, "elbow": 0.944,
             "forearm_roll": 0.47, "wrist_pitch": 0.46, "wrist_roll": 0.543, "gripper": 0.558},
    "hover": {"shoulder": 0.20, "elbow": 0.85},           # look-out pose above the table in front of the arm
    "carry": {"shoulder": 0.38, "elbow": 0.62},           # motor lifted clear of the box walls
    "gripper_open": 0.89,
    "gripper_close": 0.30,
    "gripper_release": 0.75,
    # pick alignment targets in the gripper image (normalised)
    "jaw_center_x": 0.45,       # centre of the open jaws (135 px of 299)
    "x_tol": 0.05,
    "pick_bottom_touch": 0.93,  # blob reaches the image bottom -> between the jaw tips
    "pick_top_grasp": 0.56,     # blob top this low -> jaws are around the motor (success frame: 0.57)
    "pick_min_width": 0.30,     # ...and this wide: close up (0.37 at a real grasp, 0.16 seen from home)
    "reach_step": [0.05, -0.04],      # (shoulder, elbow) per approach step
    "descend_step": [0.02, 0.0],      # (shoulder, elbow) per final descent step
    "pick_max_steps": 16,
    # grasp check: where the gripper stalls on the motor (0.42-0.44 seen on hardware)
    "grasp_min": 0.33,
    "grasp_max": 0.56,
    "grasp_extra_down": 0.015,  # shoulder + before closing: 0.615 closed on an upright motor's top, 0.624 gripped
    # held check: with the motor held its top shows above the closed jaws
    "held_top_max": 0.62,
    # place alignment (marker in the gripper image at carry height)
    "marker": "white",          # "white" (white sticker/label) or "hue" (coloured sticker, marker_hue)
    "marker_hue": [0.20, 0.75],
    "marker_min_y": 0.35,       # ignore markers higher than this: far background (e.g. a white bag)
    "marker_min_area": 0.006,   # smaller white spots are glare
    "held_region": [0.25, 0.52, 0.68, 1.0],  # x0,y0,x1,y1 where the held motor (and its label) sits
    "place_view": {"shoulder": 0.36, "elbow": 0.71},  # elbow folded: camera looks down at the box floor
    # clearance rule: sideways swings only at/above transit height (gripper, and anything it
    # carries, clears the box walls and other objects); yaw changes up to small_yaw are allowed
    # low down (alignment corrections over the target)
    # 0.36/0.71 was NOT enough: the held motor dragged the box wall while yawing (2026-11-21)
    "transit": {"shoulder": 0.31, "elbow": 0.78},
    "small_yaw": 0.03,
    # calibrated 2026-11-21: marker at (0.50, 0.48) in the view pose -> the motor landed 0.12 left of
    # the sticker, so the target x is shifted to centre it
    "place_target_x": 0.42,
    "place_target_y": 0.48,
    "place_tol": 0.05,
    "place_reach_dir": [1.0, -1.25],  # (shoulder, elbow) per unit of reach at carry height
    "place_descend": [0.16, -0.164],  # (shoulder, elbow) view pose -> release pose (hand-tested drop)
    "place_max_steps": 14,
    "scan_yaws": [0.0, 0.08, -0.08, 0.16, -0.16, 0.24, -0.24, 0.32, -0.32],
    # safety bounds for camera-driven moves
    "yaw_bounds": [0.15, 0.85],
    "shoulder_max": 0.68,
    "elbow_min": 0.40,
    "max_step": 0.05,
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class AdaptiveGain:
    """Image response per unit joint move (normalised image units / norm). Starts from a prior,
    then blends in what the camera actually showed after each move, so a wrong prior (or a
    camera bumped a few degrees) gets corrected within a couple of steps."""
    gain: float
    lo: float
    hi: float
    history: list = field(default_factory=list)

    def command(self, error: float, max_step: float, damping: float = 0.8) -> float:
        return clamp(damping * error / self.gain, -max_step, max_step)

    def observe(self, move: float, response: float) -> None:
        if abs(move) < 0.008:
            return  # too small to measure through detection noise
        measured = response / move
        self.history.append(round(measured, 3))
        if measured <= 0:
            return  # opposite sign: a detection glitch or another joint dominated; ignore
        self.gain = clamp(0.5 * self.gain + 0.5 * measured, self.lo, self.hi)


def norm_blob(b, w: int, h: int) -> dict:
    return {"cx": b.cx / w, "cy": b.cy / h, "left": b.left / w, "right": b.right / w,
            "top": b.top / h, "bottom": b.bottom / h, "area": b.area / (w * h)}


def pick_target(blobs: list[dict], cfg: dict) -> dict | None:
    """The motor among dark blobs: the biggest one, preferring blobs near the jaw centre."""
    if not blobs:
        return None
    return max(blobs, key=lambda b: b["area"] / (1.0 + 3.0 * abs(b["cx"] - cfg["jaw_center_x"])))


def pick_step(blob: dict, joints: dict, yaw_gain: AdaptiveGain, cfg: dict) -> dict:
    """Next move toward a grasp. Returns {"phase", "done", "targets"} where targets are absolute
    joint positions (yaw correction + reach or descent), or done=True when the jaws surround it."""
    ex = cfg["jaw_center_x"] - blob["cx"]
    centred = abs(ex) <= cfg["x_tol"]
    deep = blob["bottom"] >= cfg["pick_bottom_touch"]
    close_up = (blob["right"] - blob["left"]) >= cfg["pick_min_width"]
    if deep and blob["top"] >= cfg["pick_top_grasp"] and centred and close_up:
        return {"phase": "grasp", "done": True, "targets": {}}
    dyaw = yaw_gain.command(ex, cfg["max_step"]) if not centred or deep else 0.0
    if deep:
        # between the jaw tips: only go down, and only once centred (else the jaws land on it)
        ds, de = cfg["descend_step"] if centred else (0.0, 0.0)
        phase = "descend" if centred else "centre"
    else:
        ds, de = cfg["reach_step"]
        phase = "approach"
    t = {"base_yaw": clamp(joints["base_yaw"] + dyaw, *cfg["yaw_bounds"]),
         "shoulder": min(joints["shoulder"] + ds, cfg["shoulder_max"]),
         "elbow": max(joints["elbow"] + de, cfg["elbow_min"])}
    limited = (t["shoulder"] >= cfg["shoulder_max"] and ds > 0) or (t["elbow"] <= cfg["elbow_min"] and de < 0)
    return {"phase": phase, "done": False, "targets": t, "limited": limited, "dyaw": round(dyaw, 4)}


def grasp_verdict(gripper_pos: float, cfg: dict) -> str:
    """'ok' if the jaws stopped on something motor-sized; 'missed' if they closed on nothing;
    'on_top' if they stalled early (a jaw landed on the motor's top edge)."""
    if gripper_pos < cfg["grasp_min"]:
        return "missed"
    if gripper_pos > cfg["grasp_max"]:
        return "on_top"
    return "ok"


def is_held(blobs: list[dict], cfg: dict) -> bool:
    """Motor still in the jaws: a dark blob near the jaw centre whose top rises above the jaws."""
    return any(abs(b["cx"] - cfg["jaw_center_x"]) <= 0.15 and b["bottom"] >= 0.9 and b["top"] <= cfg["held_top_max"]
               for b in blobs)


def is_high(joints: dict, cfg: dict, tol: float = 0.02) -> bool:
    """At or above transit height: shoulder no further forward/down and elbow no more extended
    than the transit pose (shoulder + is forward/down, elbow - is reaching out)."""
    t = cfg["transit"]
    return joints["shoulder"] <= t["shoulder"] + tol and joints["elbow"] >= t["elbow"] - tol


def pick_marker(markers: list[dict], cfg: dict, near: dict | None = None) -> dict | None:
    """The box marker among candidates: skip far-background ones near the top of the image; while
    tracking, take the one nearest the previous detection; else the largest."""
    x0, y0, x1, y1 = cfg["held_region"]
    ms = [m for m in markers if m["cy"] >= cfg["marker_min_y"] and m["area"] >= cfg["marker_min_area"]
          and not (x0 <= m["cx"] <= x1 and y0 <= m["cy"] <= y1)]
    if not ms:
        return None
    if near is not None:
        return min(ms, key=lambda m: (m["cx"] - near["cx"]) ** 2 + (m["cy"] - near["cy"]) ** 2)
    return max(ms, key=lambda m: m["area"])


def place_step(marker: dict, joints: dict, yaw_gain: AdaptiveGain, reach_gain: AdaptiveGain, cfg: dict) -> dict:
    """Next move to bring the box marker to the calibrated drop spot in the gripper image."""
    ex = cfg["place_target_x"] - marker["cx"]
    ey = cfg["place_target_y"] - marker["cy"]
    if abs(ex) <= cfg["place_tol"] and abs(ey) <= cfg["place_tol"]:
        return {"done": True, "targets": {}}
    dyaw = yaw_gain.command(ex, cfg["max_step"]) if abs(ex) > cfg["place_tol"] else 0.0
    # reach + slides the scene down (y +): marker too high in the image (ey > 0) -> reach forward
    dr = reach_gain.command(ey, cfg["max_step"]) if abs(ey) > cfg["place_tol"] else 0.0
    rs, re = cfg["place_reach_dir"]
    t = {"base_yaw": clamp(joints["base_yaw"] + dyaw, *cfg["yaw_bounds"]),
         "shoulder": clamp(joints["shoulder"] + rs * dr, 0.05, cfg["shoulder_max"]),
         "elbow": clamp(joints["elbow"] + re * dr, cfg["elbow_min"], 0.95)}
    return {"done": False, "targets": t, "dyaw": round(dyaw, 4), "dreach": round(dr, 4)}
