"""Safety gate: every step Claude proposes passes through here before it can become motion.

parse_step() validates one step (shape, allowed values) and returns a clean dict or a reason.
Jog sizes are clamped to +-jog_max_deg, and every target joint value is clamped to Arm.limits().
Nothing outside the limits ever reaches Station (goal G3); each clamp is reported for the log.
"""
from __future__ import annotations

import json

from arm import Arm

GESTURES = ("nod", "shake", "shrug", "wave", "home", "ready")
SLOTS = ("A", "B", "C")


def parse_step(raw: str | dict, arm: Arm, taught_slots: set[str], has_gripper: bool = True
               ) -> tuple[dict | None, str]:
    try:
        step = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError) as e:
        return None, f"unreadable step ({e})"
    do = step.get("do")
    if do == "gesture":
        if step.get("name") in GESTURES:
            return {"do": "gesture", "name": step["name"]}, ""
        return None, f"unknown gesture {step.get('name')!r}"
    if do == "point":
        t = step.get("target")
        if t not in SLOTS:
            return None, f"unknown target {t!r}"
        if t not in taught_slots:
            return None, f"slot {t} has no taught pose yet"
        return {"do": "point", "target": t}, ""
    speed = step.get("speed", "normal")
    if do in ("jog", "reach", "replay", "touch") and speed not in ("slow", "normal"):
        return None, f"unknown speed {speed!r}"
    if do == "goal":
        g = str(step.get("goal") or "").strip()
        return ({"do": "goal", "goal": g[:300]}, "") if g else (None, "empty goal")
    if do == "replay":
        from recorder import safe_name
        return {"do": "replay", "speed": speed, "name": safe_name(str(step.get("name") or "")),
                "pen": step.get("pen") is True}, ""
    if do == "reach":
        d, deg = step.get("direction"), step.get("degrees")
        if d not in ("forward", "back"):
            return None, f"unknown reach direction {d!r}"
        if not {"shoulder", "elbow"} <= set(arm.jog_joints):
            return None, "this arm can't reach"
        if isinstance(deg, bool) or not isinstance(deg, (int, float)):
            return None, f"degrees must be a number, got {deg!r}"
        mag = min(abs(float(deg)), arm.jog_max_deg)
        out = {"do": "reach", "direction": d, "degrees": mag, "speed": speed}
        return out, (f"reach clamped from {deg} to {mag:g} degrees" if mag != abs(deg) else "")
    if do == "jog":
        j, deg = step.get("joint"), step.get("degrees")
        if j not in arm.jog_joints:
            return None, f"joint {j!r} can't be jogged"
        if isinstance(deg, bool) or not isinstance(deg, (int, float)):
            return None, f"degrees must be a number, got {deg!r}"
        cap = arm.jog_max_deg
        clamped = max(-cap, min(cap, float(deg)))
        out = {"do": "jog", "joint": j, "degrees": clamped, "speed": speed}
        return out, (f"jog clamped from {deg} to {clamped:g} degrees" if clamped != deg else "")
    if do == "touch":
        j, deg = step.get("joint"), step.get("degrees")
        if j not in arm.jog_joints:
            return None, f"joint {j!r} can't be used to touch"
        if isinstance(deg, bool) or not isinstance(deg, (int, float)) or deg == 0:
            return None, f"degrees must be a non-zero number, got {deg!r}"
        cap = arm.jog_max_deg
        clamped = max(-cap, min(cap, float(deg)))
        out = {"do": "touch", "joint": j, "degrees": clamped, "speed": speed}
        return out, (f"touch travel clamped from {deg} to {clamped:g} degrees" if clamped != deg else "")
    if do == "gripper":
        if not has_gripper:
            return None, "this arm has no gripper"
        if step.get("state") in ("open", "close"):
            return {"do": "gripper", "state": step["state"]}, ""
        return None, f"unknown gripper state {step.get('state')!r}"
    return None, f"unknown step type {do!r}"


def clamp_targets(arm: Arm, targets: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Clamp every joint target into the arm's limits. Returns (safe targets, clamp notes)."""
    out, notes = {}, []
    for name, v in targets.items():
        if name not in arm.joints:
            notes.append(f"dropped unknown joint {name}")
            continue
        safe, hit = arm.clamp(name, float(v))
        if hit:
            lo, hi = arm.limits(name)
            notes.append(f"{name} {v:.3f} clamped to {safe:.3f} (limits {lo:.2f}..{hi:.2f})")
        out[name] = round(safe, 4)
    return out, notes


def check_waypoints(arm: Arm, waypoints: list[dict]) -> tuple[list[dict], list[str]]:
    """Final gate on a keyframe list just before it is sent to Station."""
    safe, notes = [], []
    for w in waypoints:
        joints, n = clamp_targets(arm, w.get("joints") or {})
        notes += n
        if "t" in w:   # recorded samples keep their timestamps
            safe.append({"t": max(0.0, float(w["t"])), "joints": joints})
        else:
            safe.append({"joints": joints, "duration_s": max(0.0, float(w.get("duration_s", 0.0)))})
    return safe, notes
