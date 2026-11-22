"""Turn a checked step into keyframes, run it, and confirm from joint feedback (goal G5).

Gestures are small offsets from the pose they start in (config/poses.yaml); point/home/ready go
to taught absolute poses; jog adds degrees to one joint. Every keyframe list goes through
safety.check_waypoints() right before it is sent, so limits hold whatever the config says.
"""
from __future__ import annotations

import statistics

from arm import Arm, load_yaml
from safety import check_waypoints

# Contact sensing for "touch": creep one joint in small increments; after each one read the motors.
# Contact = the moving joint falls behind its target (it is pushing on something), or the load
# current on a lifting joint rises above what it drew while moving freely just before.
TOUCH_STEP_DEG = 1.0
CONTACT_LAG_DEG = 1.5        # servo behind its target by this much = blocked
CONTACT_CURRENT_RISE = 6     # ST3215 current units over the free-motion level (idle ~0-5, desk contact ~22)
PRESS_DEG = 0.5              # after contact, rest this far past the contact point: light, steady pressure
WATCH = ("shoulder", "elbow", "wrist_pitch")


class Skills:
    def __init__(self, io, arm: Arm, poses: dict | None = None):
        self.io, self.arm = io, arm
        self.poses = poses or load_yaml("poses.yaml")
        m = self.poses.get("motion") or {}
        self.speed = float(m.get("speed_scale", 0.5))
        self.tol = float(m.get("reach_tol", 0.03))
        self.grip = m.get("gripper") or {"open": 0.65, "close": 0.0}

    @property
    def taught_slots(self) -> set[str]:
        return {k for k, v in (self.poses.get("slots") or {}).items() if v}

    # -- planning (pure, testable) -------------------------------------------------
    def plan(self, step: dict, current: dict[str, float]) -> list[dict]:
        do = step["do"]
        if do == "gesture" and step["name"] in ("home", "ready"):
            return [{"joints": dict(self.poses[step["name"]]), "duration_s": 0.0}]
        if do == "gesture":
            used = self._joints_in(step["name"])   # every frame sets all of them, so {} = back to start
            return [{"joints": {j: current[j] + (offsets or {}).get(j, 0.0) for j in used},
                     "duration_s": float(dur)}
                    for offsets, dur in self.poses["gestures"][step["name"]]]
        if do == "point":
            return [{"joints": dict(self.poses["slots"][step["target"]]), "duration_s": 0.0}]
        if do == "jog":
            j = step["joint"]
            return [{"joints": {j: current[j] + step["degrees"] / self.arm.deg_per_norm(j)}, "duration_s": 0.0}]
        if do == "reach":   # shoulder leans out (+) while the elbow opens (-), together
            s = 1.0 if step["direction"] == "forward" else -1.0
            d = step["degrees"]
            return [{"joints": {"shoulder": current["shoulder"] + s * d / self.arm.deg_per_norm("shoulder"),
                                "elbow": current["elbow"] - s * d / self.arm.deg_per_norm("elbow")},
                     "duration_s": 0.0}]
        if do == "gripper":
            return [{"joints": {"gripper": float(self.grip[step["state"]])}, "duration_s": 0.0}]
        raise ValueError(f"no plan for {step}")

    def _joints_in(self, gesture: str) -> set[str]:
        return {j for offsets, _ in self.poses["gestures"][gesture] for j in (offsets or {})}

    # -- running ---------------------------------------------------------------------
    @property
    def absolute_ok(self) -> bool:
        """Taught poses (home/ready/A/B/C) belong to the configured follower. On any other bus
        (e.g. a leader-arm test) only relative moves are allowed."""
        return getattr(self.io, "follower", self.arm.follower_bus) == self.arm.follower_bus

    async def touch(self, step: dict) -> dict:
        """Move one joint toward a surface in TOUCH_STEP_DEG increments until contact, then stop
        just past the contact point (light pressure). step = {joint, degrees (max travel, signed =
        direction), speed}. Result: reached (the step ran), contact (bool), moved_deg, signal."""
        j, sign, travel = step["joint"], (1 if step["degrees"] > 0 else -1), abs(step["degrees"])
        speed = self.speed * (0.5 if step.get("speed") == "slow" else 1.0)
        dpn = self.arm.deg_per_norm(j)
        watch = list(dict.fromkeys([w for w in WATCH if w in self.arm.joints] + [j]))
        base = {"reached": True, "errors": {}, "clamps": [], "grasp": None, "abort_reason": None}

        motors = await self.io.read_joints()
        self.arm.update_ranges(motors)
        target = motors[j]["position_norm"]
        hold, _ = check_waypoints(self.arm, [{"joints": {j: target}, "duration_s": 0.0}])
        await self.io.play(hold, speed)            # torque on and settled before the baseline
        free: dict[str, list[int]] = {w: [] for w in watch}
        for _ in range(2):
            m = await self.io.read_joints()
            for w in watch:
                free[w].append(m[w]["current"])

        def check(m) -> str | None:
            lag = sign * (target - m[j]["position_norm"]) * dpn
            if lag >= CONTACT_LAG_DEG:
                return f"{j} held back {lag:.1f}° short of its target"
            for w in watch:
                ref = statistics.median(free[w][-4:])
                if m[w]["current"] - ref >= CONTACT_CURRENT_RISE:
                    return f"{w} current rose {ref:g} → {m[w]['current']}"
            return None

        moved = 0.0
        while moved < travel - 1e-6:
            inc = min(TOUCH_STEP_DEG, travel - moved)
            want = target + sign * inc / dpn
            wp, clamps = check_waypoints(self.arm, [{"joints": {j: want}, "duration_s": 0.0}])
            if clamps:
                return {**base, "contact": False, "moved_deg": round(moved, 1), "clamps": clamps,
                        "signal": f"reached the {j} joint limit before touching anything"}
            res = await self.io.play(wp, speed)
            target = want
            moved += inc
            reason = str(res.get("abort_reason") or "")
            if not res.get("completed"):
                if "current" in reason or "tracking" in reason or "behind" in reason:
                    # the motion engine's own guard tripped: that is a hard contact
                    return {**base, "contact": True, "moved_deg": round(moved, 1), "signal": reason}
                return {**base, "reached": False, "contact": None, "moved_deg": round(moved, 1),
                        "abort_reason": reason or "stopped"}
            m = await self.io.read_joints()
            why = check(m)
            if why:
                m2 = await self.io.read_joints()     # confirm on a second reading: one noisy sample isn't contact
                why2 = check(m2)
                if why2:
                    at = m2[j]["position_norm"]
                    rest, _ = check_waypoints(self.arm, [{"joints": {j: at + sign * PRESS_DEG / dpn},
                                                           "duration_s": 0.0}])
                    await self.io.play(rest, speed)
                    moved_to_contact = abs(at - (target - sign * moved / dpn)) * dpn
                    return {**base, "contact": True, "moved_deg": round(moved_to_contact, 1), "signal": why2,
                            "current": {w: m2[w]["current"] for w in watch}}
            for w in watch:
                free[w].append(m[w]["current"])
        return {**base, "contact": False, "moved_deg": round(moved, 1),
                "signal": f"no contact within {travel:g}° of {j}"}

    async def run(self, step: dict) -> dict:
        """Plan, clamp, send, then verify every commanded joint from fresh joint readings."""
        if step["do"] == "touch":
            return await self.touch(step)
        if not self.absolute_ok and (step["do"] == "point" or step.get("name") in ("home", "ready")):
            return {"reached": False, "abort_reason": "taught poses are disabled on this arm (not the follower)",
                    "errors": {}, "clamps": [], "elapsed_s": 0}
        motors = await self.io.read_joints()
        self.arm.update_ranges(motors)
        current = {n: m["position_norm"] for n, m in motors.items()}
        waypoints, clamps = check_waypoints(self.arm, self.plan(step, current))
        speed = self.speed * (0.5 if step.get("speed") == "slow" else 1.0)
        res = await self.io.play(waypoints, speed)
        after = {n: m["position_norm"] for n, m in (await self.io.read_joints()).items()}
        final = waypoints[-1]["joints"]
        errors = {j: round(abs(after[j] - v), 3) for j, v in final.items() if j in after}
        reached = bool(res.get("completed")) and all(e <= self.tol for e in errors.values())
        grasp = None
        if step["do"] == "gripper" and step["state"] == "close" and res.get("completed"):
            # stopping short of fully closed means something is between the fingers: that's the goal
            grasp = bool(res.get("grasp_detected")) or errors.get("gripper", 0) > self.tol
            reached = True
        return {"reached": reached, "abort_reason": res.get("abort_reason"), "errors": errors, "grasp": grasp,
                "gripper_at": after.get("gripper"),
                "clamps": clamps, "elapsed_s": res.get("elapsed_s"), "waypoints": waypoints}
