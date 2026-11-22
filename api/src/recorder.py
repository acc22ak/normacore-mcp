"""Record a movement made by hand, then replay it.

  record  -> note Station's clock (the arm must be limp: torque off)
  (move the arm by hand)
  done    -> read that stretch back from Station's joint log, trim the still ends, thin to 20 Hz,
             save logs/recordings/<time>.json (and latest.json)
  repeat  -> glide to the recording's first pose, then play the samples with their original
             timing (half speed by default); every sample goes through the joint limits

Station logs every joint all the time, so recording needs no extra sampling loop here.
Motor current is meaningless while the arm is moved by hand (torque off, the motors don't push),
so it is sampled during the replay instead, where it shows effort along the path.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from arm import Arm
from safety import check_waypoints

REC_DIR = Path(__file__).resolve().parents[1] / "logs" / "recordings"
def named_dir() -> Path:                # "record c" ... "done" -> named/c.json; "draw c" replays it
    return REC_DIR / "named"
PEN_LIFT_DEG = 6.0                     # pen mode: elbow lift (elbow - = up in the drawing pose)
PEN_DESCEND_S = 1.2


def safe_name(text: str) -> str:
    """'a C' / 'the C shape' / 'C!' -> 'c' ; 'smiley face' -> 'smiley-face'."""
    t = re.sub(r"^(a|an|the|letter|shape)\s+", "", text.strip().lower())
    t = re.sub(r"\s+(shape|letter|again|please)$", "", t)
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40]
MAX_RECORD_S = 60.0
HZ = 20.0
STILL_TOL = 0.005      # norm; a sample this close to its neighbour counts as not moving


# -- pure helpers (tested offline) -------------------------------------------------------
def trim_still(samples: list[tuple[float, dict]], tol: float = STILL_TOL) -> list[tuple[float, dict]]:
    """Drop the motionless lead-in and tail, keeping one still sample at each end."""
    def moved(a, b):
        return any(abs(a[j] - b.get(j, a[j])) > tol for j in a)
    if len(samples) < 2:
        return samples
    first = samples[0][1]
    start = next((i for i, (_, p) in enumerate(samples) if moved(p, first)), None)
    if start is None:
        return []
    last = samples[-1][1]
    end = next(i for i in range(len(samples) - 1, -1, -1) if moved(samples[i][1], last))
    s, e = max(0, start - 1), min(len(samples), end + 2)
    t0 = samples[s][0]
    out = [(t - t0, p) for t, p in samples[s:e]]
    out[0] = (0.0, dict(first))          # start exactly where the arm was resting
    out[-1] = (out[-1][0], dict(last))   # finish exactly where the arm came to rest
    return out


def thin(samples: list[tuple[float, dict]], hz: float = HZ) -> list[tuple[float, dict]]:
    """At most one sample per 1/hz s; the last sample is always kept."""
    out, nxt = [], 0.0
    for t, p in samples:
        if t >= nxt:
            out.append((t, p))
            nxt = t + 1.0 / hz
    if samples and out[-1] is not samples[-1]:
        out.append(samples[-1])
    return out


def to_waypoints(samples: list[tuple[float, dict]], joints: list[str]) -> list[dict]:
    return [{"t": round(t, 3), "joints": {j: round(p[j], 4) for j in joints if j in p}} for t, p in samples]


def travel(samples: list[tuple[float, dict]]) -> dict[str, float]:
    """Largest minus smallest position of each joint over the recording."""
    if not samples:
        return {}
    return {j: round(max(p[j] for _, p in samples) - min(p[j] for _, p in samples), 3) for j in samples[0][1]}


# -- the recorder -----------------------------------------------------------------------
class Recorder:
    def __init__(self, io, arm: Arm, replay_joints: list[str]):
        self.io, self.arm = io, arm
        self.replay_joints = replay_joints     # e.g. no 'gripper' on the leader (it's a trigger)
        self.start_ns: int | None = None
        self.start_wall = 0.0
        self.pending_name: str | None = None
        self.mirrored_from: str | None = None
        self.latest: dict | None = self._load_latest()

    @property
    def recording(self) -> bool:
        return self.start_ns is not None

    def _load_latest(self) -> dict | None:
        p = REC_DIR / "latest.json"
        try:
            rec = json.loads(p.read_text())
            return rec if rec.get("bus") == getattr(self.io, "follower", None) else None
        except (OSError, ValueError):
            return None

    # -- named recordings -------------------------------------------------------------
    def _bus(self):
        return getattr(self.io, "follower", None)

    def names(self) -> list[str]:
        out = []
        for p in sorted(named_dir().glob("*.json")):
            try:
                if json.loads(p.read_text()).get("bus") == self._bus():
                    out.append(p.stem)
            except (OSError, ValueError):
                pass
        return out

    def load(self, name: str) -> dict | None:
        try:
            rec = json.loads((named_dir() / f"{safe_name(name)}.json").read_text())
            return rec if rec.get("bus") == self._bus() else None
        except (OSError, ValueError):
            return None

    def save_as(self, name: str) -> str:
        n = safe_name(name)
        if not n:
            return "Give it a name, e.g. 'save as c'."
        if not self.latest:
            return "Nothing recorded yet."
        named_dir().mkdir(parents=True, exist_ok=True)
        (named_dir() / f"{n}.json").write_text(json.dumps({**self.latest, "name": n}, indent=1))
        return f"Saved the last recording as '{n}'. Type 'draw {n}' to play it with the pen."

    async def _mirrored(self) -> str | None:
        f = getattr(self.io, "mirrored_from", None)
        return await f() if f else None

    async def start(self, name: str | None = None) -> str:
        leader = await self._mirrored()
        if not leader:
            joints = await self.io.read_joints()
            stiff = [n for n, m in joints.items() if m.get("torque_on")]
            if stiff:
                return ("Not recording: torque is on (" + ", ".join(stiff) + "), so the arm can't be moved "
                        "by hand. Either type 'relax' first (the arm goes limp: support it), or turn on teleop "
                        "in Station (leader -> follower) and draw with the leader.")
        self.pending_name = safe_name(name) if name else None
        self.mirrored_from = leader
        self.start_ns = await self.io.stamp_ns()
        self.start_wall = time.monotonic()
        how = (f"Move the LEADER arm ({leader}); the follower copies it and is what gets recorded"
               if leader else "Move the arm by hand")
        label = f" '{self.pending_name}'" if self.pending_name else ""
        return f"● Recording{label}. {how}, then type 'done' (max {MAX_RECORD_S:g} s)."

    async def finish(self) -> str:
        if not self.recording:
            return "Not recording. Type 'record' first."
        since, self.start_ns = self.start_ns, None
        took = time.monotonic() - self.start_wall
        try:
            raw = await self.io.history_since(since)
        finally:   # a long history read can leave the connection clogged: start the replay on a fresh one
            if hasattr(self.io, "reconnect"):
                self.io.reconnect()
        raw = [(t, p) for t, p in raw if t <= MAX_RECORD_S]
        samples = thin(trim_still(raw))
        if len(samples) < 2:
            return f"Nothing recorded: no joint moved more than {STILL_TOL} in {took:.0f} s."
        rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "bus": self._bus(), "via_leader": self.mirrored_from,
               "duration_s": round(samples[-1][0], 2), "samples": len(samples), "raw_samples": len(raw),
               "travel": travel(samples), "waypoints": to_waypoints(samples, list(samples[0][1]))}
        REC_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + ".json"
        for p in (REC_DIR / name, REC_DIR / "latest.json"):
            p.write_text(json.dumps(rec, indent=1))
        self.latest = rec
        moved = {j: v for j, v in rec["travel"].items() if v > STILL_TOL and j in self.replay_joints}
        msg = (f"■ Recorded {rec['duration_s']:.1f} s ({rec['samples']} samples), saved logs/recordings/{name}. "
               f"Joints moved (norm): " + ", ".join(f"{j} {v:.2f}" for j, v in moved.items()))
        if self.pending_name:
            msg += " " + self.save_as(self.pending_name)
            self.pending_name = None
        if self.mirrored_from:
            msg += (" Before replaying, turn teleop OFF in Station; the follower goes limp when you do, "
                    "so hold it or rest the brush on the paper first.")
        return msg

    def plan(self, speed: float, rec: dict | None = None, pen: bool = False) -> tuple[list[dict], list[str]]:
        rec = rec or self.latest
        keep = [j for j in self.replay_joints if not (pen and j == "gripper")]   # pen mode: never let go of the brush
        wps = [{"t": w["t"], "joints": {j: v for j, v in w["joints"].items() if j in keep}}
               for w in rec["waypoints"]]
        safe, clamps = check_waypoints(self.arm, wps)
        notes = sorted({c.split(" ")[0] for c in clamps})
        return safe, ([f"clamped to limits: {', '.join(notes)}"] if notes else [])

    def pen_moves(self, wps: list[dict]) -> tuple[list[dict], list[dict]]:
        """Pen mode: before = go to the first pose with the elbow lifted, then lower onto it;
        after = lift the elbow again. Keeps the brush off the paper between strokes."""
        lift = PEN_LIFT_DEG / self.arm.deg_per_norm("elbow")
        first, last = wps[0]["joints"], wps[-1]["joints"]
        before, _ = check_waypoints(self.arm, [{"joints": {**first, "elbow": first["elbow"] - lift}, "duration_s": 0.0},
                                               {"joints": dict(first), "duration_s": PEN_DESCEND_S}])
        after, _ = check_waypoints(self.arm, [{"joints": {**last, "elbow": last["elbow"] - lift}, "duration_s": 0.8}])
        return before, after

    async def replay(self, speed: float = 0.5, name: str | None = None, pen: bool = False) -> dict:
        """Play the latest recording, or a named one. pen=True: lift-approach-lower first, lift after,
        gripper left alone. Returns a result shaped like Skills.run()."""
        rec = self.load(name) if name else self.latest
        if not rec:
            what = f"no saved recording called '{safe_name(name)}'" if name else "nothing recorded yet"
            have = self.names()
            return {"reached": False, "abort_reason": what + (f" (saved: {', '.join(have)})" if have else ""),
                    "errors": {}, "clamps": []}
        leader = await self._mirrored()
        if leader:
            return {"reached": False, "errors": {}, "clamps": [],
                    "abort_reason": f"teleop is still on (leader {leader} drives the follower): turn it off in "
                                    "Station first; the follower goes limp when you do, so hold it"}
        self.arm.update_ranges(await self.io.read_joints())
        wps, clamps = self.plan(speed, rec, pen)
        if pen and "elbow" in wps[0]["joints"]:
            before, after = self.pen_moves(wps)
            res = await self.io.play(before, 0.5)
            if not res.get("completed"):
                return {"reached": False, "abort_reason": f"approach: {res.get('abort_reason')}", "errors": {},
                        "clamps": clamps}
        else:
            after = None
        peak: dict[str, int] = {}
        sampling = True

        async def sample_current():   # effort along the path, ~5 Hz
            while sampling:
                try:
                    for n, m in (await self.io.read_joints()).items():
                        peak[n] = max(peak.get(n, 0), abs(int(m.get("current") or 0)))
                except Exception:
                    pass
                await asyncio.sleep(0.2)
        sampler = asyncio.create_task(sample_current())
        try:
            res = await self.io.play(wps, speed)
            if after and res.get("completed"):
                await self.io.play(after, 0.5)
        finally:
            sampling = False
            await sampler
        after_pos = {n: m["position_norm"] for n, m in (await self.io.read_joints()).items()}
        end = (after or wps)[-1]["joints"]
        errors = {j: round(abs(after_pos[j] - v), 3) for j, v in end.items() if j in after_pos}
        tol = 0.03
        return {"reached": bool(res.get("completed")) and all(e <= tol for e in errors.values()),
                "abort_reason": res.get("abort_reason"), "errors": errors, "clamps": clamps,
                "elapsed_s": res.get("elapsed_s"), "planned_s": res.get("planned_s"),
                "stretched": bool(res.get("stretched_segments")),
                "peak_current": {j: peak[j] for j in self.replay_joints if j in peak},
                "max_tracking_error": res.get("max_tracking_error")}
