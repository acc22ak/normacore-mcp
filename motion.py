"""Pure motion logic for the fast playback engine: no I/O, so it is unit-testable.

server.py runs the loop (send goals, read state ~20x/s); this module decides
*what* to command and *whether* it is still safe.

Positions are normalized 0..1 per joint (same convention as server.to_norm).
Joint ids are ints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Servo status bits (norma-core st3215/src/protocol/error.rs): bit0 Voltage, bit1 AngleLimit,
# bit2 Overheat, bit3 Range, bit4 Checksum, bit5 Overload. Station's own mirroring and
# calibration ignore the voltage bit (motors-mirroring mirror.rs MIRRORING_IGNORED_BITS,
# st3215 port_meta.rs), so we do too; anything else aborts.
IGNORED_STATUS_BITS = 0x01


@dataclass
class Point:
    t: float                    # seconds from start
    joints: dict[int, float]    # full pose (every joint)
    ease: bool = False          # smoothstep the segment that ENDS at this point


@dataclass
class Timeline:
    points: list[Point]
    stretched_segments: int = 0
    clamped: dict[int, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.points[-1].t if self.points else 0.0

    def sample(self, t: float) -> dict[int, float]:
        pts = self.points
        if t <= pts[0].t:
            return dict(pts[0].joints)
        for a, b in zip(pts, pts[1:]):
            if t <= b.t:
                span = b.t - a.t
                u = 1.0 if span <= 0 else (t - a.t) / span
                if b.ease:
                    u = u * u * (3 - 2 * u)  # smoothstep: zero velocity at both ends
                return {j: a.joints[j] + (b.joints[j] - a.joints[j]) * u for j in b.joints}
        return dict(pts[-1].joints)

    def peak_velocity(self) -> dict[int, float]:
        """Max |velocity| per joint in norm/s (smoothstep peaks at 1.5x the average)."""
        peak: dict[int, float] = {}
        for a, b in zip(self.points, self.points[1:]):
            dt = b.t - a.t
            if dt <= 0:
                continue
            k = 1.5 if b.ease else 1.0
            for j in b.joints:
                v = k * abs(b.joints[j] - a.joints[j]) / dt
                peak[j] = max(peak.get(j, 0.0), v)
        return peak


def _key(j) -> int:
    return int(j)


def parse_waypoints(waypoints: list[dict]) -> tuple[list[tuple[float | None, dict[int, float], bool]], str]:
    """Accept either dense samples [{"t": s, "joints": {...}}] or keyframes
    [{"joints": {...}, "duration_s": s}]. Returns ([(t_or_None, partial_joints, ease)], kind)."""
    if not waypoints:
        raise ValueError("no waypoints")
    dense = all("t" in w for w in waypoints)
    keyed = all("t" not in w for w in waypoints)
    if not (dense or keyed):
        raise ValueError("mix of dense ('t') and keyframe ('duration_s') waypoints")
    out = []
    t = 0.0
    last_t = -1.0
    for w in waypoints:
        joints = {_key(j): float(v) for j, v in (w.get("joints") or {}).items()}
        if dense:
            wt = float(w["t"])
            if wt < last_t:
                raise ValueError("dense waypoints must have non-decreasing 't'")
            last_t = wt
            out.append((wt, joints, False))
        else:
            d = float(w.get("duration_s", 1.0))
            if d < 0:
                raise ValueError("duration_s must be >= 0")
            t += d
            out.append((t, joints, True))
    return out, ("dense" if dense else "keyframes")


def build_timeline(current: dict[int, float], waypoints: list[dict], *, speed_scale: float = 1.0,
                   max_vel: float = 1.0, limits: dict[int, tuple[float, float]] | None = None,
                   approach_vel: float = 0.3) -> Timeline:
    """Full-pose timeline starting at `current`.

    - joints missing from a waypoint keep their previous value;
    - targets are clamped to [0, 1] and to optional soft `limits`;
    - times are divided by `speed_scale`;
    - any segment that would exceed `max_vel` (norm/s, per joint, eased peak included)
      is stretched in time instead (reported in `stretched_segments`);
    - a smooth approach from `current` to the first waypoint is prepended, at <= approach_vel.
    """
    if speed_scale <= 0:
        raise ValueError("speed_scale must be > 0")
    parsed, kind = parse_waypoints(waypoints)
    limits = limits or {}
    clamped: dict[int, str] = {}

    pose = dict(current)
    raw_pts: list[Point] = []
    for t, joints, ease in parsed:
        for j, v in joints.items():
            if j not in pose:
                raise ValueError(f"joint {j} is not on this bus")
            lo, hi = limits.get(j, (0.0, 1.0))
            c = min(max(v, lo, 0.0), hi, 1.0)
            if abs(c - v) > 1e-9:
                clamped[j] = f"{v:.3f} -> {c:.3f} (limits [{lo}, {hi}])"
            pose[j] = c
        raw_pts.append(Point(t / speed_scale, dict(pose), ease))

    pts = [Point(0.0, dict(current))]
    if kind == "dense":
        # Recorded motion: first glide (eased, <= approach_vel) to where the recording starts,
        # then play the samples with their original spacing.
        first = raw_pts[0]
        dist = max((abs(first.joints[j] - current[j]) for j in current), default=0.0)
        approach = (1.5 * dist / approach_vel) if dist > 1e-6 else 0.0   # eased -> 1.5x peak
        pts.append(Point(approach, first.joints, ease=True))
        for p in raw_pts[1:]:
            pts.append(Point(approach + (p.t - first.t), p.joints, p.ease))
    else:
        # Keyframes: each duration_s already covers the move from the previous pose
        # (the first one from the current pose); max_vel below stretches it if too fast.
        pts.extend(raw_pts)

    # Enforce max velocity by stretching segments.
    stretched = 0
    fixed = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        dt = b.t - a.t
        k = 1.5 if b.ease else 1.0
        need = max((k * abs(b.joints[j] - a.joints[j]) / max_vel for j in b.joints), default=0.0)
        if need > dt + 1e-9:
            dt = need
            stretched += 1
        fixed.append(Point(fixed[-1].t + max(dt, 0.0), b.joints, b.ease))
    return Timeline(fixed, stretched, clamped)


@dataclass
class SafetyConfig:
    current_limit: int = 100                     # Station's default current_threshold (motors-mirroring config.rs)
    per_motor_current: dict[int, int] = field(default_factory=dict)
    gripper_id: int = 8
    track_err_max: float = 0.10                  # norm; untuned default, measure on the arm
    track_err_ticks: int = 5                     # consecutive ticks over the limit before aborting
    stale_s: float = 0.5                         # no new state for this long -> abort


@dataclass
class SafetyMonitor:
    cfg: SafetyConfig
    over_count: dict[int, int] = field(default_factory=dict)
    gripper_blocked: bool = False
    max_track_err: dict[int, float] = field(default_factory=dict)

    def limit_for(self, j: int) -> int:
        return self.cfg.per_motor_current.get(j, self.cfg.current_limit)

    def check(self, motors: dict[int, dict], commanded: dict[int, float],
              state_age_s: float) -> str | None:
        """Return an abort reason, or None if it's safe to continue.

        `motors`: server.read_buses()[bus]["motors"]; `commanded`: last goals sent (norm).
        Gripper over-current is a GRASP, not a crash: it sets gripper_blocked instead of aborting.
        """
        if state_age_s > self.cfg.stale_s:
            return f"robot state is {state_age_s:.2f} s old (limit {self.cfg.stale_s} s); not moving blind"
        for j, m in motors.items():
            bits = m["error_status"] & ~IGNORED_STATUS_BITS
            if bits:
                return f"servo {j} ({m['joint']}) reports error status 0x{m['error_status']:02X}"
            if m["current"] >= self.limit_for(j):
                if j == self.cfg.gripper_id:
                    self.gripper_blocked = True
                else:
                    return (f"servo {j} ({m['joint']}) current {m['current']} >= limit {self.limit_for(j)}: "
                            "probably hit something")
            if j in commanded and m["position_norm"] is not None:
                err = abs(m["position_norm"] - commanded[j])
                self.max_track_err[j] = max(self.max_track_err.get(j, 0.0), err)
                if j == self.cfg.gripper_id and self.gripper_blocked:
                    continue  # a blocked gripper is expected to lag its target
                if err > self.cfg.track_err_max:
                    self.over_count[j] = self.over_count.get(j, 0) + 1
                    if self.over_count[j] >= self.cfg.track_err_ticks:
                        return (f"servo {j} ({m['joint']}) is {err:.3f} away from its target for "
                                f"{self.over_count[j]} ticks: blocked or too fast")
                else:
                    self.over_count[j] = 0
        return None


def steps_per_s(norm_per_s: float, range_steps: int) -> int:
    """Servo goal-speed register value for a joint, clamped to Station teleop's own bounds
    (min_speed 300, max_speed 3300 in motors-mirroring config.rs)."""
    return int(min(3300, max(300, math.ceil(norm_per_s * range_steps * 1.5))))
