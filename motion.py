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


# Moves that start and end at rest use minimum-jerk easing (10u^3 - 15u^4 + 6u^5): speed
# AND acceleration are zero at both ends, which reads as gentle and human. Its peak speed is
# 1.875x the average (smoothstep was 1.5x); callers sizing durations should use this.
EASED_PEAK = 1.875


def min_jerk(u: float) -> float:
    return u * u * u * (10 + u * (-15 + 6 * u))


def min_jerk_rate(u: float) -> float:
    """d/du of min_jerk."""
    return 30 * u * u * (1 - u) * (1 - u)


@dataclass
class Point:
    t: float                    # seconds from start
    joints: dict[int, float]    # full pose (every joint)
    stop: bool = False          # come to rest here (zero velocity for every joint)
    vel: dict[int, float] = field(default_factory=dict)  # norm/s at this point, set by build


def _segment_value(a: Point, b: Point, j: int, s: float) -> float:
    p0, p1 = a.joints[j], b.joints[j]
    v0, v1 = a.vel.get(j, 0.0), b.vel.get(j, 0.0)
    if v0 == 0.0 and v1 == 0.0:
        return p0 + (p1 - p0) * min_jerk(s)
    dt = b.t - a.t
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * dt * v0
            + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * dt * v1)


def _segment_rate(a: Point, b: Point, j: int, s: float) -> float:
    """Velocity (norm/s) at fraction s of the segment."""
    dt = b.t - a.t
    if dt <= 0:
        return 0.0
    p0, p1 = a.joints[j], b.joints[j]
    v0, v1 = a.vel.get(j, 0.0), b.vel.get(j, 0.0)
    if v0 == 0.0 and v1 == 0.0:
        return (p1 - p0) * min_jerk_rate(s) / dt
    s2 = s * s
    return ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * dt * v0
            + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * dt * v1) / dt


def _segment_peak(a: Point, b: Point, j: int) -> float:
    dt = b.t - a.t
    if dt <= 0:
        # Zero time but a real change is an instant jump: infinitely fast, so it gets stretched.
        return math.inf if abs(b.joints[j] - a.joints[j]) > 1e-9 else 0.0
    if a.vel.get(j, 0.0) == 0.0 and b.vel.get(j, 0.0) == 0.0:
        return EASED_PEAK * abs(b.joints[j] - a.joints[j]) / dt
    return max(abs(_segment_rate(a, b, j, k / 16)) for k in range(17))


def _assign_velocities(pts: list[Point]) -> None:
    """Monotone cubic (PCHIP / Fritsch-Butland) tangents: the path flows through keyframes
    without overshooting any of them, so it never leaves the range the keyframes span.
    Rest at the first and last point, at `stop` points, and wherever a joint turns around."""
    n = len(pts)
    for i, p in enumerate(pts):
        p.vel = {}
        if i == 0 or i == n - 1 or p.stop:
            continue
        a, b = pts[i - 1], pts[i + 1]
        h0, h1 = p.t - a.t, b.t - p.t
        if h0 <= 0 or h1 <= 0:
            continue
        for j in p.joints:
            d0 = (p.joints[j] - a.joints[j]) / h0
            d1 = (b.joints[j] - p.joints[j]) / h1
            if d0 * d1 <= 0:
                continue  # reversal or a joint that holds still: rest here
            w0, w1 = 2 * h1 + h0, h1 + 2 * h0
            p.vel[j] = (w0 + w1) / (w0 / d0 + w1 / d1)


@dataclass
class Timeline:
    points: list[Point]
    stretched_segments: int = 0
    clamped: dict[int, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.points[-1].t if self.points else 0.0

    def _segment(self, t: float) -> tuple[Point, Point, float] | None:
        pts = self.points
        for a, b in zip(pts, pts[1:]):
            if t <= b.t:
                span = b.t - a.t
                return a, b, (1.0 if span <= 0 else max(0.0, (t - a.t) / span))
        return None

    def sample(self, t: float) -> dict[int, float]:
        pts = self.points
        if t <= pts[0].t:
            return dict(pts[0].joints)
        seg = self._segment(t)
        if seg is None:
            return dict(pts[-1].joints)
        a, b, s = seg
        return {j: _segment_value(a, b, j, s) for j in b.joints}

    def velocity(self, t: float) -> dict[int, float]:
        seg = self._segment(t) if self.points and t > self.points[0].t else None
        if seg is None:
            return {j: 0.0 for j in self.points[-1].joints} if self.points else {}
        a, b, s = seg
        return {j: _segment_rate(a, b, j, s) for j in b.joints}

    def peak_velocity(self) -> dict[int, float]:
        """Max |velocity| per joint in norm/s over the whole timeline."""
        peak: dict[int, float] = {}
        for a, b in zip(self.points, self.points[1:]):
            for j in b.joints:
                peak[j] = max(peak.get(j, 0.0), _segment_peak(a, b, j))
        return peak


def _key(j) -> int:
    return int(j)


def parse_waypoints(waypoints: list[dict]) -> tuple[list[tuple[float | None, dict[int, float], bool]], str]:
    """Accept either dense samples [{"t": s, "joints": {...}}] or keyframes
    [{"joints": {...}, "duration_s": s, "stop": bool}]. Returns ([(t, partial_joints, stop)], kind)."""
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
        stop = bool(w.get("stop", False))
        if dense:
            wt = float(w["t"])
            if wt < last_t:
                raise ValueError("dense waypoints must have non-decreasing 't'")
            last_t = wt
            out.append((wt, joints, stop))
        else:
            d = float(w.get("duration_s", 1.0))
            if d < 0:
                raise ValueError("duration_s must be >= 0")
            t += d
            out.append((t, joints, stop))
    return out, ("dense" if dense else "keyframes")


def build_timeline(current: dict[int, float], waypoints: list[dict], *, speed_scale: float = 1.0,
                   max_vel: float = 1.0, limits: dict[int, tuple[float, float]] | None = None,
                   approach_vel: float = 0.3) -> Timeline:
    """Full-pose timeline starting at `current`, smooth enough to look gentle.

    - joints missing from a waypoint keep their previous value;
    - targets are clamped to [0, 1] and to optional soft `limits`;
    - times are divided by `speed_scale`;
    - starts and ends at rest; flows through intermediate waypoints (monotone cubic, no
      overshoot) except where a joint turns around or a keyframe has "stop": true;
      stop-to-stop moves use minimum-jerk easing;
    - any segment that would exceed `max_vel` (norm/s, per joint, true peak) is stretched
      in time instead (reported in `stretched_segments`);
    - recordings get a gentle approach from `current` to their first sample, at <= approach_vel.
    """
    if speed_scale <= 0:
        raise ValueError("speed_scale must be > 0")
    parsed, kind = parse_waypoints(waypoints)
    limits = limits or {}
    clamped: dict[int, str] = {}

    pose = dict(current)
    raw_pts: list[Point] = []
    for t, joints, stop in parsed:
        for j, v in joints.items():
            if j not in pose:
                raise ValueError(f"joint {j} is not on this bus")
            lo, hi = limits.get(j, (0.0, 1.0))
            c = min(max(v, lo, 0.0), hi, 1.0)
            if abs(c - v) > 1e-9:
                clamped[j] = f"{v:.3f} -> {c:.3f} (limits [{lo}, {hi}])"
            pose[j] = c
        raw_pts.append(Point(t / speed_scale, dict(pose), stop))

    pts = [Point(0.0, dict(current), stop=True)]
    if kind == "dense":
        # Recorded motion: first glide (at rest at both ends, <= approach_vel) to where the
        # recording starts, then play the samples with their original spacing.
        first = raw_pts[0]
        dist = max((abs(first.joints[j] - current[j]) for j in current), default=0.0)
        approach = (EASED_PEAK * dist / approach_vel) if dist > 1e-6 else 0.0
        if approach > 0:
            pts.append(Point(approach, first.joints, stop=True))
        else:
            pts[0] = Point(0.0, dict(first.joints), stop=True)
        for p in raw_pts[1:]:
            pts.append(Point(approach + (p.t - first.t), p.joints, p.stop))
    else:
        # Keyframes: each duration_s already covers the move from the previous pose
        # (the first one from the current pose); max_vel below stretches it if too fast.
        pts.extend(raw_pts)

    # Drop zero-length duplicates so tangents stay finite.
    merged = [pts[0]]
    for p in pts[1:]:
        if p.t - merged[-1].t <= 1e-9 and p.joints == merged[-1].joints:
            merged[-1].stop = merged[-1].stop or p.stop
            continue
        merged.append(p)
    pts = merged

    # Enforce max velocity by stretching segments; tangents depend on timing, so iterate.
    stretched_idx: set[int] = set()
    for _ in range(8):
        _assign_velocities(pts)
        changed = False
        new_t = [pts[0].t]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            dt = b.t - a.t
            peak = max((_segment_peak(a, b, j) for j in b.joints), default=0.0)
            if peak > max_vel * (1 + 1e-6):
                dt = dt * peak / max_vel if dt > 0 and math.isfinite(peak) else EASED_PEAK * max(
                    (abs(b.joints[j] - a.joints[j]) for j in b.joints), default=0.0) / max_vel
                stretched_idx.add(i)
                changed = True
            new_t.append(new_t[-1] + max(dt, 0.0))
        for p, t in zip(pts, new_t):
            p.t = t
        if not changed:
            break
    else:
        # Did not settle: fall back to stop-to-stop minimum-jerk segments, whose peak is
        # exact, so the speed cap is guaranteed.
        for p in pts:
            p.stop = True
        new_t = [pts[0].t]
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            need = EASED_PEAK * max((abs(b.joints[j] - a.joints[j]) for j in b.joints), default=0.0) / max_vel
            dt = max(b.t - a.t, need)
            if dt > b.t - a.t + 1e-9:
                stretched_idx.add(i)
            new_t.append(new_t[-1] + dt)
        for p, t in zip(pts, new_t):
            p.t = t
    _assign_velocities(pts)
    return Timeline(pts, len(stretched_idx), clamped)


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
