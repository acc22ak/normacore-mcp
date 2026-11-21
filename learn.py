"""Pure learning logic: turning recorded robot history into skills, and the practice ladder.

No I/O here (server.py reads Station and writes files), so everything is unit-testable.
A "sample" is (t_seconds, {joint_id: norm}, {joint_id: raw}).
"""

from __future__ import annotations

# Practice ladder: a new skill starts slow; 2 successes in a row -> next level;
# a failure -> one level down. (Team decision: 2 successes, not 3.)
SPEED_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SUCCESSES_TO_LEVEL_UP = 2


def encode_shift(n: int) -> bytes:
    """Little-endian id bytes for a read offset, width 1/2/4/8.

    Station ids are little-endian (norma-core dataset-generator bounds.go uses uintn.FromLEBytes;
    Rust side reads them with UintN::read_value_from_slice). Request-side width is inferred, so the
    recorder reports the time span it actually got back.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    for width in (1, 2, 4, 8):
        if n < 1 << (8 * width):
            return n.to_bytes(width, "little")
    raise ValueError("n too large")


def estimate_rate_hz(stamps_ns: list[int]) -> float | None:
    """Updates per second from consecutive (sorted) station stamps; None if unknown."""
    s = sorted(set(stamps_ns))
    if len(s) < 2 or s[-1] == s[0]:
        return None
    return (len(s) - 1) / ((s[-1] - s[0]) / 1e9)


def _moved(a: dict[int, float], b: dict[int, float], tol: float) -> bool:
    return any(abs(a[j] - b.get(j, a[j])) > tol for j in a)


def trim_idle(samples: list[tuple], tol: float = 0.01, pad_s: float = 0.3) -> list[tuple]:
    """Drop the still part before the first movement and after the last one (keeps pad_s)."""
    if len(samples) < 2:
        return samples
    first_pose, last_pose = samples[0][1], samples[-1][1]
    start = next((i for i, s in enumerate(samples) if _moved(s[1], first_pose, tol)), None)
    if start is None:
        return []  # nothing moved
    end = next(i for i in range(len(samples) - 1, -1, -1) if _moved(samples[i][1], last_pose, tol))
    t0 = samples[start][0] - pad_s
    t1 = samples[end][0] + pad_s
    kept = [s for s in samples if t0 <= s[0] <= t1]
    base = kept[0][0]
    return [(t - base, n, r) for t, n, r in kept]


def downsample(samples: list[tuple], hz: float) -> list[tuple]:
    """Keep at most one sample per 1/hz seconds (always keeps the first and last)."""
    if not samples:
        return []
    out, step, next_t = [samples[0]], 1.0 / hz, samples[0][0] + 1.0 / hz
    for s in samples[1:-1]:
        if s[0] >= next_t - 1e-9:
            out.append(s)
            next_t = s[0] + step
    if len(samples) > 1:
        out.append(samples[-1])
    return out


def extract_keyframes(samples: list[tuple], gripper_id: int, still_tol: float = 0.005,
                      still_s: float = 0.3, grip_tol: float = 0.02) -> list[dict]:
    """Human-readable summary of a recording: start, end, pauses (all joints still for
    >= still_s), and gripper open/close turning points. Each keyframe: {t, joints, label}."""
    if not samples:
        return []
    keys = {0: "start", len(samples) - 1: "end"}
    # pauses
    i = 0
    while i < len(samples):
        j = i
        while j + 1 < len(samples) and not _moved(samples[j + 1][1], samples[i][1], still_tol):
            j += 1
        if samples[j][0] - samples[i][0] >= still_s and 0 < i:
            keys.setdefault(i, "pause")
        i = j + 1
    # gripper direction changes
    g = [s[1].get(gripper_id) for s in samples]
    if all(v is not None for v in g):
        last_dir, anchor = 0, 0
        for k in range(1, len(g)):
            d = g[k] - g[anchor]
            if abs(d) < grip_tol:
                continue
            direction = 1 if d > 0 else -1
            if last_dir and direction != last_dir:
                keys.setdefault(anchor, "gripper turns")
            last_dir, anchor = direction, k
    return [{"t": round(samples[k][0], 3), "joints": {str(j): round(v, 4) for j, v in samples[k][1].items()},
             "label": label} for k, label in sorted(keys.items())]


# ----------------------------------------------------------------------------
# self-learning: turn Claude's own slow, successful attempt into a fast skill.
# While Claude explores it moves in steps and stops to think (the arm is still for
# seconds), so the still points are the poses it chose. Replaying only those poses,
# merged into straight moves, removes the thinking time and the step-by-step crawl.

def pause_keyframes(samples: list[tuple], still_tol: float = 0.005, still_s: float = 0.3) -> list[tuple]:
    """(norm, raw) poses where the arm rested >= still_s, plus the first and last pose."""
    if not samples:
        return []
    out = [(samples[0][1], samples[0][2])]
    i = 0
    while i < len(samples):
        j = i
        while j + 1 < len(samples) and not _moved(samples[j + 1][1], samples[i][1], still_tol):
            j += 1
        if samples[j][0] - samples[i][0] >= still_s:
            out.append((samples[j][1], samples[j][2]))
        i = j + 1
    out.append((samples[-1][1], samples[-1][2]))
    # drop consecutive duplicates (same rest pose reported twice)
    dedup = [out[0]]
    for p in out[1:]:
        if _moved(p[0], dedup[-1][0], still_tol):
            dedup.append(p)
    if len(dedup) == 1 and len(out) > 1:
        dedup.append(out[-1])
    return dedup


def has_motion(poses: list[tuple], tol: float = 0.005) -> bool:
    """True if any consecutive (norm, raw) poses differ by more than tol on some joint."""
    return any(_moved(a[0], b[0], tol) for a, b in zip(poses, poses[1:]))


def _dist_to_segment(p: dict, a: dict, b: dict) -> float:
    """Euclidean distance (joint space) from pose p to the straight move a->b."""
    keys = list(a)
    ab = [b[k] - a[k] for k in keys]
    ap = [p[k] - a[k] for k in keys]
    L2 = sum(x * x for x in ab)
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / L2))
    return sum((ap[i] - u * ab[i]) ** 2 for i in range(len(keys))) ** 0.5


def _rdp(points: list[tuple], tol: float) -> list[int]:
    """Indices kept by Ramer-Douglas-Peucker on [(norm, raw)]."""
    if len(points) <= 2:
        return list(range(len(points)))
    a, b = points[0][0], points[-1][0]
    k, dmax = 0, -1.0
    for i in range(1, len(points) - 1):
        d = _dist_to_segment(points[i][0], a, b)
        if d > dmax:
            k, dmax = i, d
    if dmax <= tol:
        return [0, len(points) - 1]
    left = _rdp(points[:k + 1], tol)
    right = _rdp(points[k:], tol)
    return left[:-1] + [k + i for i in right]


def simplify_path(points: list[tuple], gripper_id: int, tol: float = 0.02, grip_tol: float = 0.02) -> list[tuple]:
    """Merge rest poses that lie (within tol) on a straight line between their neighbours.

    Poses where the gripper starts or stops changing are always kept, so grasping/releasing
    never gets blended into an arm move.
    """
    if len(points) <= 2:
        return points
    g = [p[0].get(gripper_id) for p in points]
    forced = {0, len(points) - 1}
    if all(v is not None for v in g):
        for k in range(1, len(points) - 1):
            prev_chg = abs(g[k] - g[k - 1]) > grip_tol
            next_chg = abs(g[k + 1] - g[k]) > grip_tol
            if prev_chg != next_chg:  # gripper starts or stops moving here
                forced.add(k)
    cuts = sorted(forced)
    keep: list[int] = []
    for s, e in zip(cuts, cuts[1:]):
        idx = _rdp(points[s:e + 1], tol)
        keep.extend(s + i for i in (idx if not keep else idx[1:]))
    return [points[i] for i in keep]


def timed_keyframes(points: list[tuple], base_vel: float, min_s: float = 0.2) -> list[dict]:
    """Keyframes for play_motion: each move lasts long enough to keep the eased peak speed
    (1.5x average) at base_vel at speed_scale 1.0. The first pose is the start, so it's skipped."""
    out = []
    for a, b in zip(points, points[1:]):
        dist = max(abs(b[0][j] - a[0][j]) for j in b[0])
        out.append({"joints": {str(j): round(v, 4) for j, v in b[0].items()},
                    "raw": {str(j): int(v) for j, v in b[1].items()},
                    "duration_s": round(max(min_s, 1.5 * dist / base_vel), 3)})
    return out


def ranges_changed(saved: dict[str, list[int]], current: dict[int, tuple[int, int]], tol: int = 20) -> dict[int, str]:
    """Joints whose calibration range moved by more than tol steps since the skill was recorded."""
    out = {}
    for k, (smin, smax) in saved.items():
        j = int(k)
        if j not in current:
            out[j] = "joint missing now"
            continue
        cmin, cmax = current[j]
        if abs(cmin - smin) > tol or abs(cmax - smax) > tol:
            out[j] = f"[{smin}, {smax}] -> [{cmin}, {cmax}]"
    return out


def ladder_update(practice: dict, success: bool) -> dict:
    """Next practice state. practice = {"level": index into SPEED_LEVELS, "streak": successes in a row}."""
    level = int(practice.get("level", 0))
    streak = int(practice.get("streak", 0))
    if success:
        streak += 1
        if streak >= SUCCESSES_TO_LEVEL_UP and level < len(SPEED_LEVELS) - 1:
            level, streak = level + 1, 0
    else:
        level, streak = max(0, level - 1), 0
    return {"level": level, "streak": streak, "speed_scale": SPEED_LEVELS[level]}
