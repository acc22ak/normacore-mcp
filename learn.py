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
