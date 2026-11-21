"""Unit tests for learn.py (pure logic, no station needed).

usage: python tests/test_learn.py <normacore-mcp dir>
"""
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import learn as L

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# --- offset encoding ---
check("shift 0 -> 1 byte", L.encode_shift(0) == b"\x00")
check("shift 255 -> 1 byte (same as old bytes([n]))", L.encode_shift(255) == bytes([255]))
check("shift 256 -> 2 bytes little-endian", L.encode_shift(256) == b"\x00\x01")
check("shift 1200 -> 2 bytes LE", L.encode_shift(1200) == (1200).to_bytes(2, "little"))
check("shift 70000 -> 4 bytes LE", L.encode_shift(70000) == (70000).to_bytes(4, "little"))

# --- rate estimate ---
check("rate from stamps", abs(L.estimate_rate_hz([i * 50_000_000 for i in range(21)]) - 20.0) < 1e-6)
check("rate unknown with <2 stamps", L.estimate_rate_hz([5]) is None)

# --- trim / downsample ---
def demo(hz=50):
    s = []
    for i in range(int(10 * hz)):                  # 10 s: 2 s still, 3 s move, 5 s still
        t = i / hz
        u = min(max((t - 2.0) / 3.0, 0.0), 1.0)
        s.append((t, {1: 0.5 + 0.2 * u, 8: 0.9}, {1: 2000 + int(400 * u), 8: 2800}))
    return s
trimmed = L.trim_idle(demo())
check("trim keeps the movement + padding", 3.0 <= trimmed[-1][0] <= 3.8, trimmed[-1][0])
check("trim restarts time at 0", trimmed[0][0] == 0.0)
check("trim of a still recording is empty", L.trim_idle([(i / 10, {1: 0.5}, {1: 2000}) for i in range(30)]) == [])
ds = L.downsample(trimmed, 20)
gaps = [b[0] - a[0] for a, b in zip(ds, ds[1:-1])]
check("downsample to ~20 Hz", all(g >= 0.05 - 1e-9 for g in gaps) and len(ds) < len(trimmed), (len(ds), len(trimmed)))
check("downsample keeps first and last", ds[0] == trimmed[0] and ds[-1] == trimmed[-1])

# --- keyframes ---
s = []
for i in range(100):                                  # move, pause 0.5 s, gripper closes then opens
    t = i * 0.05
    j1 = 0.5 + 0.2 * min(t / 1.5, 1.0)
    g = 0.9 if t < 2.5 else (0.9 - 0.6 * min((t - 2.5) / 0.8, 1.0) if t < 3.6 else 0.3 + 0.5 * min((t - 3.6) / 0.8, 1.0))
    s.append((t, {1: j1, 8: g}, {}))
kf = L.extract_keyframes(s, gripper_id=8)
labels = [k["label"] for k in kf]
check("keyframes start/end", labels[0] == "start" and labels[-1] == "end", labels)
check("keyframes find the pause", "pause" in labels, labels)
check("keyframes find gripper turn", "gripper turns" in labels, labels)

# --- calibration change ---
check("range change detected beyond 20 steps", L.ranges_changed({"1": [1000, 3000]}, {1: (1030, 3000)}) == {1: "[1000, 3000] -> [1030, 3000]"})
check("small range change ignored", L.ranges_changed({"1": [1000, 3000]}, {1: (1010, 2990)}) == {})

# --- practice ladder (team rule: 2 successes in a row) ---
p = {"level": 0, "streak": 0}
p = L.ladder_update(p, True)
check("1 success: same level", p["level"] == 0 and p["streak"] == 1 and p["speed_scale"] == 0.5)
p = L.ladder_update(p, True)
check("2 successes: next level 0.75", p["level"] == 1 and p["streak"] == 0 and p["speed_scale"] == 0.75)
p = L.ladder_update(L.ladder_update(p, True), True)
check("2 more: 1.0", p["speed_scale"] == 1.0)
p = L.ladder_update(p, False)
check("failure: one level down, streak reset", p["speed_scale"] == 0.75 and p["streak"] == 0)
p = L.ladder_update({"level": 0, "streak": 0}, False)
check("failure at bottom stays at 0.5", p["speed_scale"] == 0.5)
p = {"level": len(L.SPEED_LEVELS) - 1, "streak": 1}
check("top level stays at max", L.ladder_update(p, True)["speed_scale"] == L.SPEED_LEVELS[-1])

# --- self-learning: compress a slow Claude-style attempt ---
def claude_attempt():
    samples, t, pose = [], 0.0, {1: 0.5, 2: 0.3, 8: 0.9}
    def hold(sec):
        nonlocal t
        for _ in range(int(sec * 20)):
            samples.append((t, dict(pose), {k: int(1000 + 2000 * v) for k, v in pose.items()})); t += 0.05
    def move(j, d, steps=6):
        nonlocal t
        for _ in range(steps):
            pose[j] += d / steps
            samples.append((t, dict(pose), {k: int(1000 + 2000 * v) for k, v in pose.items()})); t += 0.05
    hold(2)
    for _ in range(6): move(2, 0.05); hold(2)          # reach in 6 small steps, thinking 2 s each
    for _ in range(2): move(8, -0.3); hold(2)          # close gripper in 2 steps
    for _ in range(3): move(2, -0.05); hold(2)         # lift in 3 steps
    return samples, t
att, att_s = claude_attempt()
rest = L.pause_keyframes(att)
check("every thinking pause becomes a rest pose", len(rest) == 12, len(rest))
poses = L.simplify_path(rest, gripper_id=8)
shape = [(round(p[0][2], 2), round(p[0][8], 2)) for p in poses]
check("steps on a line merge; gripper close is one move", shape == [(0.3, 0.9), (0.6, 0.9), (0.6, 0.3), (0.45, 0.3)], shape)
kf = L.timed_keyframes(poses, base_vel=0.25)
total = sum(k["duration_s"] for k in kf)
check("attempt ~27 s -> skill ~6 s at speed 1.0", att_s > 25 and 5.5 < total < 7.0, (att_s, total))
check("keyframes carry raw positions", all("raw" in k for k in kf))
corner = [({1: 0.0}, {}), ({1: 0.5}, {}), ({1: 0.0}, {})]   # go out and come back: must keep the turn
check("a real turn is kept, not merged", len(L.simplify_path(corner, gripper_id=8)) == 3)
check("still recording -> one pose pair", len(L.pause_keyframes([(i * 0.05, {1: 0.5}, {1: 2000}) for i in range(40)])) == 2)

print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
