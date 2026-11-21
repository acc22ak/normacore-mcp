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

# --- leader-arm demonstrations: keep the path shape ---
L_path = [(i * 0.02, {1: min(i, 100) / 200, 2: max(i - 100, 0) / 200, 8: 0.9}, {}) for i in range(201)]  # continuous L, no pauses
pts = L.decimate(L_path, 0.1)
check("decimate: ~0.1 s spacing, keeps first/last", 38 <= len(pts) <= 41 and pts[0] == (L_path[0][1], L_path[0][2]) and pts[-1] == (L_path[-1][1], L_path[-1][2]), len(pts))
shape = [(round(p[0][1], 2), round(p[0][2], 2)) for p in L.simplify_path(pts, gripper_id=8, tol=0.01)]
check("continuous L keeps start, corner, end", shape == [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)], shape)
spiral = [({1: 0.5 + 0.4 * (i / 5000) * __import__("math").cos(i / 30), 2: 0.5 + 0.4 * (i / 5000) * __import__("math").sin(i / 30)}, {}) for i in range(5000)]
check("long wiggly path doesn't hit the recursion limit", len(L.simplify_path(spiral, gripper_id=8, tol=0.001)) > 100)

print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
