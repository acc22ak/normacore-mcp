"""Unit tests for motion.py (pure logic, no station needed).

usage: python tests/test_motion.py <normacore-mcp dir>
"""
import sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import motion as M

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol

cur = {1: 0.5, 2: 0.5, 8: 0.9}

# --- keyframes ---
tl = M.build_timeline(cur, [{"joints": {"1": 0.7}, "duration_s": 2.0}], max_vel=10)
check("keyframe duration kept when slow enough", approx(tl.duration, 2.0), tl.duration)
check("unlisted joints hold", tl.sample(1.0)[2] == 0.5 and tl.sample(1.0)[8] == 0.9)
check("eased midpoint is halfway", approx(tl.sample(1.0)[1], 0.6))
check("eased start is slow (smoothstep)", tl.sample(0.2)[1] - 0.5 < 0.2 * 0.1, tl.sample(0.2)[1])
check("end reached", approx(tl.sample(5.0)[1], 0.7))
check("peak velocity = 1.5x average for eased", approx(tl.peak_velocity()[1], 1.5 * 0.2 / 2.0))

tl = M.build_timeline(cur, [{"joints": {"1": 0.9}, "duration_s": 0.1}], max_vel=0.5)
check("too-fast segment is stretched to max_vel", approx(tl.duration, 1.5 * 0.4 / 0.5) and tl.stretched_segments == 1, tl.duration)
check("peak never exceeds max_vel", max(tl.peak_velocity().values()) <= 0.5 + 1e-9)

tl = M.build_timeline(cur, [{"joints": {"1": 0.6}, "duration_s": 2.0}], speed_scale=2.0, max_vel=10)
check("speed_scale 2 halves the time", approx(tl.duration, 1.0))

tl = M.build_timeline(cur, [{"joints": {"1": 0.6}, "duration_s": 1.0}, {"joints": {"2": 0.2}, "duration_s": 1.0}], max_vel=10)
check("keyframes chain", approx(tl.duration, 2.0) and approx(tl.sample(2.0)[1], 0.6) and approx(tl.sample(2.0)[2], 0.2))

tl = M.build_timeline(cur, [{"joints": {"1": 1.4, "2": -0.3}, "duration_s": 1.0}], max_vel=10, limits={1: (0.1, 0.8)})
check("targets clamped to soft limits and 0..1", approx(tl.sample(9)[1], 0.8) and approx(tl.sample(9)[2], 0.0) and set(tl.clamped) == {1, 2}, tl.clamped)

# --- dense recordings ---
dense = [{"t": 0.0, "joints": {"1": 0.6}}, {"t": 0.5, "joints": {"1": 0.65}}, {"t": 1.0, "joints": {"1": 0.7}}]
tl = M.build_timeline(cur, dense, max_vel=10, approach_vel=0.2)
appr = 1.5 * 0.1 / 0.2
check("dense: gentle approach to first sample", approx(tl.points[1].t, appr) and approx(tl.points[1].joints[1], 0.6))
check("dense: original spacing kept after approach", approx(tl.duration, appr + 1.0))
check("dense samples are linear (not eased)", approx(tl.sample(appr + 0.25)[1], 0.625))
tl = M.build_timeline({1: 0.6, 2: 0.5, 8: 0.9}, dense, max_vel=10)
check("dense: no approach when already at start", approx(tl.points[1].t, 0.0) and approx(tl.duration, 1.0))

# --- bad input ---
for name, wps in [("empty", []), ("mixed", [{"t": 0, "joints": {}}, {"joints": {}, "duration_s": 1}]),
                  ("time goes back", [{"t": 1, "joints": {}}, {"t": 0.5, "joints": {}}]),
                  ("unknown joint", [{"joints": {"9": 0.5}, "duration_s": 1}])]:
    try:
        M.build_timeline(cur, wps)
        check(f"rejects {name}", False, "no error")
    except ValueError:
        check(f"rejects {name}", True)

# --- safety monitor ---
def motors(**over):
    base = {j: {"joint": f"j{j}", "error_status": 0, "current": 10, "position_norm": 0.5} for j in (1, 2, 8)}
    for k, v in over.items():
        j, field = k.split("_", 1)
        base[int(j[1:])][field] = v
    return base

cfg = M.SafetyConfig(current_limit=100, per_motor_current={2: 60}, gripper_id=8, track_err_max=0.1, track_err_ticks=3, stale_s=0.5)
cmd = {1: 0.5, 2: 0.5, 8: 0.5}
mon = M.SafetyMonitor(cfg)
check("all fine -> continue", mon.check(motors(), cmd, 0.05) is None)
check("stale state aborts", "old" in (M.SafetyMonitor(cfg).check(motors(), cmd, 0.8) or ""))
check("overload bit aborts", "error status 0x20" in (M.SafetyMonitor(cfg).check(motors(j1_error_status=0x20), cmd, 0) or ""))
check("voltage bit alone is ignored (like Station)", M.SafetyMonitor(cfg).check(motors(j1_error_status=0x01), cmd, 0) is None)
check("arm over-current aborts", "current 150" in (M.SafetyMonitor(cfg).check(motors(j1_current=150), cmd, 0) or ""))
check("per-motor current limit used", "limit 60" in (M.SafetyMonitor(cfg).check(motors(j2_current=70), cmd, 0) or ""))
mon = M.SafetyMonitor(cfg)
check("gripper over-current = grasp, not abort", mon.check(motors(j8_current=300), cmd, 0) is None and mon.gripper_blocked)
mon = M.SafetyMonitor(cfg)
r = [mon.check(motors(j8_position_norm=0.75), cmd, 0) for _ in range(5)]
check("stalled gripper = grasp, never a tracking abort", all(x is None for x in r) and mon.gripper_blocked, r)
mon = M.SafetyMonitor(cfg)
r = [mon.check(motors(j1_position_norm=0.75), cmd, 0) for _ in range(3)]
check("tracking error aborts only after N ticks", r[0] is None and r[1] is None and "away from its target" in (r[2] or ""), r)
mon = M.SafetyMonitor(cfg)
mon.check(motors(j1_position_norm=0.75), cmd, 0); mon.check(motors(), cmd, 0)
check("tracking counter resets when back on track", mon.over_count[1] == 0)
check("max tracking error recorded", approx(mon.max_track_err[1], 0.25))

check("steps_per_s floor 300", M.steps_per_s(0.0, 2000) == 300)
check("steps_per_s cap 3300 (Station max_speed)", M.steps_per_s(5.0, 2000) == 3300)
check("steps_per_s scales with velocity", M.steps_per_s(0.5, 2000) == 1500)

print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
