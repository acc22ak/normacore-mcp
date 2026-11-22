"""Offline tests for vision.py + pickplace.py on real gripper-camera frames from 2026-11-21.

usage: python tests/test_pickplace.py <normacore-mcp dir>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
import pickplace as P  # noqa: E402
import vision as V  # noqa: E402
from PIL import ImageDraw  # noqa: E402

FRAMES = ROOT / "tests" / "frames"
CFG = P.DEFAULTS
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    fails += 0 if cond else 1


def load(name):
    return V.decode((FRAMES / f"{name}.jpg").read_bytes())


def blobs(img, mask_jaws=True):
    w, h = img.size
    return [P.norm_blob(b, w, h) for b in V.dark_blobs(img, mask_jaws=mask_jaws)]


def decide(name, yaw=0.5):
    t = P.pick_target(blobs(load(name)), CFG)
    j = {"base_yaw": yaw, "shoulder": 0.4, "elbow": 0.7}
    return t, (P.pick_step(t, j, P.AdaptiveGain(1000 / 299, 1, 10), CFG) if t else None)


# --- pick decisions on real frames
t, d = decide("pick_far_left")
check("far motor found left of the jaws", t and t["cx"] < 0.4, t)
check("far motor -> approach and turn yaw +", d and d["phase"] == "approach" and d["dyaw"] > 0, d)
t, d = decide("pick_mid")
check("mid motor -> keep approaching", d and d["phase"] == "approach" and not d["done"], d)
t, d = decide("pick_ready")
check("motor between the jaws -> grasp", d and d["done"], (t, d))
t, d = decide("home_motor_ahead")
check("small motor low in view (seen from high up) is NOT a grasp", d and not d["done"], (t, d))
check("approach respects the shoulder limit",
      P.pick_step({"cx": 0.45, "cy": 0.3, "left": 0.4, "right": 0.5, "top": 0.2, "bottom": 0.4, "area": 0.02},
                  {"base_yaw": 0.5, "shoulder": 0.67, "elbow": 0.6}, P.AdaptiveGain(3, 1, 10), CFG)["limited"])

# --- held check
check("held motor at carry height -> held", P.is_held(blobs(load("carry_held"), False), CFG))
check("open jaws, no motor -> not held", not P.is_held(blobs(load("home_no_motor"), False), CFG))

# --- grasp verdicts (hardware: stall 0.42-0.44 on the motor, 0.66-0.69 when a jaw hit its top)
check("grasp 0.43 ok", P.grasp_verdict(0.43, CFG) == "ok")
check("grasp 0.67 on_top", P.grasp_verdict(0.67, CFG) == "on_top")
check("grasp 0.30 missed", P.grasp_verdict(0.30, CFG) == "missed")

# --- markers: none in the real scene, a synthetic green sticker is found
img = load("carry_over_box")
w, h = img.size
check("no false marker in the real box scene", not V.marker_blobs(img), [b.as_dict() for b in V.marker_blobs(img)])
for name in ("pick_far_left", "home_motor_ahead", "after_release"):
    check(f"no false marker in {name}", not V.marker_blobs(load(name)))
ImageDraw.Draw(img).rectangle([170, 60, 200, 85], fill=(40, 170, 70))
ms = V.marker_blobs(img)
check("green sticker detected", len(ms) == 1 and abs(ms[0].cx - 185) < 4 and abs(ms[0].cy - 72) < 4,
      [b.as_dict() for b in ms])

# --- white sticker (the "HACK LAB" label on the box floor) vs glare / background
def white(name):
    im = load(name)
    w, h = im.size
    return P.pick_marker([P.norm_blob(b, w, h) for b in V.white_blobs(im)], CFG)


m = white("box_white_sticker")
check("white sticker found mid-box", m and abs(m["cx"] - 0.52) < 0.05 and abs(m["cy"] - 0.48) < 0.05, m)
for name in ("white_bag_background", "carry_held", "carry_over_box", "pick_ready", "home_no_motor"):
    check(f"no white marker in {name} (glare, held motor's label, background)", white(name) is None, white(name))

# --- place step directions
mk = {"cx": 0.65, "cy": 0.25}
dec = P.place_step(mk, {"base_yaw": 0.5, "shoulder": 0.38, "elbow": 0.62},
                   P.AdaptiveGain(3.3, 1, 10), P.AdaptiveGain(2, 0.5, 8), CFG)
check("marker right of target -> yaw -", dec["dyaw"] < 0, dec)
check("marker above target -> reach forward (shoulder +, elbow -)",
      dec["dreach"] > 0 and dec["targets"]["shoulder"] > 0.38 and dec["targets"]["elbow"] < 0.62, dec)
check("marker on target -> done",
      P.place_step({"cx": CFG["place_target_x"], "cy": CFG["place_target_y"]}, {"base_yaw": 0.5, "shoulder": 0.4,
                   "elbow": 0.6}, P.AdaptiveGain(3, 1, 10), P.AdaptiveGain(2, 0.5, 8), CFG)["done"])

# --- clearance rule: high = at/above the transit pose
check("home is high", P.is_high(CFG["home"], CFG))
check("hover is high", P.is_high({"shoulder": 0.20, "elbow": 0.85}, CFG))
check("transit is high", P.is_high(CFG["transit"], CFG))
check("carry is NOT high enough to swing", not P.is_high({"shoulder": 0.40, "elbow": 0.625}, CFG))
check("grasp pose is low", not P.is_high({"shoulder": 0.615, "elbow": 0.59}, CFG))
check("release pose is low", not P.is_high({"shoulder": 0.527, "elbow": 0.546}, CFG))

# --- adaptive gain learns the real response and ignores glitches
g = P.AdaptiveGain(gain=2.0, lo=0.5, hi=8.0)
for _ in range(4):
    g.observe(0.04, 0.04 * 5.0)
check("gain converges toward the measured 5.0", abs(g.gain - 5.0) < 0.4, g.gain)
g.observe(0.04, -0.3)
check("opposite-sign glitch ignored", abs(g.gain - 5.0) < 0.4, g.gain)
check("command clamps to max_step", abs(g.command(10.0, 0.05)) == 0.05)

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
