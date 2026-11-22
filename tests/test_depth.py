"""Unit tests for depth.py (RealSense frames, geometry, object detection; no camera needed).

usage: python tests/test_depth.py <normacore-mcp dir>
Builds synthetic depth frames by ray-casting a scene (tilted table + boxes) from a known camera pose.
"""
import io
import json
import struct
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import numpy as np
from PIL import Image
import depth as D

FAILS = []
def check(name, cond, detail=None):
    cond = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail is not None and not cond else ""))
    if not cond:
        FAILS.append(name)

W, H = 320, 240
INTR = {"fx": 190.0, "fy": 190.0, "ppx": 159.5, "ppy": 119.5}
POSE = {"x": 0.89, "y": 0.57, "z": 0.01, "yaw": 1.556, "pitch": -0.632, "roll": -0.014}


def render(boxes, table_tilt=(0.0, 0.0, 0.0)):
    """Depth (mm) + color for a table y = a*x + b*z + c with axis-aligned boxes standing on it."""
    a, b, c = table_tilt
    T = D.pose_matrix(POSE)
    R, t = T[:3, :3], T[:3, 3]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(u - INTR["ppx"]) / INTR["fx"], -(v - INTR["ppy"]) / INTR["fy"], -np.ones_like(u, float)], -1)
    dw = dirs @ R.T                                   # world ray per unit optical depth
    # table: t_y + s*dw_y = a*(t_x + s*dw_x) + b*(t_z + s*dw_z) + c
    s_tab = (a * t[0] + b * t[2] + c - t[1]) / (dw[..., 1] - a * dw[..., 0] - b * dw[..., 2])
    s = np.where(s_tab > 0, s_tab, np.inf)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (200, 160, 110)
    for (x0, x1, z0, z1, h, color) in boxes:
        base = a * (x0 + x1) / 2 + b * (z0 + z1) / 2 + c
        lo = np.array([x0, base, z0])
        hi = np.array([x1, base + h, z1])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - t) / dw
            t2 = (hi - t) / dw
        tn = np.nanmax(np.minimum(t1, t2), axis=-1)
        tf = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tn <= tf) & (tn > 0) & (tn < s)
        s = np.where(hit, tn, s)
        rgb[hit] = color
    depth = np.where(np.isfinite(s), s * 1000, 0).astype("<u2")
    return depth, rgb


def make_frame(depth, rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((640, 480)).save(buf, format="JPEG", quality=95)
    jpeg = buf.getvalue()
    header = json.dumps({"type": "frame", "seq": 1, "ts": 0, "serial": "x",
                         "color": {"w": 640, "h": 480, "bytes": len(jpeg)},
                         "depth": {"w": W, "h": H, "bytes": depth.nbytes, "unit_m": 0.001},
                         "intrinsics": INTR}).encode()
    return struct.pack("<I", len(header)) + header + jpeg + depth.tobytes()


# A wrong axis convention (optical vs rig vs world) would put objects mirrored or at the wrong
# height while the code still "finds" something: compare against the scene we rendered.
minifig = (0.35, 0.366, 0.067, 0.083, 0.042, (240, 240, 240))
candy = (0.27, 0.30, 0.28, 0.31, 0.015, (250, 250, 250))
pen = (0.25, 0.39, -0.215, -0.205, 0.011, (30, 140, 60))
depth, rgb = render([minifig, candy, pen], table_tilt=(-0.0011, -0.002, 0.0025))
frame = D.parse_frame(make_frame(depth, rgb))
check("frame parses", frame.depth_mm.shape == (H, W))

objs = D.find_objects(frame, POSE)
by_color = {o["color"]: o for o in objs}
check("finds 3 objects on a slightly tilted table, no table noise", len(objs) == 3,
      [(o["color"], o["center"], o["size_cm"]) for o in objs])
fig = min(objs, key=lambda o: np.hypot(o["center"][0] - 0.358, o["center"][2] - 0.075))
check("minifigure position within 1 cm", np.hypot(fig["center"][0] - 0.358, fig["center"][2] - 0.075) < 0.01,
      fig["center"])
check("minifigure height within 6 mm", abs(fig["top_m"] - 0.042) < 0.006, fig["top_m"])
long_obj = [o for o in objs if "ends" in o]
check("the pen is reported as elongated with two ends", len(long_obj) == 1 and len(long_obj[0]["ends"]) == 2,
      [o["size_cm"] for o in objs])
if long_obj:
    ends_x = sorted(e["xz"][0] for e in long_obj[0]["ends"])
    check("pen ends near its true ends (x 0.25 / 0.39)", abs(ends_x[0] - 0.25) < 0.02 and abs(ends_x[1] - 0.39) < 0.02,
          ends_x)
check("color naming: green pen", any(o["color"] == "green" for o in objs), [o["color"] for o in objs])

# Arm exclusion: points near the arm model must not become objects.
arm = np.array([[0.358, y, 0.075] for y in np.linspace(0, 0.05, 20)])
objs2 = D.find_objects(frame, POSE, arm_points=arm)
check("depth points on the arm are ignored", len(objs2) == 2, [o["center"] for o in objs2])

p = D.pixel_to_world(frame, POSE, 0.5, 0.9)
check("pixel_to_world lands on the table", p is not None and abs(p[1] - (-0.0011 * p[0] - 0.002 * p[2] + 0.0025)) < 0.004,
      p)


# ---- ArUco marker on the gripper: a slip between optical / rig / world axes would report the marker
# mirrored or facing the wrong way, and the gripper correction would push the arm the wrong way.
import cv2

def render_marker(world_center, normal, x_axis, size_m=0.035, marker_id=0):
    T = D.pose_matrix(POSE)
    R, t = T[:3, :3], T[:3, 3]
    ci = {"fx": INTR["fx"] * 2, "fy": INTR["fy"] * 2, "ppx": (INTR["ppx"] + 0.5) * 2 - 0.5, "ppy": (INTR["ppy"] + 0.5) * 2 - 0.5}
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    xa = np.asarray(x_axis, float); xa -= n * (xa @ n); xa /= np.linalg.norm(xa)
    ya = np.cross(n, xa)
    s = size_m / 2
    corners_w = [world_center + (-s) * xa + s * ya, world_center + s * xa + s * ya,
                 world_center + s * xa - s * ya, world_center - s * xa - s * ya]          # TL TR BR BL
    def project(pw):
        local = R.T @ (pw - t)                       # rig frame
        opt = D.OPTICAL_TO_RIG @ local               # optical (diag is its own inverse)
        return [ci["fx"] * opt[0] / opt[2] + ci["ppx"], ci["fy"] * opt[1] / opt[2] + ci["ppy"]], opt[2]
    px = np.array([project(c)[0] for c in corners_w], np.float32)
    # Marker image with a white quiet zone, warped into a light-gray 640x480 scene.
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, 200)
    m = cv2.copyMakeBorder(m, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    src = np.array([[50, 50], [250, 50], [250, 250], [50, 250]], np.float32)
    H = cv2.getPerspectiveTransform(src, px)
    scene = np.full((480, 640), 170, np.uint8)
    warped = cv2.warpPerspective(m, H, (640, 480), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(m, 255), H, (640, 480), borderValue=0)
    scene[mask > 0] = warped[mask > 0]
    # Depth: the marker plane where it covers the depth grid, table elsewhere.
    depth, _ = render([])
    u, v = np.meshgrid(np.arange(W), np.arange(H_ := 240))
    dirs = np.stack([(u - INTR["ppx"]) / INTR["fx"], -(v - INTR["ppy"]) / INTR["fy"], -np.ones_like(u, float)], -1) @ R.T
    sm = ((world_center - t) @ n) / (dirs @ n)
    small_mask = cv2.resize(mask, (W, 240), interpolation=cv2.INTER_NEAREST) > 0
    depth = np.where(small_mask & (sm > 0), sm * 1000, depth).astype("<u2")
    rgb = np.repeat(cv2.resize(scene, (W, 240))[..., None], 3, axis=2)
    buf = make_frame(depth, rgb)
    # make_frame resizes rgb back up to 640x480; use the sharp full-res scene instead.
    hl = struct.unpack_from("<I", buf, 0)[0]
    header = json.loads(buf[4:4 + hl])
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    header["color"]["bytes"] = len(jpg)
    hb = json.dumps(header).encode()
    return D.parse_frame(struct.pack("<I", len(hb)) + hb + jpg.tobytes() + depth.tobytes())

centre = np.array([0.30, 0.12, -0.05])
to_cam = np.array([POSE["x"], POSE["y"], POSE["z"]]) - centre
mframe = render_marker(centre, to_cam, [0, 0, 1])
marks = D.detect_markers(mframe, POSE, 0.035)
check("marker detected", len(marks) == 1 and marks[0]["id"] == 0, marks)
if marks:
    mk = marks[0]
    check("marker position within 5 mm (depth)", np.linalg.norm(np.array(mk["position"]) - centre) < 0.005,
          (mk["position"], mk["source"]))
    nrm = to_cam / np.linalg.norm(to_cam)
    # A 3.5 cm marker at ~80 cm: orientation from 4 close corners is only good to ~20 deg; position is what
    # the gripper correction uses. This guards the sign/axis convention, not the precision.
    check("marker faces the camera (normal, sign/axes)", float(np.dot(mk["normal"], nrm)) > 0.9, (mk["normal"], nrm.round(3)))
    check("marker top edge direction (x_axis ~ +z)", float(np.dot(mk["x_axis"], [0, 0, 1])) > 0.9, mk["x_axis"])
    check("solvePnP agrees with depth within 1.5 cm", mk["pnp_depth_gap_mm"] < 15, mk["pnp_depth_gap_mm"])

# ---- merge_frames: median depth over several frames
import copy
f2 = copy.copy(frame); f2.depth_mm = frame.depth_mm.copy(); f2.depth_mm[frame.depth_mm > 0] += 6
f3 = copy.copy(frame); f3.depth_mm = np.zeros_like(frame.depth_mm)
merged = D.merge_frames([frame, f2, f2])
check("merge_frames takes the per-pixel median", np.array_equal(merged.depth_mm, f2.depth_mm))
check("merge_frames ignores frames with no depth at a pixel",
      np.array_equal(D.merge_frames([frame, f3, frame]).depth_mm, frame.depth_mm))
check("merge_frames drops pixels seen in under half the frames", not D.merge_frames([frame, f3, f3]).depth_mm.any())

# ---- find_jaw_tips on the real calibration frames (skipped when the data isn't there)
import json
from pathlib import Path
HERE = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
cal_file = HERE / "realsense-bridge" / "calibration.json"
try:        # the frames the current calibration was fitted on (each recalibration captures a new folder)
    samples = HERE / "data" / json.loads(cal_file.read_text())["arm_fit"].get("folder", "arm_fit_samples")
except Exception:
    samples = HERE / "data" / "arm_fit_samples"
try:
    import arm_fit as AF, arm_model as AM
    have = samples.exists() and cal_file.exists() and AF.URDF.exists()
except Exception:
    have = False
if have:
    fit = json.loads(cal_file.read_text())["arm_fit"]
    model = AM.ArmModel.load(AF.URDF)
    model.offsets, model.spans = fit["joint_offsets"], fit["joint_spans"]
    xs = {}
    for name, fr, norm in AF.load_samples(samples):
        T = model.end_transform(norm)
        jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
        found = D.find_jaw_tips(fr, fit["camera_pose"], jaw_end)
        if name == "p0_home":
            check("jaw tips: nothing reported for the folded home pose", found is None, found)
        if found and model.approach_dir(norm)[1] < -0.7:
            xs[name] = float((T[:3, :3].T @ (np.array(found["midpoint"]) - T[:3, 3]))[0])
            check(f"jaw tips ({name}): gap is a plausible jaw opening", 0.02 < found["gap_m"] < 0.05, found["gap_m"])
    check("jaw tips found in at least three frames with the jaws pointing down", len(xs) >= 3, xs)
    check("real jaw midpoint is 1-2.5 cm sideways of the model's, always to the same side",
          all(-0.026 < v < -0.008 for v in xs.values()), xs)
else:
    print("SKIP jaw tip checks (no data/arm_fit_samples or calibration.json)")

# ---- find_objects_rgb on real frames (skipped when the data isn't there)
if have:
    whole = json.loads(cal_file.read_text())
    cases = [("arm_fit_samples_1122", whole["arm_fit"])]
    if "arm_fit_prev" in whole and (HERE / "data" / "arm_fit_samples").exists():
        cases.append(("arm_fit_samples", whole["arm_fit_prev"]))
    for folder, f in cases:
        mdl = AM.ArmModel.load(AF.URDF)
        mdl.offsets, mdl.spans = f["joint_offsets"], f["joint_spans"]
        seen = {}
        for name, fr, norm in AF.load_samples(HERE / "data" / folder):
            if name in ("p0_home", "p4"):
                seen[name] = D.find_objects_rgb(fr, f["camera_pose"], mdl.surface_points(norm))
        home_objs = seen["p0_home"]
        pens = [o for o in home_objs if "ends" in o and 11 < o["size_cm"][0] < 17 and o["size_cm"][1] < 3.5]
        check(f"rgb objects ({folder}): the pen is found as a long flat object with two ends", len(pens) == 1,
              [(o["size_cm"], "ends" in o) for o in home_objs])
        check(f"rgb objects ({folder}): nothing is reported as standing on a table of flat things",
              not any(o["standing"] for o in home_objs), [(o["size_cm"], o["standing"]) for o in home_objs])
        check(f"rgb objects ({folder}): no junk beyond the arm's reach", all(o["distance_m"] <= 0.46 for o in home_objs))
        # Same scene, arm in a different pose: the same objects at the same places (the arm isn't an object).
        a = sorted((o["center"][0], o["center"][2]) for o in home_objs)
        b = sorted((o["center"][0], o["center"][2]) for o in seen["p4"])
        check(f"rgb objects ({folder}): same objects within 5 mm whatever the arm's pose",
              len(a) == len(b) and all(abs(p[0] - q[0]) < 0.005 and abs(p[1] - q[1]) < 0.005 for p, q in zip(a, b)), (a, b))
    if len(cases) > 1:
        # 2026-11-21: a minifigure lying flat (0.8 cm) that the depth-only detector could not see.
        prev = whole["arm_fit_prev"]
        name, fr, norm = next(s for s in AF.load_samples(HERE / "data" / "arm_fit_samples") if s[0] == "p0_home")
        mdl = AM.ArmModel.load(AF.URDF)
        mdl.offsets, mdl.spans = prev["joint_offsets"], prev["joint_spans"]
        arm = mdl.surface_points(norm)
        lying = [o for o in D.find_objects_rgb(fr, prev["camera_pose"], arm) if 3.5 < o["size_cm"][0] < 6 and o["top_m"] < 0.015
                 and abs(o["center"][0] - 0.38) < 0.02 and abs(o["center"][2] - 0.175) < 0.02]
        check("rgb objects: the lying minifigure is found", len(lying) == 1, lying)
        check("depth-only detector misses that same lying minifigure (why this exists)",
              not any(abs(o["center"][0] - 0.38) < 0.03 and abs(o["center"][2] - 0.175) < 0.03
                      for o in D.find_objects(fr, prev["camera_pose"], arm)))

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
