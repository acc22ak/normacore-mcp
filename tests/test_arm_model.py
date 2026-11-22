"""Unit tests for arm_model.py (ElRobot kinematics, no station needed).

usage: python tests/test_arm_model.py <normacore-mcp dir> [<norma-core repo>]
The URDF comes from the norma-core checkout (station-viewer's public/devices/elrobot).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import numpy as np
from arm_model import ArmModel

REPO = Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NORMA_CORE_REPO", "../norma-core"))
URDF = REPO / "software/station/clients/station-viewer/public/devices/elrobot/elrobot_follower.urdf"

FAILS = []
def check(name, cond, detail=None):
    cond = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail is not None and not cond else ""))
    if not cond:
        FAILS.append(name)

m = ArmModel.load(URDF)
m.offsets = {"rev_motor_03": -0.004511528608027504, "rev_motor_05": -0.5699085487077533,
             "rev_motor_07": -0.027848717948717905}

# Tip positions read from the station-viewer 3D view (window.__realsense.tip()) for the same joint
# readings, 2026-11-21. If the Python port drifts from the viewer, the depth camera calibration
# (fitted against the viewer's model) and pick/place would silently disagree by centimetres.
VIEWER = [
    ({1: 0.448, 2: 0.06, 3: 0.52, 4: 0.651, 5: 0.543, 6: 0.405, 7: 0.544, 8: 0.547}, [0.0717, 0.3197, 0.0086]),
    ({1: 0.513, 2: 0.017, 3: 0.502, 4: 0.938, 5: 0.545, 6: 0.468, 7: 0.543, 8: 0.538}, [0.1473, 0.1210, -0.0052]),
]
for i, (q, want) in enumerate(VIEWER):
    got = m.tip(q)
    err = float(np.linalg.norm(got - want))
    check(f"tip matches the 3D view ({i})", err < 0.003, f"{got.round(4)} vs {want}: {err * 1000:.1f} mm")

home = VIEWER[1][0]
target = [0.357, 0.05, 0.075]
r = m.solve_ik(target, home, point_down=0.1, approach_y=-0.5, limits={2: (0.01, 0.85)})
check("IK reaches a table point 36 cm out", r.error < 0.003, r.error)
check("IK respects the shoulder limit", r.positions[2] <= 0.85 + 1e-9, r.positions[2])
check("IK tilt goal: jaws ~30 deg below horizontal", abs(m.approach_dir(r.positions | {8: 0.5})[1] + 0.5) < 0.1,
      m.approach_dir(r.positions | {8: 0.5}))
check("IK solution's FK lands on the target", np.linalg.norm(m.grasp_point(r.positions | {8: 0.5}, 0.0) - target) < 0.003)

# The aim point is a fixed point of the gripper: opening the jaws or rolling the wrist must not move it.
# (It used to come from the gripper's world bounding box and wandered up to 14 mm with the pose.)
import arm_model as AM
down = m.solve_ik([0.25, 0.04, 0.05], home, point_down=0.5, approach_y=-1.0, limits={2: (0.01, 0.9)}, reach=0.04)
q = down.positions | {8: 0.8}
gp = m.grasp_point(q, 0.04)
check("vertical grasp reachable 25 cm out", down.error < 0.003, down.error)
check("jaws really point straight down", m.approach_dir(q)[1] < -0.99, m.approach_dir(q))
check("opening the jaws doesn't move the aim point", np.linalg.norm(m.grasp_point(q | {8: 0.2}, 0.04) - gp) < 1e-9)
check("rolling the wrist 60 deg moves the aim point < 3 mm",
      np.linalg.norm(m.grasp_point(q | {7: q[7] + 60 / 343}, 0.04) - gp) < 0.003)
locals_ = []
for pose in (q, home, r.positions | {8: 0.5}):
    T = m.end_transform(pose)
    locals_.append(T[:3, :3].T @ (m.grasp_point(pose, 0.04) - T[:3, 3]))
check("aim point is the same point of the gripper in every pose",
      all(np.linalg.norm(l - (AM.TIP_LOCAL + AM.JAW_AXIS_LOCAL * 0.04)) < 1e-9 for l in locals_), locals_)
check("jaws close sideways (perpendicular to where they point)", abs(m.jaw_close_dir(q) @ m.approach_dir(q)) < 1e-9)

# A fixed joint stays exactly where it was put, and the solver still reaches the target with the rest.
fx = m.solve_ik([0.25, 0.04, 0.05], home, point_down=0.5, approach_y=-1.0, limits={2: (0.01, 0.9)}, reach=0.04,
                fixed={7: 0.70})
check("IK with a fixed wrist roll keeps it", fx.positions[7] == 0.70, fx.positions[7])
check("IK with a fixed wrist roll still reaches", fx.error < 0.003, fx.error)
check("IK honours hard roll limits", all(0.29 - 1e-9 <= m.solve_ik([0.42, 0.03, 0.1], home, point_down=0.5,
      approach_y=-1.0, limits={2: (0.01, 0.9), 5: (0.29, 0.79), 7: (0.29, 0.79)}, reach=0.04).positions[j] <= 0.79 + 1e-9
      for j in (5, 7)))
check("roll joints stay near straight (camera cable)",
      all(abs(r.positions[j] - home[j]) < 0.2 for j in (3, 5, 7)), {j: r.positions[j] for j in (3, 5, 7)})

far = m.solve_ik([1.0, 0.05, 0.0], home, limits={2: (0.01, 0.85)})
check("IK reports an unreachable target instead of pretending", far.error > 0.1, far.error)

# True joint scale: the servo range's real rotation, centred on the URDF range's middle.
import math
from arm_fit import servo_spans
spans = servo_spans({2: (947, 3149), 6: (1005, 3091)})
check("servo span of the shoulder is 193.5 deg", abs(math.degrees(spans["rev_motor_02"]) - 193.5) < 0.1,
      math.degrees(spans["rev_motor_02"]))
check("servo span handles a range wrapping past 4095", abs(servo_spans({1: (4000, 100)})["rev_motor_01"]
      - 196 * 2 * math.pi / 4096) < 1e-9)
j2 = m.joints["rev_motor_02"]
m.spans = {"rev_motor_02": spans["rev_motor_02"]}
mid = (j2.lower + j2.upper) / 2
check("spans keep the range middle", abs(m.joint_values({2: 0.5})["rev_motor_02"] - mid) < 1e-9)
check("spans scale the angle away from the middle",
      abs(m.joint_values({2: 0.9})["rev_motor_02"] - (mid + 0.4 * spans["rev_motor_02"])) < 1e-9)
m.spans = {}
check("no spans: the viewer's mapping", abs(m.joint_values({2: 0.9})["rev_motor_02"]
      - (j2.lower + 0.9 * (j2.upper - j2.lower))) < 1e-9)

print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
