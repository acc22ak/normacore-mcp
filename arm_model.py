"""ElRobot kinematics in the station-viewer's 3D frame (pure numpy, no I/O besides loading the URDF).

The frame matches station-viewer's 3D view and the RealSense calibration: metres, robot base at the
origin, Y up, arm front = +Z (the URDF's Z-up frame rotated by the viewer's baseRpy [-pi/2, 0, -pi/2]).

Joint mapping is the viewer's (BaseRobotRenderer / ik.ts): a motor's normalized position p (0..1
inside its calibrated range, the same number as get_robot_state's position_norm) drives URDF joint
`rev_motor_0<id>` at  angle = lower + p * (upper - lower) + offset,  where offset is the per-arm
"Align 3D" roll offset. The "tip" is the viewer's gripper reference point: the centre of the
world-axis-aligned bounding box of the gripper (end link subtree), so tip positions here agree with
the 3D view's orange calibration dot and with `window.__realsense.tip()`.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_JOINTS = [f"rev_motor_0{i}" for i in range(1, 8)]   # motor ids 1..7
GRIPPER_JOINT = "rev_motor_08"
END_LINK_JOINT = "rev_motor_07"                         # its child link's subtree = the gripper
ROLL_JOINTS = {3, 5, 7}                                 # motor ids; cable-limited around 0.54
DOWN = np.array([0.0, -1.0, 0.0])
# The gripper's own frame (child link of the wrist roll joint), measured from the URDF meshes: the jaws run
# along +y out to 104 mm, close along x, and are centred on the wrist roll axis (x = z = 0).
# Aim points are fixed in this frame. The old ones were derived from the gripper's world-axis bounding box,
# which moved them up to 14 mm across the jaws depending on tilt, roll and opening (measured 2026-11-22).
TIP_LOCAL = np.array([0.0, 0.049, 0.0])      # reach 0: where the old bounding-box tip sat on average
JAW_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])   # wrist -> jaw ends
JAW_CLOSE_LOCAL = np.array([1.0, 0.0, 0.0])  # the direction the jaws close along
JAW_END_M = 0.104                            # jaw ends, along the jaw axis from the gripper frame origin


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _tf(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)       # URDF convention


def load_stl_vertices(path: Path) -> np.ndarray:
    """Unique-ish vertices of a binary or ASCII STL, (N, 3) float."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].reshape(-1, 3).astype(float)
    pts = [list(map(float, line.split()[1:4])) for line in data.decode("ascii", "ignore").splitlines()
           if line.strip().startswith("vertex")]
    return np.array(pts, dtype=float).reshape(-1, 3)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    mimic: tuple[str, float, float] | None = None


@dataclass
class ArmModel:
    joints: dict[str, Joint]
    children: dict[str, list[str]]          # link -> child joint names
    root: str
    link_points: dict[str, np.ndarray]      # link -> (N,3) visual vertices in link frame (subsampled)
    base: np.ndarray = field(default_factory=lambda: _tf(_rot_x(-np.pi / 2) @ _rot_z(-np.pi / 2), [0, 0, 0]))
    offsets: dict[str, float] = field(default_factory=dict)   # "Align 3D" roll offsets (rad)
    # True rotation (rad) of each joint's calibrated servo range: (range_max - range_min) * 2pi/4096. Without
    # it a joint's range is squeezed into the URDF limits (the viewer's mapping), which under-reports angles
    # by 2-5% on our arm (shoulder 193.5 deg real vs 184.7 URDF): ~2 cm at the gripper with the arm leaned out.
    spans: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, urdf: Path, max_points_per_link: int = 400) -> "ArmModel":
        root_el = ET.parse(urdf).getroot()
        link_points: dict[str, np.ndarray] = {}
        for link in root_el.findall("link"):
            pts = []
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                o = vis.find("origin")
                T = _tf(_rpy(*_floats(o, "rpy")), _floats(o, "xyz"))
                scale = np.array(_floats(mesh, "scale", "1 1 1"))
                v = load_stl_vertices(urdf.parent / mesh.get("filename")) * scale
                if len(v) > max_points_per_link:
                    v = v[:: int(np.ceil(len(v) / max_points_per_link))]
                pts.append(v @ T[:3, :3].T + T[:3, 3])
            link_points[link.get("name")] = np.vstack(pts) if pts else np.zeros((0, 3))

        joints: dict[str, Joint] = {}
        children: dict[str, list[str]] = {}
        child_links = set()
        for j in root_el.findall("joint"):
            o = j.find("origin")
            axis_el = j.find("axis")
            lim = j.find("limit")
            mim = j.find("mimic")
            joint = Joint(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_tf(_rpy(*_floats(o, "rpy")), _floats(o, "xyz")),
                axis=_unit(np.array(_floats(axis_el, "xyz", "1 0 0"))),
                lower=float(lim.get("lower", 0)) if lim is not None else 0.0,
                upper=float(lim.get("upper", 0)) if lim is not None else 0.0,
                mimic=(mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0)))
                if mim is not None else None,
            )
            joints[joint.name] = joint
            children.setdefault(joint.parent, []).append(joint.name)
            child_links.add(joint.child)
        root = next(n for n in link_points if n not in child_links)
        return cls(joints=joints, children=children, root=root, link_points=link_points)

    # -------------------------------------------------------------- mapping
    def joint_values(self, norm: dict[int, float]) -> dict[str, float]:
        """Motor id -> normalized position  =>  URDF joint name -> angle/offset (viewer mapping)."""
        vals: dict[str, float] = {}
        for mid in range(1, 8):
            name = ARM_JOINTS[mid - 1]
            j = self.joints[name]
            p = float(norm.get(mid, 0.5))
            if name in self.spans:
                vals[name] = (j.lower + j.upper) / 2 + (p - 0.5) * self.spans[name] + self.offsets.get(name, 0.0)
            else:
                vals[name] = j.lower + p * (j.upper - j.lower) + self.offsets.get(name, 0.0)
        g = self.joints.get(GRIPPER_JOINT)
        if g is not None:
            p = float(norm.get(8, 0.5))
            # resolveElrobotJointValue: joints not ending in "_1" are mirrored.
            vals[GRIPPER_JOINT] = g.upper - (g.lower + p * (g.upper - g.lower))
        for name, j in self.joints.items():
            if j.mimic and j.mimic[0] in vals:
                vals[name] = vals[j.mimic[0]] * j.mimic[1] + j.mimic[2]
        return vals

    # ------------------------------------------------------------------- FK
    def link_transforms(self, norm: dict[int, float]) -> dict[str, np.ndarray]:
        vals = self.joint_values(norm)
        out = {self.root: self.base}
        stack = [self.root]
        while stack:
            link = stack.pop()
            for jn in self.children.get(link, []):
                j = self.joints[jn]
                v = vals.get(jn, 0.0)
                if j.type in ("revolute", "continuous"):
                    motion = _tf(_axis_angle(j.axis, v), [0, 0, 0])
                elif j.type == "prismatic":
                    motion = _tf(np.eye(3), j.axis * v)
                else:
                    motion = np.eye(4)
                out[j.child] = out[link] @ j.origin @ motion
                stack.append(j.child)
        return out

    def _subtree_links(self, link: str) -> list[str]:
        out, stack = [], [link]
        while stack:
            lk = stack.pop()
            out.append(lk)
            stack.extend(self.joints[jn].child for jn in self.children.get(lk, []))
        return out

    def end_transform(self, norm: dict[int, float], tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """World transform of the gripper's base link (child of the wrist roll joint)."""
        tfs = tfs or self.link_transforms(norm)
        return tfs[self.joints[END_LINK_JOINT].child]

    def gripper_points(self, norm: dict[int, float], tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        tfs = tfs or self.link_transforms(norm)
        end = self.joints[END_LINK_JOINT].child
        pts = [self.link_points[lk] @ tfs[lk][:3, :3].T + tfs[lk][:3, 3]
               for lk in self._subtree_links(end) if len(self.link_points[lk])]
        return np.vstack(pts)

    def tip(self, norm: dict[int, float], tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Gripper reference point in world coordinates (centre of its world AABB, like the viewer)."""
        pts = self.gripper_points(norm, tfs)
        return (pts.min(axis=0) + pts.max(axis=0)) / 2

    def approach_dir(self, norm: dict[int, float], tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Unit vector along the jaws, wrist -> jaw ends (where the gripper points)."""
        return self.end_transform(norm, tfs)[:3, :3] @ JAW_AXIS_LOCAL

    def jaw_close_dir(self, norm: dict[int, float], tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Unit vector along which the jaws close (world)."""
        return self.end_transform(norm, tfs)[:3, :3] @ JAW_CLOSE_LOCAL

    def grasp_point(self, norm: dict[int, float], reach: float, tfs: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """The point between the jaws, `reach` m past TIP_LOCAL along the jaw axis: a fixed point of the
        gripper, on the wrist roll axis (rolling the wrist or opening the jaws doesn't move it). The jaw
        ends are at 10.4 cm, so reach 0.04 is ~1.5 cm back from them. Aim this point, not tip(), at what
        should end up between the jaws."""
        T = self.end_transform(norm, tfs)
        return T[:3, :3] @ (TIP_LOCAL + JAW_AXIS_LOCAL * reach) + T[:3, 3]

    def surface_points(self, norm: dict[int, float], links: list[str] | None = None) -> np.ndarray:
        tfs = self.link_transforms(norm)
        pts = [self.link_points[lk] @ tfs[lk][:3, :3].T + tfs[lk][:3, 3]
               for lk in (links or self.link_points) if len(self.link_points[lk])]
        return np.vstack(pts)

    # ------------------------------------------------------------------- IK
    def solve_ik(self, target, start: dict[int, float], *, point_down: float = 0.0, approach_y: float = -0.5,
                 limits: dict[int, tuple[float, float]] | None = None,
                 roll_keep: float = 0.05, iterations: int = 120, tol: float = 0.002,
                 reach: float = 0.0, fixed: dict[int, float] | None = None) -> "IkResult":
        """Damped least squares over motor ids 1..7 in normalized units.

        point_down > 0 (weight, ~0.05-0.2) adds a soft goal on how steeply the gripper points down:
        the vertical component of its pointing direction should be `approach_y` (-1 = straight down,
        -0.5 = 30 deg below horizontal, which is what reaches 35-40 cm without maxing the shoulder).
        Roll joints are softly kept near their start (cable) with weight `roll_keep`.
        `limits`: per motor id (lo, hi) normalized bounds; default 0.01..0.99.
        The solved point is grasp_point(reach): reach 0 is TIP_LOCAL (close to, but not the same as, tip()).
        `fixed`: motor id -> value for joints the solver must not move (e.g. a chosen wrist roll).
        """
        target = np.asarray(target, dtype=float)
        fixed = dict(fixed or {})
        ids = [i for i in range(1, 8) if i not in fixed]
        lim = {i: (0.01, 0.99) for i in ids}
        lim.update({i: v for i, v in (limits or {}).items() if i in lim})
        q = np.array([min(max(start.get(i, 0.5), lim[i][0]), lim[i][1]) for i in ids])
        q0 = q.copy()
        extra = {8: start.get(8, 0.5)} | fixed
        rolls = [k for k, i in enumerate(ids) if i in ROLL_JOINTS]

        def residual(qv: np.ndarray) -> np.ndarray:
            norm = {i: qv[k] for k, i in enumerate(ids)} | extra
            tfs = self.link_transforms(norm)
            r = [target - self.grasp_point(norm, reach, tfs)]
            if point_down > 0:
                r.append(np.array([point_down * (approach_y - self.approach_dir(norm, tfs)[1])]))
            if roll_keep > 0 and rolls:
                r.append(roll_keep * (q0[rolls] - qv[rolls]))
            return np.concatenate(r)

        def pos_err(qv: np.ndarray) -> float:
            norm = {i: qv[k] for k, i in enumerate(ids)} | extra
            return float(np.linalg.norm(target - self.grasp_point(norm, reach)))

        r = residual(q)
        cost = float(r @ r)
        damping = 0.02
        eps = 1e-3
        for _ in range(iterations):
            if pos_err(q) < tol and point_down == 0:
                break
            if cost < 1e-8:
                break
            J = np.zeros((len(r), len(q)))
            for k in range(len(q)):
                h = eps if q[k] + eps <= lim[ids[k]][1] else -eps
                qh = q.copy()
                qh[k] += h
                J[:, k] = (residual(qh) - r) / h
            # residual = goal - f(q), so d(residual)/dq = -J_f; solve for step that reduces |r|.
            A = J.T @ J + damping ** 2 * np.eye(len(q))
            step = -np.linalg.solve(A, J.T @ r)
            m = np.max(np.abs(step))
            if m > 0.04:
                step *= 0.04 / m
            accepted = False
            for _ in range(6):
                qn = np.array([min(max(q[k] + step[k], lim[ids[k]][0]), lim[ids[k]][1]) for k in range(len(q))])
                rn = residual(qn)
                cn = float(rn @ rn)
                if cn < cost:
                    q, r, cost = qn, rn, cn
                    damping = max(0.01, damping * 0.7)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                damping *= 3
                if damping > 1:
                    break
        norm = {i: float(q[k]) for k, i in enumerate(ids)} | {i: float(v) for i, v in fixed.items()}
        return IkResult(positions=norm, error=pos_err(q))


@dataclass
class IkResult:
    positions: dict[int, float]
    error: float


def _floats(el, attr: str, default: str = "0 0 0") -> list[float]:
    return [float(x) for x in ((el.get(attr) if el is not None else None) or default).split()]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
