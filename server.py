"""NormaCore Station MCP server: lets Claude (or Codex) see through the station
cameras and move an ST3215 arm (ElRobot / SO-101).

Talks to a running `station --tcp` over its normfs TCP API (port 8888) using the
pure-python client that ships in the norma-core repo.

Env vars:
  NORMA_CORE_REPO   path to a checkout of github.com/norma-core/norma-core (required)
  STATION_HOST      station address, default "localhost" (port 8888 implied)
  FOLLOWER_BUS      ST3215 bus serial of the arm Claude may move. If unset and the
                    station has exactly one bus, that bus is used.
  MAX_STEP_NORM     max change per joint per move call, in normalized units (0..1).
                    Default 0.15. Bigger requests are clamped, not rejected.
  MOVE_SPEED        servo goal speed (steps/s) written before each move. Default 600.
  LEADER_BUS        ST3215 bus serial of the leader arm (for teach_mode). If unset and
                    the station has exactly two buses, the non-follower one is used.
  TEACH_MAX_POSE_DIFF  teach_mode(on) refuses (unless force=True) when any joint of the
                    leader and follower differs by more than this (normalized). Default 0.2.
  CAMERAS           optional default camera roles, e.g. "desk=32e6:9228:202604081837,gripper=3f2a"
                    (camera_id from list_cameras, queue id or queue hash prefix). Roles saved
                    with set_camera_role in data/cameras.json (CAMERAS_FILE) take precedence.
  CAMERA_GONE_S     a camera whose newest frame is older than this (s), or that reported a
                    disconnect, counts as unplugged and is hidden from look("all") and
                    list_cameras. Default 60. Cameras between FRAME_MAX_AGE_S and this show as STALE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import sys
import time
from pathlib import Path

# --- logging must go to stderr: stdout is the MCP stdio channel ---------------
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("normacore-mcp")

REPO = Path(os.environ.get("NORMA_CORE_REPO", "")).expanduser()
if not (REPO / "software" / "station" / "shared" / "station_py").is_dir():
    sys.exit("NORMA_CORE_REPO must point at a norma-core checkout "
             f"(got {str(REPO)!r})")
sys.path.insert(0, str(REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(REPO))

import anyio  # noqa: E402  (dependency of mcp; used to shield the safety hold from cancellation)
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion as M  # noqa: E402

from station_py import ErrQueueNotFound, new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import usbvideo  # noqa: E402
from target.gen_python.protobuf.station import commands as station_commands  # noqa: E402
from target.gen_python.protobuf.station import drivers, inference  # noqa: E402

STATION_HOST = os.environ.get("STATION_HOST", "localhost")
FOLLOWER_BUS = os.environ.get("FOLLOWER_BUS", "").strip()
MAX_STEP_NORM = float(os.environ.get("MAX_STEP_NORM", "0.15"))
MOVE_SPEED = int(os.environ.get("MOVE_SPEED", "600"))
LEADER_BUS = os.environ.get("LEADER_BUS", "").strip()
TEACH_MAX_POSE_DIFF = float(os.environ.get("TEACH_MAX_POSE_DIFF", "0.2"))
# A camera frame older than this is reported as STALE by `look`. Camera queues
# persist across restarts, so an unplugged or frozen camera still has a "latest"
# frame. Age = this machine's clock minus the frame's station wall-clock stamp,
# so it assumes station and MCP server share a clock (same Mac, or NTP-synced).
FRAME_MAX_AGE_S = float(os.environ.get("FRAME_MAX_AGE_S", "2.0"))
# A camera whose newest frame is older than this (or that reported a disconnect) counts as
# unplugged: hidden from look("all") / list_cameras, but still reachable by id for diagnosis.
CAMERA_GONE_S = float(os.environ.get("CAMERA_GONE_S", "60"))
READ_TIMEOUT_S = 5.0
MIRRORING_QUEUE = "inference/mirroring"


class QueueMissing(RuntimeError):
    pass

# ST3215 RAM registers (same as the repo's teleop example)
RAM_TORQUE_ENABLE = 0x28
RAM_ACC = 0x29
RAM_GOAL_POSITION = 0x2A
RAM_GOAL_SPEED = 0x2E
RAM_PRESENT_POSITION = 0x38
RAM_STATUS = 0x40
RAM_PRESENT_CURRENT = 0x45

# ElRobot joint names by motor id (SO-101 uses ids 1-6, 6 = gripper).
ELROBOT_JOINTS = {
    1: "base_yaw", 2: "shoulder", 3: "upper_arm_roll", 4: "elbow",
    5: "forearm_roll", 6: "wrist_pitch", 7: "wrist_roll", 8: "gripper",
}
GRIPPER_ID = int(os.environ.get("GRIPPER_MOTOR_ID", "8"))

# Transport: "stdio" (default, Claude Code spawns the server) or "streamable-http"
# (run it yourself, e.g. in a tmux pane to watch the logs, then
# `claude mcp add --transport http normacore http://127.0.0.1:8790/mcp`).
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8790"))

mcp = FastMCP("normacore-station", host=MCP_HOST, port=MCP_PORT, instructions=(
    "Controls a real robot arm through NormaCore Station. Always call `look` and "
    "`get_robot_state` before moving. Move in small steps, then `look` again to "
    "check the result. Positions are normalized 0..1 inside each joint's "
    "calibrated range. Call `relax` (torque off) if anything looks wrong."
))

_client = None
_client_lock = asyncio.Lock()


async def client():
    global _client
    async with _client_lock:
        if _client is None:
            _client = await new_station_client(STATION_HOST, log)
        return _client


# ----------------------------------------------------------------------------
# queue helpers

async def read_last(queue_id: str, n: int = 1) -> list[memoryview]:
    """Return up to the last `n` entries of a queue, oldest first."""
    c = await client()
    qr = c.read_from_tail(queue_id, offset=bytes([n]), limit=n, step=1, buf_size=n + 1)
    out: list[memoryview] = []
    deadline = time.monotonic() + READ_TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            entry = await asyncio.wait_for(qr.data.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if entry is None:
            break
        out.append(memoryview(bytes(entry.Data)))
    if not out and qr.err:
        if qr.err is ErrQueueNotFound or isinstance(qr.err, ErrQueueNotFound):
            raise QueueMissing(f"queue {queue_id} not found")
        raise RuntimeError(f"reading {queue_id}: {qr.err}")
    return out


def _u16(b: bytes, a: int) -> int:
    return struct.unpack_from("<H", b, a)[0] if len(b) >= a + 2 else 0


def _u8(b: bytes, a: int) -> int:
    return b[a] if len(b) > a else 0


def _pos(raw: int) -> int:
    if raw & 0x8000:
        return (4096 - (raw & 0x0FFF)) & 0x0FFF
    return raw & 0x0FFF


# Position <-> normalized mapping, ported from Station's own mirroring code so our
# numbers match what teleop does. Ranges may WRAP past 4095 -> 0 (range_max < range_min).
MAX_STEPS = 4096           # software/drivers/motors-mirroring/src/config.rs (max_steps: 4096)
MIN_CALIBRATED_RANGE = 100  # station-viewer/src/pages/St3215BusCalibrationPage.tsx (MIN_CALIBRATED_RANGE)


def range_size(rmin: int, rmax: int) -> int:
    # motors-mirroring/src/inference/normalize.rs get_steps_range
    return rmax - rmin if rmax >= rmin else (MAX_STEPS - rmin) + rmax


def to_norm(pos: int, rmin: int, rmax: int) -> float:
    # motors-mirroring/src/inference/normalize.rs normalize_position, scaled to 0..1
    size = range_size(rmin, rmax)
    if size == 0:
        return 0.5
    if rmax >= rmin:
        rel = pos - rmin if rmin <= pos <= rmax else (0 if pos < rmin else size)
    elif pos >= rmin:
        rel = pos - rmin
    elif pos <= rmax:
        rel = (MAX_STEPS - rmin) + pos
    else:
        rel = 0 if (rmin - pos) < (pos - rmax) else size
    return min(1.0, max(0.0, rel / size))


def from_norm(norm: float, rmin: int, rmax: int) -> int:
    # motors-mirroring/src/inference/mirror.rs:96-98 (without its safety_margin)
    norm = min(1.0, max(0.0, norm))
    return (rmin + int(round(norm * range_size(rmin, rmax)))) & (MAX_STEPS - 1)


async def read_buses() -> dict[str, dict]:
    entries = await read_last("st3215/inference", 1)
    if not entries:
        raise RuntimeError("no data in st3215/inference: is an arm plugged in and powered?")
    state = st3215.InferenceStateReader(entries[-1])
    buses: dict[str, dict] = {}
    for bus in state.get_buses() or []:
        info = bus.get_bus()
        serial = info.get_serial_number() if info else "?"
        motors = {}
        for m in bus.get_motors() or []:
            s = bytes(m.get_state())
            rmin, rmax = int(m.get_range_min()), int(m.get_range_max())
            pos = _pos(_u16(s, RAM_PRESENT_POSITION))
            calibrated = range_size(rmin, rmax) >= MIN_CALIBRATED_RANGE
            motors[int(m.get_id())] = {
                "joint": ELROBOT_JOINTS.get(int(m.get_id()), f"motor_{m.get_id()}"),
                "position_raw": pos,
                "position_norm": round(to_norm(pos, rmin, rmax), 3) if calibrated else None,
                "range_min": rmin,
                "range_max": rmax,
                "calibrated": calibrated,
                # Until frozen (web UI "Save"), Station widens the range whenever the joint reaches a
                # new position (st3215/src/port_meta.rs:349,407-423), so normalized values drift.
                "range_frozen": bool(m.get_range_freezed()),
                "torque_on": _u8(s, RAM_TORQUE_ENABLE) != 0,
                "current": _u16(s, RAM_PRESENT_CURRENT),
                "error_status": _u8(s, RAM_STATUS),
            }
        # stamp_ns: station monotonic time of this bus snapshot; the motion engine uses it
        # to detect stale state (it must keep advancing while we move).
        buses[serial] = {"motors": dict(sorted(motors.items())), "stamp_ns": int(bus.get_monotonic_stamp_ns())}
    return buses


def pick_bus(buses: dict[str, dict], requested: str | None) -> str:
    want = (requested or FOLLOWER_BUS or "").strip()
    if want:
        if want not in buses:
            raise RuntimeError(f"bus {want!r} not found; available: {list(buses)}")
        return want
    if len(buses) == 1:
        return next(iter(buses))
    raise RuntimeError(
        f"{len(buses)} ST3215 buses found {list(buses)}. Set FOLLOWER_BUS or pass bus= "
        "with the FOLLOWER arm's serial (never command the leader arm)."
    )


# ----------------------------------------------------------------------------
# mirroring (Station's built-in leader -> follower teleop)
# Facts from software/drivers/motors-mirroring/src/lib.rs + inference/mod.rs:
# - every entry in inference/mirroring carries the full current state (merge_modes);
# - `mirroring` = active source->targets links; `modes` = remembered roles, kept after stop;
# - START enables torque on targets; STOP disables torque on the freed buses (async).

async def mirroring_state() -> dict:
    """Current mirroring links. Raises on read errors (callers fail closed)."""
    try:
        entries = await read_last(MIRRORING_QUEUE, 1)
    except QueueMissing:
        return {"available": False, "links": [], "modes": {}}
    if not entries:
        return {"available": True, "links": [], "modes": {}}
    st = mirroring.RxEnvelopeReader(entries[-1]).get_state()
    links, modes = [], {}
    if st is not None:
        for b in st.get_modes() or []:
            bid = b.get_id()
            if bid is not None:
                modes[bid.get_unique_id()] = b.get_mode().name
        for link in st.get_mirroring() or []:
            src = link.get_source()
            leader = src.get_id().get_unique_id() if src and src.get_id() else "?"
            followers = [t.get_id().get_unique_id() for t in link.get_targets() or [] if t.get_id()]
            links.append({"leader": leader, "followers": followers})
    return {"available": True, "links": links, "modes": modes}


def _is_mirrored_follower(ms: dict, serial: str) -> str | None:
    for link in ms["links"]:
        if serial in link["followers"]:
            return link["leader"]
    return None


async def guard_motion(serial: str) -> None:
    """Refuse to command `serial` if it is a leader or is being mirrored. Fails closed."""
    ms = await mirroring_state()
    leader = _is_mirrored_follower(ms, serial)
    if leader is not None:
        raise RuntimeError(
            f"bus {serial} is being mirrored from leader {leader} (teach mode is on); two writers "
            "would fight over the arm. Call teach_mode(on=False) first.")
    is_leader = serial == LEADER_BUS or ms["modes"].get(serial) == "BR_LEADER" \
        or any(link["leader"] == serial for link in ms["links"])
    if is_leader:
        raise RuntimeError(f"bus {serial} is the LEADER arm; refusing to command it.")


def pick_leader(buses: dict[str, dict], follower: str, ms: dict) -> str:
    if LEADER_BUS:
        if LEADER_BUS not in buses:
            raise RuntimeError(f"LEADER_BUS {LEADER_BUS!r} not found; available: {list(buses)}")
        return LEADER_BUS
    known = [s for s, mode in ms["modes"].items() if mode == "BR_LEADER" and s in buses and s != follower]
    if len(known) == 1:
        return known[0]
    others = [s for s in buses if s != follower]
    if len(others) == 1:
        return others[0]
    raise RuntimeError(f"can't tell which bus is the leader among {others}; set LEADER_BUS.")


def _mirroring_command(kind, source: str, targets: list[str]):
    def mb(s):
        return mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=s)
    cmd = mirroring.Command(type=kind, source=mb(source), targets=[mb(t) for t in targets])
    return station_commands.DriverCommand(
        type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND, body=cmd.encode())


def _sync_write(bus: str, address: int, writes: list[tuple[int, bytes]]):
    cmd = st3215.Command(
        target_bus_serial=bus,
        sync_write=st3215.ST3215SyncWriteCommand(
            address=address,
            motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=i, value=v)
                    for i, v in writes],
        ),
    )
    return station_commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


async def camera_queues() -> list[str]:
    """Every camera queue Station has ever created (from the inference-states index).

    Station never deletes these: each replug of a camera can add a new one (see `cameras`).
    """
    entries = await read_last("inference-states", 1)
    if not entries:
        return []
    rx = inference.InferenceRxReader(entries[-1])
    return sorted({e.get_queue() for e in rx.get_entries() or []
                   if e.get_type() == drivers.QueueDataType.QDT_USB_VIDEO_FRAMES})


# ----------------------------------------------------------------------------
# camera identity: Station names queues usbvideo/<md5 of the camera's unique_id>
# (usbvideo/src/pipeline.rs generate_queue_id), and real queue ids carry a station
# prefix: "/<station id>/usbvideo/<md5>". On Linux unique_id is
# "<vid>:<pid>:<usb bus>:<usb device number>" (decimal), and the device number changes
# on every replug (seen on hardware: 13030:37416:1:25 -> :28 -> :35), so the same
# camera gets a new queue each time. On macOS it is AVFoundation's device.uniqueID
# (osx/avflib/avflib/avf.m:418). Roles are therefore saved against camera_id
# (USB vendor:product:serial), which survives replugs and port changes.
CAMERAS_FILE = Path(os.environ.get("CAMERAS_FILE", str(Path(__file__).resolve().parent / "data" / "cameras.json")))
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_CAM_GONE_EVENTS = (usbvideo.RxEnvelopeType.ET_DEVICE_DISCONNECTED,
                    # what Station writes last when a camera is unplugged ("Capture session ... ended")
                    usbvideo.RxEnvelopeType.ET_DEVICE_RECORDING_END)
_CAM_STARTING_EVENTS = (usbvideo.RxEnvelopeType.ET_DEVICE_CONNECTED,
                        usbvideo.RxEnvelopeType.ET_DEVICE_RECORDING_START)


def camera_id(cam, queue_id: str) -> str:
    """Stable id of the physical camera: "vid:pid:serial" (hex ids), or "vid:pid:product"
    when it has no serial (two identical serial-less cameras then share an id). Falls back
    to the queue id when Station reported no USB ids."""
    if cam is not None:
        vid, pid = int(cam.get_vendor_id() or 0), int(cam.get_product_id() or 0)
        if vid or pid:
            tail = (cam.get_serial_number() or "").strip() or (cam.get_product() or "").strip()
            return f"{vid:04x}:{pid:04x}:{tail}"
    return queue_id


def _queue_hash(queue_id: str) -> str:
    return queue_id.rsplit("usbvideo/", 1)[-1]


async def camera_snapshot(queue_id: str) -> dict:
    """Identity, newest frame and liveness of one camera queue.

    Returns {"queue", "product", "camera_id", "label", "jpeg" (None if no frame),
    "age_s" (None if unknown), "disconnected", "gone"}. `disconnected`: the newest entry
    is a disconnect / capture-end event. `gone`: the camera is not streaming any more
    (disconnected, no frame left in the queue, or newest frame older than CAMERA_GONE_S).
    """
    try:
        entries = await read_last(queue_id, 8)
    except RuntimeError:
        entries = []
    cam = None
    jpeg = age = None
    newest_kind = None
    for raw in reversed(entries):
        env = usbvideo.RxEnvelopeReader(raw)
        kind = env.get_type()
        if newest_kind is None:
            newest_kind = kind
        c = env.get_camera()
        if cam is None and c is not None and (c.get_product() or c.get_vendor_id()):
            cam = c
        if jpeg is None and kind == usbvideo.RxEnvelopeType.ET_FRAMES:
            pack = env.get_frames()
            frames = pack.get_frames_data() if pack else []
            if frames:
                jpeg, age = bytes(frames[-1]), _frame_age_s(env, pack)
        if cam is not None and jpeg is not None:
            break
    product = ((cam.get_product() if cam else "") or "").strip() or None
    disconnected = newest_kind in _CAM_GONE_EVENTS
    if jpeg is None:
        gone = newest_kind not in _CAM_STARTING_EVENTS  # a camera that is just starting has no frame yet
    else:
        gone = disconnected or (age is not None and age > CAMERA_GONE_S)
    return {"queue": queue_id, "product": product, "camera_id": camera_id(cam, queue_id),
            "label": f"{product} ({queue_id})" if product else queue_id,
            "jpeg": jpeg, "age_s": age, "disconnected": disconnected, "gone": gone}


async def cameras() -> tuple[list[dict], list[dict]]:
    """(connected cameras in queue order, gone cameras). Indexes in list_cameras/look are
    positions in the connected list, so they can shift when cameras come and go: use roles."""
    snaps = await asyncio.gather(*(camera_snapshot(q) for q in await camera_queues()))
    return [s for s in snaps if not s["gone"]], [s for s in snaps if s["gone"]]


def _env_roles() -> dict[str, str]:
    """CAMERAS="desk=32e6:9228:2026...,gripper=3f2a" (camera_id, queue id or queue hash prefix)."""
    out = {}
    for part in os.environ.get("CAMERAS", "").split(","):
        if "=" in part:
            role, ref = (s.strip() for s in part.split("=", 1))
            if role and ref:
                out[role.lower()] = ref
    return out


def _file_roles() -> dict[str, str]:
    try:
        data = json.loads(CAMERAS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable %s: %s", CAMERAS_FILE, e)
        return {}
    return {str(k).lower(): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _match_cameras(ref: str, cams: list[dict]) -> list[dict]:
    """Cameras matching a camera_id, a full queue id, or a queue hash prefix
    (with or without "usbvideo/" and the station prefix)."""
    ref = ref.strip()
    hits = [c for c in cams if ref in (c["camera_id"], c["queue"])]
    if hits:
        return hits
    h = _queue_hash(ref)
    return [c for c in cams if _queue_hash(c["queue"]).startswith(h)] if h else []


def _describe(cam: dict) -> str:
    return f"{cam['product'] or 'camera'} [{cam['camera_id']}]"


def camera_roles(cams: list[dict], gone: list[dict] = ()) -> tuple[dict[str, dict], list[str]]:
    """(role -> connected camera for roles that resolve, problems). data/cameras.json wins over CAMERAS env."""
    merged = {**_env_roles(), **_file_roles()}
    roles, problems = {}, []
    for role, ref in merged.items():
        hits = _match_cameras(ref, cams)
        if len(hits) == 1:
            roles[role] = hits[0]
        elif hits:
            problems.append(f"role {role!r} -> {ref!r} matches several connected cameras: "
                            f"{[c['queue'] for c in hits]}")
        else:
            seen = _match_cameras(ref, list(gone))
            what = f" ({_describe(seen[-1])} is unplugged)" if seen else ""
            problems.append(f"role {role!r} -> {ref!r} matches no connected camera{what}")
    by_cam: dict[str, list[str]] = {}
    for role, c in roles.items():
        by_cam.setdefault(c["queue"], []).append(role)
    for q, rs in by_cam.items():
        if len(rs) > 1:
            problems.append(f"camera {q} has several roles {sorted(rs)} (CAMERAS env vs data/cameras.json?)")
    return roles, problems


def _frame_age_s(env, pack) -> float | None:
    """Seconds since the frame was captured (station wall clock vs ours), or None if unknown."""
    stamps = pack.get_stamps() if pack else []
    stamp = stamps[-1] if stamps else env.get_stamp()
    local_ns = stamp.get_local_stamp_ns() if stamp else 0
    if not local_ns:
        return None
    return (time.time_ns() - local_ns) / 1e9


def _fmt_age(s: float) -> str:
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def frame_status(got: dict) -> tuple[str, bool]:
    """(text for the label, is_stale) for a camera_snapshot that has a frame."""
    age = got["age_s"]
    if got["disconnected"]:
        when = f", last frame {_fmt_age(age)} old" if age is not None and age >= 0 else ""
        return f"STALE: camera reported disconnected{when}", True
    if age is None:
        return "frame age unknown (no timestamp)", False
    if age < -FRAME_MAX_AGE_S:
        return (f"frame age unknown: its timestamp is {_fmt_age(-age)} in the future, "
                "so the station and MCP server clocks disagree"), False
    if age > FRAME_MAX_AGE_S:
        return f"STALE: last frame is {_fmt_age(age)} old (limit {FRAME_MAX_AGE_S:g} s)", True
    return f"frame {max(age, 0.0):.1f} s old", False


# ----------------------------------------------------------------------------
# tools

@mcp.tool()
async def look(camera: str = "all") -> list:
    """Grab the latest frame from the station cameras (gripper cam and desk cam).

    camera: "all" (default: every connected camera), a role ("desk", "gripper", set with
    `set_camera_role`), or an index / camera_id / queue id / hash prefix from `list_cameras`.
    Unplugged cameras are skipped by "all" and indexes, but can still be fetched by
    camera_id or queue id for diagnosis.
    Returns one labeled JPEG per camera. Each label says how old the frame is;
    a frame marked STALE is NOT the current scene, so never plan moves from it.
    """
    cams, gone = await cameras()
    roles, problems = camera_roles(cams, gone)
    role_of = {c["queue"]: r for r, c in roles.items()}
    selected: list[tuple[int | None, dict]] = list(enumerate(cams))  # keep each camera's index from list_cameras
    if camera != "all":
        key = camera.strip().lower()
        configured = {**_env_roles(), **_file_roles()}
        if key in roles:
            selected = [(i, c) for i, c in selected if c is roles[key]]
        elif key in configured:
            return [f"Camera role {key!r} can't be shown: "
                    + "; ".join(p for p in problems if p.startswith(f"role {key!r}"))
                    + ". Check the camera's USB connection, or re-assign the role with set_camera_role."]
        elif camera.isdigit():
            if int(camera) >= len(cams):
                return [f"No camera {camera}: only {len(cams)} connected ({[c['queue'] for c in cams]}). "
                        f"Use an index 0..{len(cams) - 1} or 'all'."]
            selected = [selected[int(camera)]]
        else:
            selected = [(i, c) for i, c in selected if camera in c["queue"] or camera in c["camera_id"]]
            if not selected:
                selected = [(None, c) for c in gone if camera in c["queue"] or camera in c["camera_id"]][-1:]
            if not selected:
                known = f" Known roles: {sorted(roles)}." if roles else " No camera roles set yet (set_camera_role)."
                return [f"No camera matching {camera!r}.{known} Use list_cameras."]
    if not selected:
        hidden = f" ({len(gone)} unplugged camera(s) hidden, see list_cameras(include_gone=True))" if gone else ""
        return [f"No cameras connected{hidden}. Check the USB hub and that usb-video is enabled in station.yaml."]
    out: list = [f"camera role problems: {problems}"] if problems else []
    for i, c in selected:
        tag = "unplugged camera" if i is None else f"camera {i}"
        if c["queue"] in role_of:
            tag += f" [{role_of[c['queue']]}]"
        if c["jpeg"] is None:
            out.append(f"{tag}: {c['label']} has no frames yet")
            continue
        status, stale = frame_status(c)
        text = f"{tag}: {c['label']} - {status}"
        if stale:
            text += (". This is NOT the current scene: don't plan or verify moves with it. "
                     "Check the camera's USB connection and the station log.")
            log.warning("look: camera %s %s", c["queue"], status)
        out.append(text)
        out.append(Image(data=c["jpeg"], format="jpeg"))
    return out


@mcp.tool()
async def list_cameras(include_gone: bool = False) -> dict:
    """List connected cameras: index, queue id, product name, camera_id (stable USB
    vendor:product:serial), role (if set) and frame status, plus role problems.

    Station keeps a queue for every camera it ever saw (a new one per replug on Linux);
    unplugged ones are hidden unless include_gone=True.
    """
    cams, gone = await cameras()
    roles, problems = camera_roles(cams, gone)
    role_of = {c["queue"]: r for r, c in roles.items()}
    out = {"cameras": [{"index": i, "queue": c["queue"], "product": c["product"], "camera_id": c["camera_id"],
                        "role": role_of.get(c["queue"]),
                        "frame": frame_status(c)[0] if c["jpeg"] is not None else "no frames yet"}
                       for i, c in enumerate(cams)],
           "role_problems": problems, "roles_file": str(CAMERAS_FILE), "unplugged_hidden": len(gone)}
    if include_gone:
        out["unplugged"] = [{"queue": c["queue"], "product": c["product"], "camera_id": c["camera_id"],
                             "last_frame_age": _fmt_age(c["age_s"]) if c["age_s"] is not None else None}
                            for c in gone]
    return out


@mcp.tool()
async def set_camera_role(role: str, camera: str) -> dict:
    """Name a camera, e.g. set_camera_role("desk", "1") or ("gripper", "32e6:9251:2025090142113").

    camera: index, camera_id, queue id or queue hash prefix from list_cameras; "none" removes the role.
    Identify cameras from the picture first (`look`): the gripper cam shows the jaws and its
    view moves with the arm; the desk cam is fixed and shows the whole workspace.
    Saved to data/cameras.json by camera_id, so the role survives replugging the camera.
    A camera holds at most one role.
    """
    role = role.strip().lower()
    if not _ROLE_RE.match(role) or role == "all" or role == "none":
        raise RuntimeError("role must be a short lowercase word like 'desk' or 'gripper' (not 'all'/'none')")
    saved = _file_roles()
    if camera.strip().lower() == "none":
        saved.pop(role, None)
        cam = None
    else:
        cams, _ = await cameras()
        if camera.isdigit() and int(camera) < len(cams):
            hits = [cams[int(camera)]]
        else:
            hits = _match_cameras(camera, cams)
        if len(hits) != 1:
            raise RuntimeError(f"{camera!r} matches no single camera; connected: "
                               f"{[(c['queue'], c['camera_id']) for c in cams]}")
        cam = hits[0]
        if len([c for c in cams if c["camera_id"] == cam["camera_id"]]) > 1:
            raise RuntimeError(f"two connected cameras share camera_id {cam['camera_id']!r} (identical models "
                               "without serial numbers), so a role can't tell them apart")
        saved = {r: ref for r, ref in saved.items() if not _match_cameras(ref, [cam])}  # one role per camera
        saved[role] = cam["camera_id"]
    CAMERAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMERAS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(CAMERAS_FILE)
    return {"role": role, "camera": cam["queue"] if cam else None,
            "camera_id": cam["camera_id"] if cam else None, "saved_roles": saved, "file": str(CAMERAS_FILE)}


@mcp.tool()
async def get_robot_state() -> dict:
    """Joint positions for every ST3215 bus on the station.

    position_norm is 0..1 within the calibrated range (None if uncalibrated).
    `follower_bus` is the bus that the move tools will command.
    """
    buses = await read_buses()
    try:
        follower = pick_bus(buses, None)
    except RuntimeError as e:
        follower = str(e)
    try:
        ms = await mirroring_state()
    except Exception as e:  # report, don't hide
        ms = f"unknown ({e})"
    warnings = []
    for serial, b in buses.items():
        loose = [m["joint"] for m in b["motors"].values() if m["calibrated"] and not m["range_frozen"]]
        if loose:
            warnings.append(f"bus {serial}: calibration not saved/frozen for {loose}; ranges will keep "
                            "changing and shift normalized positions. Press Save on the calibration page.")
    return {"follower_bus": follower, "mirroring": ms, "warnings": warnings, "buses": buses}


@mcp.tool()
async def move_joints(targets: dict[str, float], bus: str | None = None) -> dict:
    """Move joints to absolute normalized positions (0..1 in calibrated range).

    targets: {"<motor id or joint name>": position_norm}, e.g. {"1": 0.5, "gripper": 0.2}
    Each joint moves at most MAX_STEP_NORM per call (larger requests are clamped
    and reported), so call repeatedly and `look` in between.
    """
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    motors = buses[serial]["motors"]
    names = {v["joint"]: k for k, v in motors.items()}

    goals: list[tuple[int, bytes]] = []
    report = {}
    for key, want in targets.items():
        mid = int(key) if str(key).isdigit() else names.get(key)
        if mid is None or mid not in motors:
            report[key] = f"unknown joint; known: {list(names)}"
            continue
        m = motors[mid]
        if not m["calibrated"]:
            report[key] = "not calibrated - calibrate the arm in the station web UI first"
            continue
        if m["error_status"]:
            report[key] = f"servo error status {m['error_status']}, skipped"
            continue
        want = max(0.0, min(1.0, float(want)))
        cur = m["position_norm"]
        step = max(-MAX_STEP_NORM, min(MAX_STEP_NORM, want - cur))
        tgt = cur + step
        raw = from_norm(tgt, m["range_min"], m["range_max"])  # stays inside the (possibly wrapping) range
        goals.append((mid, raw.to_bytes(2, "little")))
        report[m["joint"]] = {"from": cur, "to": round(tgt, 3),
                              "clamped": abs(tgt - want) > 1e-3}

    if goals:
        ids = [g[0] for g in goals]
        await send_commands(await client(), [
            _sync_write(serial, RAM_GOAL_SPEED, [(i, MOVE_SPEED.to_bytes(2, "little")) for i in ids]),
            _sync_write(serial, RAM_ACC, [(i, bytes([30])) for i in ids]),
            _sync_write(serial, RAM_TORQUE_ENABLE, [(i, b"\x01") for i in ids]),
            _sync_write(serial, RAM_GOAL_POSITION, goals),
        ])
        await asyncio.sleep(0.4)
    return {"bus": serial, "result": report}


@mcp.tool()
async def nudge_joints(deltas: dict[str, float], bus: str | None = None) -> dict:
    """Move joints relative to where they are now (normalized units, e.g. {"elbow": -0.05})."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    motors = buses[serial]["motors"]
    names = {v["joint"]: k for k, v in motors.items()}
    targets = {}
    for key, d in deltas.items():
        mid = int(key) if str(key).isdigit() else names.get(key)
        if mid in motors and motors[mid]["position_norm"] is not None:
            targets[str(mid)] = motors[mid]["position_norm"] + float(d)
    return await move_joints(targets, serial)


@mcp.tool()
async def set_gripper(opening: float, bus: str | None = None) -> dict:
    """Open/close the gripper. opening: 0.0 = one end of its range, 1.0 = the other.
    Check with `look` which end is 'open' on your build."""
    return await move_joints({str(GRIPPER_ID): opening}, bus)


@mcp.tool()
async def relax(bus: str | None = None) -> str:
    """EMERGENCY / SAFE STOP: torque off on every motor of the follower bus (arm goes limp - support it!)."""
    _abort.set()  # also stop a running go_to/play_motion
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    ids = list(buses[serial]["motors"])
    await send_commands(await client(), [_sync_write(serial, RAM_TORQUE_ENABLE, [(i, b"\x00") for i in ids])])
    return f"torque disabled on {serial} motors {ids}"


async def _hold(serial: str, buses: dict[str, dict]) -> None:
    motors = buses[serial]["motors"]
    goals = [(i, m["position_raw"].to_bytes(2, "little")) for i, m in motors.items()]
    await send_commands(await client(), [
        _sync_write(serial, RAM_GOAL_POSITION, goals),
        _sync_write(serial, RAM_TORQUE_ENABLE, [(i, b"\x01") for i in motors]),
    ])


@mcp.tool()
async def hold(bus: str | None = None) -> str:
    """Torque on, holding the current pose (also stops a running go_to/play_motion)."""
    _abort.set()
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    await _hold(serial, buses)
    return f"holding pose on {serial}"


async def _wait_for(pred, timeout_s: float = 2.0, every_s: float = 0.1):
    deadline = time.monotonic() + timeout_s
    while True:
        result = await pred()
        if result or time.monotonic() > deadline:
            return result
        await asyncio.sleep(every_s)


@mcp.tool()
async def teach_mode(on: bool, force: bool = False) -> dict:
    """Turn Station's built-in leader -> follower mirroring on/off (for human demonstrations).

    on=True: the follower copies the leader arm. It JUMPS to the leader's pose immediately,
      so this refuses if any joint differs by more than TEACH_MAX_POSE_DIFF unless force=True.
      Move the leader close to the follower's pose first.
    on=False: stops mirroring. Station turns the follower's torque off when it does, so this
      waits for that and then holds the current pose. The arm may sag briefly; support it.
    While teach mode is on, all motion tools refuse to move the follower.
    """
    buses = await read_buses()
    follower = pick_bus(buses, None)
    ms = await mirroring_state()
    if not ms["available"]:
        raise RuntimeError(f"queue {MIRRORING_QUEUE} not found: is the motors-mirroring driver running in station?")
    current_leader = _is_mirrored_follower(ms, follower)

    if on:
        leader = pick_leader(buses, follower, ms)
        if current_leader == leader:
            return {"teach_mode": "already on", "leader": leader, "follower": follower}
        lm, fm = buses[leader]["motors"], buses[follower]["motors"]
        problems = {}
        for mid, f in fm.items():
            l = lm.get(mid)
            if l is None:
                problems[f["joint"]] = "missing on leader"
            elif not (l["calibrated"] and f["calibrated"]):
                problems[f["joint"]] = "not calibrated on leader and/or follower"
            elif abs(l["position_norm"] - f["position_norm"]) > TEACH_MAX_POSE_DIFF:
                problems[f["joint"]] = {"leader": l["position_norm"], "follower": f["position_norm"]}
        if problems and not force:
            return {"teach_mode": "refused", "reason": "follower would jump; align the leader first or pass force=True",
                    "problems": problems}
        await send_commands(await client(), [_mirroring_command(
            mirroring.CommandType.CT_START_MIRROR, leader, [follower])])

        async def linked():
            return _is_mirrored_follower(await mirroring_state(), follower) == leader
        ok = await _wait_for(linked)
        return {"teach_mode": "on" if ok else "sent, but not confirmed in inference/mirroring",
                "leader": leader, "follower": follower, "forced_past": problems or None}

    if current_leader is None:
        return {"teach_mode": "already off", "follower": follower}
    # Same shape the web UI sends to stop (station-viewer/src/st3215/BusCard.tsx): source = follower.
    await send_commands(await client(), [_mirroring_command(
        mirroring.CommandType.CT_STOP_MIRROR, follower, [])])

    async def unlinked_and_limp():
        if _is_mirrored_follower(await mirroring_state(), follower) is not None:
            return False
        motors = (await read_buses())[follower]["motors"]
        return all(not m["torque_on"] for m in motors.values())
    limp = await _wait_for(unlinked_and_limp)
    unlinked = _is_mirrored_follower(await mirroring_state(), follower) is None
    if not unlinked:
        return {"teach_mode": "stop sent, but mirroring still active", "follower": follower}
    await _hold(follower, await read_buses())
    return {"teach_mode": "off", "follower": follower, "holding": True,
            "note": None if limp else "didn't observe Station's torque-off within 2 s; held anyway"}


# ----------------------------------------------------------------------------
# fast playback engine: runs whole motions inside the server (~20 Hz), so Claude
# plans once and checks once instead of thinking between every small step.

MOTION_RATE_HZ = float(os.environ.get("MOTION_RATE_HZ", "20"))
MAX_VEL_NORM = float(os.environ.get("MAX_VEL_NORM", "0.5"))        # norm/s per joint; untuned, start low
APPROACH_VEL_NORM = float(os.environ.get("APPROACH_VEL_NORM", "0.3"))
MOTION_ACCEL = int(os.environ.get("MOTION_ACCEL", "50"))           # servo accel register; Station teleop uses 5..100
SETTLE_S = float(os.environ.get("SETTLE_S", "1.0"))
SETTLE_TOL = float(os.environ.get("SETTLE_TOL", "0.02"))
LIMITS_FILE = Path(os.environ.get("LIMITS_FILE", str(Path(__file__).resolve().parent / "data" / "limits.json")))

_motion_lock = asyncio.Lock()
_abort = asyncio.Event()


def _safety_config() -> M.SafetyConfig:
    per = {}
    for part in os.environ.get("CURRENT_LIMITS", "").split(","):   # e.g. "8:40,5:60" like station.yaml
        if ":" in part:
            k, v = part.split(":", 1)
            per[int(k)] = int(v)
    return M.SafetyConfig(
        current_limit=int(os.environ.get("CURRENT_LIMIT", "100")),
        per_motor_current=per, gripper_id=GRIPPER_ID,
        track_err_max=float(os.environ.get("TRACK_ERR_MAX", "0.10")),
        track_err_ticks=int(os.environ.get("TRACK_ERR_TICKS", "5")),
        stale_s=float(os.environ.get("STALE_STATE_S", "0.5")))


def _soft_limits() -> dict[int, tuple[float, float]]:
    """data/limits.json: {"2": [0.1, 0.9], ...} normalized soft limits per joint (optional)."""
    try:
        data = json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
        return {int(k): (float(v[0]), float(v[1])) for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, IndexError) as e:
        raise RuntimeError(f"{LIMITS_FILE} is invalid ({e}); fix it before moving") from e


def _resolve_joint_keys(d: dict, motors: dict) -> dict[int, float]:
    names = {v["joint"]: k for k, v in motors.items()}
    out = {}
    for key, val in d.items():
        mid = int(key) if str(key).isdigit() else names.get(str(key))
        if mid is None or mid not in motors:
            raise RuntimeError(f"unknown joint {key!r}; known: {list(names)}")
        out[mid] = val
    return out


async def _hold_shielded(serial: str) -> None:
    """Hold the current pose even if this tool call is being cancelled."""
    with anyio.CancelScope(shield=True):
        try:
            await _hold(serial, await read_buses())
        except Exception:
            log.exception("SAFETY: failed to hold %s after a stopped motion", serial)


async def _run_motion(waypoints: list[dict], speed_scale: float, bus: str | None) -> dict:
    if _motion_lock.locked():
        raise RuntimeError("another motion is running; call stop_motion first")
    async with _motion_lock:
        _abort.clear()
        buses = await read_buses()
        serial = pick_bus(buses, bus)
        await guard_motion(serial)
        motors = buses[serial]["motors"]
        bad = {m["joint"]: ("not calibrated" if not m["calibrated"] else f"error 0x{m['error_status']:02X}")
               for m in motors.values()
               if not m["calibrated"] or (m["error_status"] & ~M.IGNORED_STATUS_BITS)}
        if bad:
            raise RuntimeError(f"refusing to move: {bad}")
        ids = sorted(motors)
        current = {j: motors[j]["position_norm"] for j in ids}
        wps = [{**w, "joints": {str(k): v for k, v in _resolve_joint_keys(w.get("joints") or {}, motors).items()}}
               for w in waypoints]
        tl = M.build_timeline(current, wps, speed_scale=speed_scale, max_vel=MAX_VEL_NORM,
                              limits=_soft_limits(), approach_vel=APPROACH_VEL_NORM)
        ranges = {j: (motors[j]["range_min"], motors[j]["range_max"]) for j in ids}
        peak = tl.peak_velocity()
        speeds = [(j, M.steps_per_s(peak.get(j, 0.0), range_size(*ranges[j])).to_bytes(2, "little")) for j in ids]
        monitor = M.SafetyMonitor(_safety_config())
        c = await client()
        await send_commands(c, [
            _sync_write(serial, RAM_GOAL_SPEED, speeds),
            _sync_write(serial, RAM_ACC, [(j, bytes([MOTION_ACCEL])) for j in ids]),
            _sync_write(serial, RAM_TORQUE_ENABLE, [(j, b"\x01") for j in ids]),
        ])

        period = 1.0 / MOTION_RATE_HZ
        reason, ticks, late = None, 0, 0
        last_stamp, last_change = None, time.monotonic()
        gripper_hold = None
        latest = motors
        t0 = time.monotonic()
        try:
            while True:
                if _abort.is_set():
                    reason = "stopped by stop_motion/relax/hold"
                    break
                t = time.monotonic() - t0
                target = tl.sample(t)
                if monitor.gripper_blocked and GRIPPER_ID in target:
                    if gripper_hold is None:
                        gripper_hold = latest[GRIPPER_ID]["position_norm"]
                    target[GRIPPER_ID] = gripper_hold  # grasped: stop squeezing, like Station's mirroring does
                await send_commands(c, [_sync_write(serial, RAM_GOAL_POSITION, [
                    (j, from_norm(target[j], *ranges[j]).to_bytes(2, "little")) for j in ids])])
                state = (await read_buses())[serial]
                latest = state["motors"]
                if state["stamp_ns"] != last_stamp:
                    last_stamp, last_change = state["stamp_ns"], time.monotonic()
                reason = monitor.check(latest, target, time.monotonic() - last_change)
                if reason:
                    break
                ticks += 1
                if t >= tl.duration:
                    break
                delay = t0 + ticks * period - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    late += 1
            settled = None
            if reason is None:
                deadline = time.monotonic() + SETTLE_S
                final = tl.sample(tl.duration)
                while True:
                    latest = (await read_buses())[serial]["motors"]
                    errs = {j: abs(latest[j]["position_norm"] - final[j]) for j in ids
                            if not (j == GRIPPER_ID and monitor.gripper_blocked)}
                    settled = max(errs.values(), default=0.0) <= SETTLE_TOL
                    if settled or time.monotonic() > deadline:
                        break
                    await asyncio.sleep(period)
        except BaseException:
            await _hold_shielded(serial)
            raise
        if reason:
            await _hold_shielded(serial)
            log.warning("motion aborted: %s", reason)
        elapsed = time.monotonic() - t0
        return {
            "completed": reason is None,
            "abort_reason": reason,
            "bus": serial,
            "planned_s": round(tl.duration, 2),
            "elapsed_s": round(elapsed, 2),
            "ticks": ticks,
            "late_ticks": late,
            "settled": settled,
            "grasp_detected": monitor.gripper_blocked,
            "stretched_segments": tl.stretched_segments,
            "clamped_targets": tl.clamped or None,
            "max_tracking_error": {motors[j]["joint"]: round(e, 3) for j, e in monitor.max_track_err.items()},
            "final": {motors[j]["joint"]: latest[j]["position_norm"] for j in ids},
        }


@mcp.tool()
async def go_to(targets: dict[str, float], duration_s: float = 0.0, speed_scale: float = 1.0,
                bus: str | None = None) -> dict:
    """FAST single move: glide to absolute normalized positions in one smooth motion.

    targets: {"<motor id or joint name>": 0..1}. duration_s=0 picks the fastest time allowed
    by MAX_VEL_NORM; a too-short duration is stretched, never exceeded. Runs inside the server
    at ~20 Hz with safety checks (servo errors, over-current = collision, tracking error,
    stale data); on any problem it stops and holds. Gripper over-current counts as a grasp.
    Use move_joints/nudge_joints for careful single steps while learning.
    """
    return await _run_motion([{"joints": targets, "duration_s": max(0.0, duration_s)}], speed_scale, bus)


@mcp.tool()
async def play_motion(waypoints: list[dict], speed_scale: float = 1.0, bus: str | None = None) -> dict:
    """FAST playback of a whole motion in one call (the "reflex" path).

    waypoints, either form:
      keyframes: [{"joints": {"shoulder": 0.3, "8": 0.9}, "duration_s": 1.5}, ...]
        (eased segments; joints not listed keep their value)
      dense recording: [{"t": 0.00, "joints": {...}}, {"t": 0.05, ...}, ...]
        (played with original timing after a gentle approach to the first sample)
    speed_scale: 1.0 = as recorded/specified; 1.5 = 50% faster (velocity cap still applies).
    Returns completed / abort_reason / grasp_detected / max_tracking_error / final pose.
    Check the result with `look` once at the end, not between waypoints.
    """
    return await _run_motion(waypoints, speed_scale, bus)


@mcp.tool()
async def stop_motion() -> str:
    """Stop a running go_to/play_motion: the arm holds where it is.
    (Claude Code runs one tool at a time, so this is mainly for a second client such as an
    e-stop script; hold/relax also stop a running motion.)"""
    _abort.set()
    return "stop requested; a running motion will hold its current pose"


if __name__ == "__main__":
    log.info("starting MCP server (transport=%s, station=%s)", MCP_TRANSPORT, STATION_HOST)
    mcp.run(transport=MCP_TRANSPORT)
