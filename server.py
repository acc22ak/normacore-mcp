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
  LEADER_BUS        ST3215 bus serial of the leader arm: never commanded; used by the optional
                    teach_mode. If unset and the station has exactly two buses, the other one.
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
import learn as L  # noqa: E402
import estop as E  # noqa: E402  (latch protocol only; the e-stop itself runs as its own process)

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

import activity as A  # noqa: E402


class ObservedFastMCP(FastMCP):
    """FastMCP that reports the connected client and every tool call to the viewer's agent panel
    (activity.py, GET /events). Reporting never changes a call's result."""

    def _note_client(self) -> None:
        try:
            info = self.get_context().session.client_params.clientInfo
            A.note_client(info.name, info.version)
        except Exception:  # no request context / older client: the panel just shows no client name
            pass

    async def list_tools(self):
        self._note_client()
        return await super().list_tools()

    async def call_tool(self, name, arguments):
        self._note_client()
        call_id = A.call_started(name, arguments)
        ok = False
        try:
            result = await super().call_tool(name, arguments)
            ok = A.result_ok(result)
            return result
        finally:
            A.call_finished(call_id, ok)


mcp = ObservedFastMCP("normacore-station", host=MCP_HOST, port=MCP_PORT, instructions=(
    "Controls a real robot arm (ElRobot follower) through NormaCore Station, with a calibrated depth "
    "camera looking at the table.\n"
    "PICK AND PLACE, the normal workflow:\n"
    "1. find_objects (arm at home): numbered objects with positions, and for each one `reach.how`, the "
    "call to use for it.\n"
    "2. pick(object_id) for a standing object in easy reach. For flat, small or far objects: "
    "pick(object_id, close=False), look at the zoomed photo, nudge_jaws if the object isn't between the "
    "jaw tips, then close_and_lift.\n"
    "3. place(x, z) or place_near(object_id).\n"
    "4. home once the whole task is done.\n"
    "RULES: After a miss do NOT go home: lift_and_look, then try again. If a photo doesn't clearly show "
    "where the object is, lift_and_look instead of guessing. Objects are always approached from above "
    "(the tools do this; don't work around them). Never guess joint angles: move_joints, nudge_joints, "
    "go_to and play_motion are for calibration and recovery, not for reaching objects. `hold` freezes "
    "the arm; `relax` cuts torque and the arm DROPS, so it is for emergencies only.\n"
    "COORDINATES: metres, robot base at the origin, +x forward (toward the depth camera), y up, +z to "
    "the robot's right (the LEFT of the depth camera's image). Joint positions are normalized 0..1 inside "
    "each joint's calibrated range."
))

_client = None
_client_lock = asyncio.Lock()


async def client():
    global _client
    async with _client_lock:
        if _client is None:
            _client = await new_station_client(STATION_HOST, log)
        return _client


async def _reconnect() -> None:
    """Drop the Station connection so the next read opens a fresh one."""
    global _client
    async with _client_lock:
        old, _client = _client, None
    if old is not None:
        try:
            old.close_connection()
        except Exception as e:  # the old connection is already broken: nothing to save
            log.warning("closing the old Station connection: %s", e)


# ----------------------------------------------------------------------------
# queue helpers

async def read_last(queue_id: str, n: int = 1, timeout: float = READ_TIMEOUT_S, step: int = 1) -> list[memoryview]:
    """Return up to the last `n` entries of a queue, oldest first; with step > 1, Station
    sends only every step-th of them (ceil(n / step) entries)."""
    c = await client()
    limit = -(-n // step)
    # Tail offsets count from 0 = the newest entry (verified on a real Station: offset 0 -> [e2],
    # offset 1 -> [e1, e2]). So the last n entries start at offset n - 1; offset n with limit 1
    # would return the second-newest entry, i.e. state one write behind (critical for estop/state).
    qr = c.read_from_tail(queue_id, offset=L.encode_shift(max(n - 1, 0)), limit=limit, step=step, buf_size=limit + 1)
    out: list[memoryview] = []
    deadline = time.monotonic() + timeout
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


_last_reading: dict[tuple[str, int], dict] = {}


async def read_buses() -> dict[str, dict]:
    entries = await read_last("st3215/inference", 1)
    if not entries:
        # The Station connection can go dead without an error: every read then comes back empty for
        # good while a fresh connection works (seen twice: overnight, and 70 s into a tool call on
        # 2026-11-22, which left the arm holding mid-motion). Reconnect once before giving up.
        log.warning("st3215/inference read came back empty: reconnecting to Station")
        await _reconnect()
        entries = await read_last("st3215/inference", 1)
    if not entries:
        raise RuntimeError("no data in st3215/inference: is an arm plugged in and powered?")
    buses = parse_inference_state(entries[-1])
    # A motor whose read just failed keeps its last real reading (marked stale_reading), so one dropped
    # packet doesn't look like a joint jumping to the end of its range to the motion checks or the IK.
    for serial, bus in buses.items():
        for mid, m in bus["motors"].items():
            if m["has_reading"]:
                _last_reading[(serial, mid)] = m
            elif (serial, mid) in _last_reading:
                bus["motors"][mid] = {**_last_reading[(serial, mid)], "stale_reading": True}
    return buses


def parse_inference_state(raw) -> dict[str, dict]:
    """One st3215/inference entry -> {bus serial: {"motors": {...}, "stamp_ns": ...}}."""
    state = st3215.InferenceStateReader(raw)
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
                # False when this motor's last read failed: Station then sends an error and an empty
                # register dump, which would otherwise parse as position 0 (state.rs: state = envelope.data).
                "has_reading": len(s) >= RAM_PRESENT_CURRENT + 2,
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


async def estop_latch() -> dict | None:
    """The e-stop latch (estop.py) from Station's estop/state queue: a dict if latched, else None.
    A missing queue means no e-stop was ever pressed. Read errors raise (callers fail closed)."""
    try:
        entries = await read_last(E.QUEUE, 1)
    except QueueMissing:
        return None
    return E.decode_latch(entries[-1]) if entries else None


async def guard_motion(serial: str) -> None:
    """Refuse to command `serial` if the e-stop is latched, or it is a leader or is being
    mirrored. Fails closed. (hold goes through here too: after a RELAX e-stop, someone may be
    holding the limp arm, so re-enabling torque must wait for a human release.)"""
    latch = await estop_latch()
    if latch:
        raise RuntimeError(E.refusal(latch))
    ms = await mirroring_state()
    leader = _is_mirrored_follower(ms, serial)
    if leader is not None:
        raise RuntimeError(
            f"bus {serial} is being mirrored from leader {leader} (Station teleop is on); two writers "
            "would fight over the arm. Call teach_mode(on=False), or ask the human to turn mirroring off "
            "on the follower's bus card in the Station web UI (http://localhost:8889).")
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
    """Grab the latest frame from Station's USB cameras (the gripper camera, and a desk camera if one is
    plugged in). The depth camera is NOT one of these: use find_objects / lift_and_look for the table view.

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
    """LOW-LEVEL (calibration / recovery; to reach objects use pick, nudge_jaws, place).
    Move joints to absolute normalized positions (0..1 in calibrated range).

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
    """LOW-LEVEL (calibration / recovery; to shift the jaws near an object use nudge_jaws).
    Move joints relative to where they are now (normalized units, e.g. {"elbow": -0.05})."""
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
    """Open/close the gripper by hand: 1.0 = fully open, 0.0 = fully closed (0.8 is the normal open width).
    Moves at most MAX_STEP_NORM per call. For grasping use pick / close_and_lift, which stop on contact."""
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


async def _wait_settled(serial: str, timeout_s: float = 3.0, tol: float = 0.003, still_s: float = 0.3):
    """Wait until every joint of `serial` stays within tol for still_s. Returns (buses, settled)."""
    deadline = time.monotonic() + timeout_s
    ref, since = None, time.monotonic()
    while True:
        buses = await read_buses()
        pose = {j: (m["position_norm"] or 0.0) for j, m in buses[serial]["motors"].items()}
        if ref is None or any(abs(pose[j] - ref.get(j, pose[j])) > tol for j in pose):
            ref, since = pose, time.monotonic()
        elif time.monotonic() - since >= still_s:
            return buses, True
        if time.monotonic() > deadline:
            return buses, False
        await asyncio.sleep(0.05)


@mcp.tool()
async def teach_mode(on: bool, force: bool = False) -> dict:
    """OPTIONAL: drive the follower with the leader arm (Station's built-in leader -> follower mirroring).

    Use it to move the robot by hand, or to show a task. Nothing else to record: turning it on
    starts a learning attempt, so after a human shows a task, call save_attempt_as_skill(name).
    on=True: the follower JUMPS to the leader's pose at once, so this refuses if any joint differs by
      more than TEACH_MAX_POSE_DIFF unless force=True (move the leader close to the follower first).
      The attempt starts once the follower has settled after that jump.
    on=False: marks the end of the demonstration, stops mirroring, waits for Station's torque-off and
      holds the pose (the brief sag is not part of the demonstration). Support the arm.
    While teach mode is on, all other motion tools refuse to move the follower.
    """
    latch = await estop_latch()
    if latch:
        raise RuntimeError(E.refusal(latch))
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
        if not await _wait_for(linked):
            return {"teach_mode": "sent, but not confirmed in inference/mirroring", "leader": leader,
                    "follower": follower, "forced_past": problems or None}
        settled_buses, settled = await _wait_settled(follower)
        attempt = {"task": "demonstration with the leader arm (teach mode)", "bus": follower, "source": "teach",
                   "since_ns": settled_buses[follower]["stamp_ns"], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_json(ATTEMPT_FILE, attempt)
        return {"teach_mode": "on", "leader": leader, "follower": follower, "forced_past": problems or None,
                "attempt": "started: the human can show the task now; afterwards teach_mode(on=False), then "
                           "save_attempt_as_skill(name)",
                "settled_before_start": settled}

    if current_leader is None:
        return {"teach_mode": "already off", "follower": follower}
    try:  # end of the demonstration = now, before Station's torque-off makes the arm sag
        attempt = json.loads(ATTEMPT_FILE.read_text(encoding="utf-8"))
        if attempt.get("source") == "teach" and "until_ns" not in attempt:
            attempt["until_ns"] = buses[follower]["stamp_ns"]
            _save_json(ATTEMPT_FILE, attempt)
    except (FileNotFoundError, ValueError):
        pass
    # Same shape the web UI sends to stop (station-viewer/src/st3215/BusCard.tsx): source = follower.
    await send_commands(await client(), [_mirroring_command(
        mirroring.CommandType.CT_STOP_MIRROR, follower, [])])

    async def unlinked_and_limp():
        if _is_mirrored_follower(await mirroring_state(), follower) is not None:
            return False
        motors = (await read_buses())[follower]["motors"]
        return all(not m["torque_on"] for m in motors.values())
    limp = await _wait_for(unlinked_and_limp)
    if _is_mirrored_follower(await mirroring_state(), follower) is not None:
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
SERVO_SPEED_HEADROOM = float(os.environ.get("SERVO_SPEED_HEADROOM", "1.2"))
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
        # Servo speed cap a little above the planned peak, so it never flattens the smooth curve.
        speeds = [(j, M.steps_per_s(SERVO_SPEED_HEADROOM * peak.get(j, 0.0), range_size(*ranges[j])).to_bytes(2, "little")) for j in ids]
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
        estop_mode = None
        t0 = time.monotonic()
        # The motion follows its own clock, which never advances more than two periods per tick. On a
        # loaded machine (battery throttling, a busy browser) ticks arrive late; following the wall clock
        # then makes the goal jump ahead and the arm lunge at full servo speed with nothing checking it
        # (2026-11-22: it ran into the object while the engine had done only a handful of ticks). With
        # this clock a slow machine gives a slower motion instead.
        clock, last_tick = 0.0, t0
        try:
            while True:
                if _abort.is_set():
                    reason = "stopped by stop_motion/relax/hold"
                    break
                latch = await estop_latch()
                if latch:
                    estop_mode = latch["mode"]
                    reason = f"E-STOP {estop_mode.upper()} pressed by {latch.get('by') or '?'}"
                    break
                now = time.monotonic()
                clock += min(now - last_tick, 2 * period)
                last_tick = now
                t = clock
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
            if estop_mode != "relax":      # after a RELAX e-stop the arm must stay limp
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
    """LOW-LEVEL (calibration / recovery / recording a skill; to reach objects use pick and place).
    FAST single move: glide to absolute normalized positions in one smooth motion.

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
        (smooth: flows through keyframes, rests where a joint turns around or at "stop": true;
        joints not listed keep their value)
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


# ----------------------------------------------------------------------------
# learning: record a demonstration from Station's history -> skill -> practice ladder

SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", str(Path(__file__).resolve().parent / "data" / "skills")))
RUNS_FILE = Path(os.environ.get("RUNS_FILE", str(Path(__file__).resolve().parent / "data" / "runs.jsonl")))
# Most entries one history read transfers. Station writes st3215/inference at ~400 entries/s
# (measured on hardware, 2 buses), so longer spans are read with a server-side stride instead
# of being cut short.
HISTORY_MAX_ENTRIES = int(os.environ.get("HISTORY_MAX_ENTRIES", "20000"))
HISTORY_TIMEOUT_S = float(os.environ.get("HISTORY_TIMEOUT_S", "30"))
_SKILL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# Which end of the gripper range is open (normalized 1.0 or 0.0) for {"gripper": "open"/"close"}
# skill steps. Unset until someone checks it on the arm with set_gripper + look.
GRIPPER_OPEN_NORM = os.environ.get("GRIPPER_OPEN_NORM", "").strip()
RAW_RANGE_SLACK = 20  # steps; same as Station's mirroring deadband
_pending_runs: dict[str, dict] = {}


def _skill_path(name: str) -> Path:
    if not _SKILL_RE.match(name):
        raise RuntimeError("skill name: lowercase letters, digits, '-' or '_' (e.g. 'pick_a')")
    return SKILLS_DIR / f"{name}.json"


def _load_skill(name: str) -> dict:
    p = _skill_path(name)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        known = sorted(f.stem for f in SKILLS_DIR.glob("*.json")) if SKILLS_DIR.exists() else []
        raise RuntimeError(f"no skill {name!r}; known: {known}") from None


def _save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def _log_run(record: dict) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _entry_stamp(state: dict[str, dict]) -> int:
    return max((b["stamp_ns"] for b in state.values()), default=0)


async def _read_history(serial_hint: str | None, seconds: float | None = None, hz: float = 20.0,
                        since_ns: int | None = None, until_ns: int | None = None) -> tuple[str, list[tuple], dict]:
    """Follower samples (t, norm, raw) from the last `seconds` of st3215/inference, or (since_ns)
    everything since a station stamp, e.g. the one start_attempt saved.

    Reads with a server-side stride so a long span stays fast (one entry per ~1/(2*hz) s), and
    grows the read until the stamps cover `seconds`. Fewer seconds come back only when Station's
    history itself is shorter (reported as a warning); otherwise it raises.
    """
    probe = [parse_inference_state(r) for r in await read_last("st3215/inference", 200)]
    if not probe:
        raise RuntimeError("no data in st3215/inference")
    serial = pick_bus(probe[-1], serial_hint)
    rate = L.estimate_rate_hz([b[serial]["stamp_ns"] for b in probe if serial in b]) or 20.0
    if since_ns is not None:
        newest_probe = max(b[serial]["stamp_ns"] for b in probe if serial in b)
        seconds = max(1.0, (newest_probe - since_ns) / 1e9 + 1.0)
    entry_rate = L.estimate_rate_hz([_entry_stamp(b) for b in probe]) or rate  # all buses share the queue
    if len({_entry_stamp(b) for b in probe}) < len(probe):  # several entries per stamp: count entries, not stamps
        span = (_entry_stamp(probe[-1]) - _entry_stamp(probe[0])) / 1e9
        entry_rate = (len(probe) - 1) / span if span > 0 else entry_rate
    step = max(1, int(entry_rate / (2 * hz)))
    back = int(seconds * entry_rate * 1.1) + 50
    covered, exhausted = 0.0, False
    for _ in range(4):
        step = max(step, -(-back // HISTORY_MAX_ENTRIES))
        limit = -(-back // step)
        entries = await read_last("st3215/inference", back, timeout=HISTORY_TIMEOUT_S, step=step)
        snaps, seen = [], set()
        for raw in entries:
            bus = parse_inference_state(raw).get(serial)
            if bus is None or bus["stamp_ns"] in seen:
                continue
            seen.add(bus["stamp_ns"])
            snaps.append(bus)
        snaps.sort(key=lambda b: b["stamp_ns"])
        if not snaps:
            raise RuntimeError(f"no history for bus {serial}")
        covered = (snaps[-1]["stamp_ns"] - snaps[0]["stamp_ns"]) / 1e9
        exhausted = len(entries) < limit  # reached the start of Station's history
        if covered >= seconds - 0.5 or exhausted:
            break
        back = int(back * seconds / max(covered, 0.1) * 1.1) + 50
    else:
        raise RuntimeError(f"could only read {covered:.1f} s of the requested {seconds:g} s of history "
                           f"(last read: {len(entries)} entries, step {step}); try fewer seconds")
    newest = snaps[-1]["stamp_ns"]
    snaps = [b for b in snaps if b["stamp_ns"] >= newest - int(seconds * 1e9)]
    if since_ns is not None:
        snaps = [b for b in snaps if b["stamp_ns"] >= since_ns]
    if until_ns is not None:
        snaps = [b for b in snaps if b["stamp_ns"] <= until_ns]
    if (since_ns is not None or until_ns is not None) and not snaps:
        raise RuntimeError("no robot history within the attempt (start_attempt / teach mode)")
    uncal = sorted({m["joint"] for b in snaps for m in b["motors"].values() if m["position_norm"] is None})
    if uncal:
        raise RuntimeError(f"joints {uncal} were uncalibrated during the recording; calibrate + Save first")
    base = snaps[0]["stamp_ns"]
    samples = [((b["stamp_ns"] - base) / 1e9,
                {j: m["position_norm"] for j, m in b["motors"].items()},
                {j: m["position_raw"] for j, m in b["motors"].items()}) for b in snaps]
    covered_s = round((newest - base) / 1e9, 2)
    info = {"entries_read": len(entries), "read_step": step, "rate_hz": round(rate, 1),
            "covered_s": covered_s, "requested_s": seconds}
    if since_ns is not None:
        info["requested_s"] = "since start_attempt"
        if base > since_ns + int(0.5e9):
            info["note"] = f"Station's history starts {(base - since_ns) / 1e9:.1f} s after start_attempt"
    elif covered_s < seconds - 0.5:
        info["note"] = f"Station's history only goes back {covered_s:g} s"
    return serial, samples, {"info": info, "last": snaps[-1]}


DEMO_PATH_TOL = float(os.environ.get("DEMO_PATH_TOL", "0.01"))   # norm; how closely a demo's path is kept
SKILL_BASE_VEL = float(os.environ.get("SKILL_BASE_VEL", "0.25"))  # norm/s peak at speed_scale 1.0
ATTEMPT_FILE = Path(os.environ.get("ATTEMPT_FILE", str(Path(__file__).resolve().parent / "data" / "attempt.json")))


@mcp.tool()
async def start_attempt(task: str, bus: str | None = None) -> dict:
    """OPTIONAL (replayable skills, for a task repeated with objects on fixed, marked spots; ordinary
    pick-and-place needs none of this). SELF-LEARNING step 1: mark the start of your own attempt at a task.

    Then solve the task yourself, carefully: look with both cameras, move with go_to /
    move_joints / nudge_joints / set_gripper, and verify with look. Pause (just think) between
    moves: every pose where the arm rests becomes a candidate keyframe. When the task is done
    AND verified on camera, call save_attempt_as_skill. If it failed, call start_attempt again.
    """
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    attempt = {"task": task, "bus": serial, "since_ns": buses[serial]["stamp_ns"],
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _save_json(ATTEMPT_FILE, attempt)
    return {**attempt, "next": "solve the task slowly with the cameras, verify success, then save_attempt_as_skill"}


@mcp.tool()
async def save_attempt_as_skill(name: str, description: str = "", overwrite: bool = False) -> dict:
    """SELF-LEARNING step 2: turn your successful attempt (since start_attempt) into a fast skill.

    Keeps only the poses where the arm rested, merges poses that lie on a straight line
    (gripper open/close moments are always kept), and times each move from SKILL_BASE_VEL.
    The thinking time and the step-by-step crawl disappear, so a minutes-long attempt becomes
    a skill of a few seconds. Only call this after verifying with `look` that the task succeeded.
    """
    try:
        attempt = json.loads(ATTEMPT_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError("no attempt in progress: call start_attempt(task) before solving the task") from None
    path = _skill_path(name)
    old = None
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"skill {name!r} exists; pass overwrite=True to save this attempt as a new version")
        old = _load_skill(name)
    until = attempt.get("until_ns")
    serial, samples, meta = await _read_history(attempt["bus"], since_ns=int(attempt["since_ns"]),
                                                until_ns=int(until) if until is not None else None)
    demo = attempt.get("source") == "teach"
    if demo:
        # A human with the leader arm moves along a path without stopping to think, so keep the
        # path's shape: sample it every 0.1 s and merge only near-straight stretches.
        rest = L.decimate(samples, 0.1)
        poses = L.simplify_path(rest, GRIPPER_ID, tol=DEMO_PATH_TOL)
    else:
        # Claude rests while it thinks: those rest poses are the poses it chose.
        rest = L.pause_keyframes(samples)
        poses = L.simplify_path(rest, GRIPPER_ID)
    if len(poses) < 2 or not L.has_motion(poses):
        raise RuntimeError("the arm didn't move since start_attempt: nothing to learn")
    keyframes = L.timed_keyframes(poses, SKILL_BASE_VEL)
    skill_s = round(sum(k["duration_s"] for k in keyframes), 2)
    last = meta["last"]["motors"]
    version = (old or {}).get("version", 0) + 1
    skill = {
        "name": name,
        "description": description or attempt["task"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bus": serial,
        "source": {"type": "demo" if demo else "self", "task": attempt["task"], "attempt_s": meta["info"]["covered_s"],
                   "rest_poses": len(rest), "kept_poses": len(poses), "skill_s_at_1x": skill_s,
                   "history": meta["info"]},
        "calibration": {"frozen": all(m["range_frozen"] for m in last.values()),
                        "ranges": {str(j): [m["range_min"], m["range_max"]] for j, m in last.items()}},
        "start_pose": {str(j): v for j, v in poses[0][0].items()},
        "start_raw": {str(j): int(v) for j, v in poses[0][1].items()},
        "keyframes": keyframes,
        "practice": {"level": 0, "streak": 0, "speed_scale": L.SPEED_LEVELS[0]},
        "status": "new",
        "version": version,
        "stats": {"runs": 0, "success": 0},
        "changelog": (old or {}).get("changelog", []) + [{"version": version,
                     "note": f"{'learned from a leader-arm demonstration' if demo else 'self-learned'}: "
                             f"{meta['info']['covered_s']} s -> {len(poses)} poses"}],
    }
    _save_json(path, skill)
    ATTEMPT_FILE.unlink(missing_ok=True)
    return {"saved": str(path), "source": "demo" if demo else "self", "attempt_s": meta["info"]["covered_s"],
            "history": meta["info"], "rest_poses": len(rest),
            "kept_poses": len(poses), "skill_s": {str(s): round(skill_s / s, 2) for s in (0.5, 1.0, 2.0)},
            "warning": None if skill["calibration"]["frozen"] else "calibration not saved/frozen: press Save in the web UI",
            "next": ("teach_mode(on=False) first, then " if demo and until is None else "") +
                    "reset the scene (or learn the reverse task), then run_skill(name) and report_skill_result"}


@mcp.tool()
async def list_skills() -> list[dict]:
    """Saved skills with status, practice speed, success stats and duration."""
    out = []
    for p in sorted(SKILLS_DIR.glob("*.json")) if SKILLS_DIR.exists() else []:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            out.append({"name": p.stem, "error": f"unreadable: {e}"})
            continue
        out.append({"name": s.get("name", p.stem), "description": s.get("description"),
                    "status": s.get("status"), "version": s.get("version"),
                    "speed_scale": (s.get("practice") or {}).get("speed_scale"),
                    "streak": (s.get("practice") or {}).get("streak"), "stats": s.get("stats"),
                    "source": (s.get("source") or {}).get("type"),
                    "duration_s": (s.get("source") or {}).get("duration_s") or (s.get("source") or {}).get("skill_s_at_1x")})
    return out


def _raw_in_range(pos: int, rmin: int, rmax: int, slack: int = RAW_RANGE_SLACK) -> bool:
    """Is a raw encoder position inside a (possibly wrapping) calibrated range, +- slack steps?"""
    return (pos - (rmin - slack)) % MAX_STEPS <= range_size(rmin, rmax) + 2 * slack


def _gripper_target(value, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{where}: gripper value {value} must be within 0..1")
        return float(value)
    if value in ("open", "close"):
        if GRIPPER_OPEN_NORM not in ("0", "1", "0.0", "1.0"):
            raise RuntimeError(f"{where}: {{'gripper': {value!r}}} needs GRIPPER_OPEN_NORM (1 or 0): check with "
                               "set_gripper + look which end of the range is open, or use a number 0..1")
        open_norm = float(GRIPPER_OPEN_NORM)
        return open_norm if value == "open" else 1.0 - open_norm
    raise RuntimeError(f"{where}: gripper must be 'open', 'close' or a number 0..1, not {value!r}")


def _skill_waypoints(skill: dict, motors: dict) -> tuple[list[dict], dict[int, str], dict | None]:
    """Waypoints for play_motion, the joints whose calibration changed, and (if changed) the
    rebased skill data to store after a successful run.

    If calibration changed since recording (>20 steps), the normalized targets are rebuilt from
    the recorded RAW positions with today's ranges (CLAUDE.md §6 rule 2). A raw position outside
    today's range refuses the run instead of being clamped to the range edge.
    """
    current = {j: (m["range_min"], m["range_max"]) for j, m in motors.items()}
    changed = L.ranges_changed((skill.get("calibration") or {}).get("ranges", {}), current)
    outside: dict[str, int] = {}

    def rebuild(p: dict, joints: dict) -> dict:
        if not changed or "raw" not in p:
            return joints
        joints = dict(joints)
        for k, raw in p["raw"].items():
            j = int(k)
            if j not in current or k not in joints:
                continue
            if not _raw_in_range(int(raw), *current[j]):
                name = motors[j]["joint"]
                outside[name] = outside.get(name, 0) + 1
                continue
            joints[k] = round(to_norm(int(raw), *current[j]), 4)
        return joints

    if "motion" in skill:
        field, points = "motion", [dict(p, joints=rebuild(p, p["joints"])) for p in skill["motion"]]
        wps = [{"t": p["t"], "joints": p["joints"]} for p in points]
    elif "keyframes" in skill:  # hand-written skill: [{"joints"/"gripper": ..., "duration_s": ...}]
        field, points, wps = "keyframes", [], []
        if "start_pose" in skill:  # self-learned: glide to where the attempt began, at the skill's pace
            start = rebuild({"raw": skill.get("start_raw", {})}, dict(skill["start_pose"]))
            dist = max((abs(float(v) - (motors[int(k)]["position_norm"] or 0.0)) for k, v in start.items()
                        if int(k) in motors), default=0.0)
            wps.append({"joints": start, "duration_s": max(0.2, M.EASED_PEAK * dist / SKILL_BASE_VEL)})
        composite = [i for i, k in enumerate(skill["keyframes"]) if "skill" in k]
        if composite:
            raise RuntimeError(f"keyframes {composite}: composite steps {{'skill': ...}} aren't supported by "
                               "run_skill yet; run the sub-skills one after another")
        for i, k in enumerate(skill["keyframes"]):
            where = f"keyframe {i}" + (f" ({k['label']})" if k.get("label") else "")
            joints = rebuild(k, k.get("joints") or {})
            if "gripper" in k:
                joints = {**joints, str(GRIPPER_ID): _gripper_target(k["gripper"], where)}
            if not joints:
                raise RuntimeError(f"{where} has neither 'joints' nor 'gripper'")
            points.append(dict(k, joints=joints) if "joints" in k else k)
            wps.append({"joints": joints, "duration_s": k.get("duration_s", 1.0)})
    else:
        raise RuntimeError("skill has neither 'motion' nor 'keyframes'")
    if outside:
        raise RuntimeError(f"calibration changed since this skill was recorded, and recorded positions are outside "
                           f"today's range for {outside} (samples per joint). Re-record the skill, or re-calibrate "
                           "so the range covers them. Not clamping: that would move to the range edge instead.")
    rebased = None
    if changed:
        rebased = {"field": field, "points": points,
                   **({"start_pose": wps[0]["joints"]} if field == "keyframes" and "start_pose" in skill else {}),
                   "calibration": {"frozen": all(m["range_frozen"] for m in motors.values()),
                                   "ranges": {str(j): [lo, hi] for j, (lo, hi) in current.items()}}}
    return wps, changed, rebased


@mcp.tool()
async def run_skill(name: str, speed_scale: float | None = None, bus: str | None = None) -> dict:
    """Run a saved skill FAST in one call, at its practice speed (or speed_scale if given).

    Afterwards `look` once and call report_skill_result(run_id, success): 2 successes in a row
    raise the practice speed one level (0.5, 0.75, 1.0, 1.25, ...), a failure lowers it one
    level. If the engine aborts (collision, error, ...), that already counts as a failure.
    If calibration changed since recording, positions are rebuilt from raw values and the
    run is capped at 0.5 (status needs_check).
    """
    skill = _load_skill(name)
    buses = await read_buses()
    serial = pick_bus(buses, bus or skill.get("bus"))
    wps, changed, rebased = _skill_waypoints(skill, buses[serial]["motors"])
    practice = skill.get("practice") or {"level": 0, "streak": 0, "speed_scale": L.SPEED_LEVELS[0]}
    practice_speed = float(practice.get("speed_scale", L.SPEED_LEVELS[0]))
    speed, speed_note = practice_speed, None
    if speed_scale is not None:
        if speed_scale <= 0:
            raise RuntimeError("speed_scale must be > 0")
        speed = min(float(speed_scale), L.SPEED_LEVELS[-1])
        if skill.get("status") != "ok" and speed > practice_speed:
            # CLAUDE.md §5: no faster replay before a watched success at practice speed
            speed_note = (f"speed_scale {speed_scale:g} capped to the practice speed {practice_speed:g}: "
                          f"status is {skill.get('status')!r}, it must succeed at practice speed first")
            speed = practice_speed
    if changed:
        speed = min(speed, L.SPEED_LEVELS[0])
        skill["status"] = "needs_check"
    report = await _run_motion(wps, speed, serial)
    run_id = os.urandom(4).hex()
    record = {"run_id": run_id, "skill": name, "version": skill.get("version"), "speed_scale": speed,
              "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine": {k: report[k] for k in
              ("completed", "abort_reason", "elapsed_s", "planned_s", "grasp_detected", "max_tracking_error")},
              "calibration_changed": {str(k): v for k, v in changed.items()} or None}
    if not report["completed"]:
        _finish_run(skill, record, success=False, note=f"engine aborted: {report['abort_reason']}")
        return {**report, "run_id": run_id, "speed_scale": speed, "speed_note": speed_note,
                "counted_as": "failure", "practice": skill["practice"]}
    _pending_runs[run_id] = {"skill": skill, "record": record, "rebased": rebased}
    _save_json(_skill_path(name), skill)  # persists a needs_check status
    return {**report, "run_id": run_id, "speed_scale": speed, "speed_note": speed_note,
            "calibration_changed": record["calibration_changed"],
            "next": "look at the result, then report_skill_result(run_id, success, note)"}


def _finish_run(skill: dict, record: dict, success: bool, note: str) -> None:
    skill["practice"] = L.ladder_update(skill.get("practice") or {}, success)
    stats = skill.setdefault("stats", {"runs": 0, "success": 0})
    stats["runs"] = stats.get("runs", 0) + 1
    stats["success"] = stats.get("success", 0) + (1 if success else 0)
    if skill.get("status") != "needs_check" or success:
        skill["status"] = "ok" if success else "flaky"
    _save_json(_skill_path(skill["name"]), skill)
    _log_run({**record, "success": success, "note": note, "practice_after": skill["practice"]})


@mcp.tool()
async def report_skill_result(run_id: str, success: bool, note: str = "") -> dict:
    """Judge a finished run_skill from what you saw with `look` (did the task actually succeed?).

    Updates the practice ladder (2 successes in a row -> faster, failure -> slower), the
    skill's stats/status, and appends the run to data/runs.jsonl.
    """
    pending = _pending_runs.pop(run_id, None)
    if pending is None:
        raise RuntimeError(f"unknown or already reported run_id {run_id!r}")
    skill = _load_skill(pending["skill"]["name"])  # re-read: the file may have been edited meanwhile
    rebased = pending.get("rebased")
    if success and rebased:
        # The replay from raw positions worked with today's calibration: store it (CLAUDE.md §6 rule 2),
        # so later runs don't rebuild again and the practice speed is no longer capped.
        old_version = skill.get("version", 1)
        skill["last_good"] = {"version": old_version, rebased["field"]: skill.get(rebased["field"]),
                              "calibration": skill.get("calibration")}
        skill[rebased["field"]] = rebased["points"]
        if "start_pose" in rebased:
            skill["last_good"]["start_pose"] = skill.get("start_pose")
            skill["start_pose"] = rebased["start_pose"]
        skill["calibration"] = rebased["calibration"]
        skill["version"] = old_version + 1
        skill.setdefault("changelog", []).append({
            "version": skill["version"],
            "note": f"re-based on new calibration after a successful replay from raw (run {run_id}); "
                    f"changed joints: {pending['record']['calibration_changed']}"})
    _finish_run(skill, pending["record"], success, note)
    return {"skill": skill["name"], "success": success, "status": skill["status"],
            "practice": skill["practice"], "stats": skill["stats"],
            "rebased_to_new_calibration": bool(success and rebased)}


# =============================================================================
# Depth camera + kinematics: find objects and pick/place them in one call each.
# Uses the RealSense bridge (realsense-bridge/) and its calibration file, written by the station-viewer
# page (Level to table + Fit to arm shape), plus a Python port of the viewer's ElRobot kinematics.
# Frame everywhere: station-viewer 3D view = metres, robot base at origin, Y up, arm faces +x.
# =============================================================================
import numpy as np  # noqa: E402
import arm_model as AM  # noqa: E402
import depth as D  # noqa: E402

URDF_PATH = Path(os.environ.get("ELROBOT_URDF", str(
    REPO / "software/station/clients/station-viewer/public/devices/elrobot/elrobot_follower.urdf")))
REALSENSE_URL = os.environ.get("REALSENSE_URL", "ws://127.0.0.1:8890")
DEPTH_FRAMES = int(os.environ.get("DEPTH_FRAMES", "5"))   # find_objects / depth_point use the median of this many
REALSENSE_CALIBRATION = Path(os.environ.get("REALSENSE_CALIBRATION", str(
    Path(__file__).resolve().parent / "realsense-bridge" / "calibration.json")))
# Shoulder past ~0.9 leans the extended arm so far it is very slow to lift back (home() managed from 0.99).
# Forearm roll and wrist roll: hard cable limit (camera cable loops round the wrist), +-0.25 of straight.
# Before 2026-11-22 the solver only had a soft pull on these, so nothing stopped it winding the cable.
IK_LIMITS = {2: (0.01, 0.90), 5: (0.29, 0.79), 7: (0.29, 0.79)}
IK_TILT_WEIGHT = 0.5          # how strongly IK holds the chosen jaw angle
IK_APPROACH_Y = -0.5          # fallback angle (30 deg below horizontal) for moves without a plan
# Grasp from above: steepest jaw angle the arm can reach at the object (vertical reaches ~32 cm from the
# base, -0.9 ~36 cm, -0.8 ~38 cm, -0.7 ~41 cm). Shallow angles "scoop" and push small things away.
GRASP_TILTS = (-1.0, -0.9, -0.8, -0.7, -0.5)
GRASP_CURRENT = int(os.environ.get("GRASP_CURRENT", "12"))   # gripper current that means "holding" (empty ~2)
GRIPPER_OPEN = 0.80
GRIPPER_FLOOR = 0.12          # closing below this = nothing between the jaws
JAW_GAP_MIN_M, JAW_GAP_MAX_M = 0.035, 0.062  # open jaws are ~5 cm apart: another gap = tips misread
PLACE_OVER_MARGIN_M = 0.03     # held object bottom above a `clear_m` obstacle while carrying
PLACE_CLEAR_Y = 0.15         # tip height (m above the table) place() rises to after letting go
TRAVEL_CLEARANCE_M = 0.08     # tip height above the grasp height while approaching / carrying
JAW_MIN_Y = 0.005             # never put the jaw ends lower than this above the table
MIN_GRASP_TIP_Y = 0.03        # place() without a pick: grasp-point height to release at
TIP_TOL_M = 0.004
WRIST_ROLL_NORM_PER_DEG = 1 / 343.0   # wrist_roll range 98..3998 raw of 4096 steps = ~343 deg
WRIST_ROLL_MAX_DEG = 85.0             # camera cable: stay within ~±90 deg of straight
GRASP_REACH_M = 0.04          # aim the point 4 cm past the tip along the jaws (~1.5 cm back from their ends)
HOME_POSE = {1: 0.512, 2: 0.014, 3: 0.503, 4: 0.943, 5: 0.544, 6: 0.464, 7: 0.544, 8: 0.547}
ROLL_IDS = (3, 5, 7)
# Extra IK starting poses: from an awkward current pose the solver can stall far from the target
# (3 of 30 random starts in an offline test), so it retries from these and keeps the best answer.
REACH_POSE = {1: 0.512, 2: 0.46, 3: 0.503, 4: 0.46, 5: 0.544, 6: 0.61, 7: 0.544}
IK_SEEDS = (HOME_POSE, REACH_POSE)
DESCENT_STEPS = 3             # the descent runs through this many straight-line points (joint-space drift was ~6 mm)

# Where the real jaws are, sideways, compared with the arm model. Measured 2026-11-22 on three saved
# calibration frames: the midpoint between the real jaw tips sat 19-21 mm from the model's, along the jaw
# closing direction (the model's "centre" was on one jaw tip). pick measures it live with the depth camera
# and saves the latest value here; this file is the fallback when the jaws can't be seen.
JAW_OFFSET_FILE = Path(os.environ.get("JAW_OFFSET_FILE", str(Path(__file__).resolve().parent / "data" / "jaw_offset.json")))
JAW_CORRECT_MAX_M = 0.035     # a bigger sideways gap than this is a wrong detection, not a real offset
JAW_CORRECT_TOL_M = 0.004
JAW_LOOK_MAX_DIR_Y = -0.7     # the jaw tips are only measured with the jaws pointing down at least this steeply

# How far short (or long) the real jaws land along the reach direction, by distance from the base. The arm
# bends and the joint zeros are a few degrees uncertain, and the depth camera can't see reach on the jaws
# head-on (2026-11-22: two picks landed 2-3 cm short and pushed the object away). calibrate_reach measures
# it from the side, where reach is across the camera's view, and pick/place read the result from here.
REACH_MAP_FILE = Path(os.environ.get("REACH_MAP_FILE", str(Path(__file__).resolve().parent / "data" / "reach_map.json")))
REACH_PROBE_JAW_Y = 0.03      # probes keep the jaw ends this far above the table
REACH_CORRECT_MAX_M = 0.06    # a bigger reach error than this is a wrong measurement


def _jaw_low_y(top: float) -> float:
    """Where the jaw ends should go for an object `top` m tall: ~30% of its height (a standing minifigure:
    1.3 cm; anything flat: JAW_MIN_Y). The old fixed tip height kept the jaws 1.5 cm above flat objects."""
    return max(JAW_MIN_Y, 0.3 * top)

# Gripper tag: a white-bordered AprilTag 16h5 (id 3, 4.0 cm black square) on top of the wrist camera.
# 16h5 has few bits, so other things can decode as it: always match the id.
MARKER_DICT = os.environ.get("GRIPPER_MARKER_DICT", "DICT_APRILTAG_16h5")
MARKER_ID = 3
MARKER_SIZE_CM = 4.0
MARKER_FILE = Path(os.environ.get("GRIPPER_MARKER_FILE", str(Path(__file__).resolve().parent / "data" / "gripper_marker.json")))
MARKER_CORRECT_MIN_M = 0.005   # ignore marker/model disagreements smaller than this (detection noise)
MARKER_CORRECT_MAX_M = 0.06    # bigger than this: probably a wrong detection, don't trust it

_arm_model: AM.ArmModel | None = None
_last_pose: dict[str, dict[int, float]] = {}   # last read pose per bus, as an IK start for planning
_last_objects: list[dict] = []
_held: dict | None = None


class MotionFailed(RuntimeError):
    def __init__(self, step: str, result: dict):
        super().__init__(f"{step}: {result.get('abort_reason')}")
        self.step, self.result = step, result


def _calibration() -> dict:
    """Camera pose + arm model corrections. `arm_fit` (written by calibrate_depth_camera: camera pose, true
    joint scales and pitch-joint zero offsets fitted together) wins over the station-viewer's Align result."""
    cal = D.load_calibration(REALSENSE_CALIBRATION)
    fit = cal.get("arm_fit")
    if fit and fit.get("camera_pose"):
        cal = {**cal, "camera_pose": fit["camera_pose"],
               "joint_offsets": {**(cal.get("joint_offsets") or {}), fit["serial"]: fit["joint_offsets"]},
               "joint_spans": {fit["serial"]: fit.get("joint_spans", {})}}
    return cal


def _model(serial: str) -> AM.ArmModel:
    global _arm_model
    if _arm_model is None:
        _arm_model = AM.ArmModel.load(URDF_PATH)
    try:
        cal = _calibration()
        _arm_model.offsets = (cal.get("joint_offsets") or {}).get(serial, {})
        _arm_model.spans = (cal.get("joint_spans") or {}).get(serial, {})
    except (OSError, ValueError):
        pass  # keep the last known offsets
    return _arm_model


def _norm_pose(motors: dict) -> dict[int, float]:
    return {i: m["position_norm"] for i, m in motors.items() if m["position_norm"] is not None}


def _move_duration(cur: dict[int, float], tgt: dict[int, float]) -> float:
    """Seconds for a joint move, from what the arm managed on hardware (data/exploration.md):
    lifting the extended arm with the shoulder and folding the elbow up are the slow, heavy moves."""
    d = 1.2
    for j, v in tgt.items():
        delta = v - cur.get(j, v)
        if j == 2 and delta < 0:
            k = 30.0 if cur.get(2, 0.0) > 0.4 else 15.0
        elif j == 2:
            k = 12.0      # leaning out: 0.37 -> 0.72 in 4 s fell behind (tracking abort)
        elif j == 4 and delta > 0:
            k = 12.0
        elif j == GRIPPER_ID:
            k = 4.0
        elif j in ROLL_IDS:
            k = 12.0
        else:
            k = 8.0
        d = max(d, abs(delta) * k)
    return min(d, 12.0)


async def _go(serial: str, targets: dict[int, float], step: str, duration: float | None = None) -> dict:
    motors = (await read_buses())[serial]["motors"]
    d = duration if duration is not None else _move_duration(_norm_pose(motors), targets)
    res = await _run_motion([{"joints": {str(k): float(v) for k, v in targets.items()}, "duration_s": d}], 1.0, serial)
    if not res["completed"]:
        raise MotionFailed(step, res)
    return res


async def _tip_now(serial: str, settle: bool = True) -> tuple[np.ndarray, dict]:
    buses = (await _wait_settled(serial, timeout_s=1.5))[0] if settle else await read_buses()
    motors = buses[serial]["motors"]
    return _model(serial).tip(_norm_pose(motors)), motors


def _wrist_roll_target(deg: float) -> float:
    deg = max(-WRIST_ROLL_MAX_DEG, min(WRIST_ROLL_MAX_DEG, deg))
    return HOME_POSE[7] + deg * WRIST_ROLL_NORM_PER_DEG


def _ik(serial: str, target, start: dict[int, float], approach_y: float | None,
        fixed: dict[int, float] | None = None, tol: float = 0.004) -> AM.IkResult:
    """IK for the point between the jaws. Tries from `start`, then from IK_SEEDS, and keeps the closest
    answer. approach_y None = no tilt goal (position only)."""
    best = None
    for seed in (start, *IK_SEEDS):
        s = dict(seed) | {GRIPPER_ID: start.get(GRIPPER_ID, 0.5)}
        ik = _model(serial).solve_ik(target, s, point_down=IK_TILT_WEIGHT if approach_y is not None else 0.0,
                                     approach_y=approach_y if approach_y is not None else IK_APPROACH_Y,
                                     limits=IK_LIMITS, reach=GRASP_REACH_M, fixed=fixed)
        if best is None or ik.error < best.error:
            best = ik
        if ik.error < tol:
            break
    return best


async def _move_tip(serial: str, target, step: str, *, corrections: int = 2, gripper: float | None = None,
                    wrist_roll_deg: float | None = None, approach_y: float = IK_APPROACH_Y,
                    steps: int = 1) -> dict:
    """Move the point between the jaws to `target` (world, m): IK -> go_to, then compare with where the
    joint readings put it and re-aim, up to `corrections` times.

    The reported model_error_cm is NOT the real error: it is computed from the encoders through the same
    arm model that planned the move, so it only shows servos stopping short. A wrong model, a bent arm or
    a camera calibration error don't show up in it (the jaw look in pick is what catches those).
    steps > 1: the first move goes through that many points on the straight line to the target, so a
    descent comes down vertically instead of along a joint-space curve.
    """
    target = np.asarray(target, dtype=float)
    aim = target.copy()
    tries = []
    fixed = {7: _wrist_roll_target(wrist_roll_deg)} if wrist_roll_deg is not None else None
    for attempt in range(corrections + 1):
        motors = (await read_buses())[serial]["motors"]
        start = _norm_pose(motors)
        m = _model(serial)
        here = m.grasp_point(start, GRASP_REACH_M)
        n = steps if attempt == 0 and np.linalg.norm(aim - here) > 0.02 else 1
        path, prev = [], start
        for k in range(1, n + 1):
            point = here + (aim - here) * (k / n)
            # IK is CPU work (up to seconds near the edge of reach): keep it off the event loop, or the
            # Station connection starves and goes dead (2026-11-22).
            ik = await asyncio.to_thread(_ik, serial, point, prev, approach_y, fixed)
            if ik.error > 0.008:     # the tilt goal is soft: at the edge of reach, position wins
                plain = await asyncio.to_thread(_ik, serial, point, prev, None, fixed)
                if plain.error < ik.error:
                    ik = plain
            if ik.error > 0.015:
                raise RuntimeError(f"{step}: {np.round(target, 3).tolist()} is out of reach "
                                   f"(closest the arm gets is {ik.error * 100:.1f} cm away)")
            tgt = dict(ik.positions)
            if gripper is not None:
                tgt[GRIPPER_ID] = gripper
            path.append(tgt)
            prev = start | tgt
        await _go_path(serial, path, step)
        motors = (await _wait_settled(serial, timeout_s=1.5))[0][serial]["motors"]
        tip = _model(serial).grasp_point(_norm_pose(motors), GRASP_REACH_M)
        err = target - tip
        tries.append(round(float(np.linalg.norm(err)) * 100, 1))
        if np.linalg.norm(err) < TIP_TOL_M:
            break
        aim = aim + err
    jaw_y = float(_model(serial).approach_dir(_norm_pose(motors))[1])
    return {"tip": np.round(tip, 4).tolist(), "model_error_cm": tries[-1], "tries_model_error_cm": tries,
            "jaw_angle_deg_below_horizontal": round(float(np.degrees(np.arcsin(-max(-1.0, min(1.0, jaw_y))))), 0)}


async def _go_path(serial: str, path: list[dict[int, float]], step: str) -> dict:
    """Run joint targets one after another in a single motion."""
    cur = _norm_pose((await read_buses())[serial]["motors"])
    waypoints = []
    for tgt in path:
        waypoints.append({"joints": {str(k): float(v) for k, v in tgt.items()},
                          "duration_s": _move_duration(cur, tgt)})
        cur = cur | tgt
    res = await _run_motion(waypoints, 1.0, serial)
    if not res["completed"]:
        raise MotionFailed(step, res)
    return res


def _reach_error(distance: float) -> float:
    """Real minus model reach (m) at this distance from the base, from calibrate_reach (0 without a map).
    Negative = the real jaws land short. Interpolated between probes, held flat outside them."""
    try:
        pts = sorted((float(p["r_m"]), float(p["error_m"])) for p in json.loads(REACH_MAP_FILE.read_text())["points"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    if not pts:
        return 0.0
    return float(np.interp(distance, [p[0] for p in pts], [p[1] for p in pts]))


def _reach_shift(x: float, z: float) -> np.ndarray:
    """Horizontal world vector (real minus model) of the reach error at (x, z): aim the model at
    `target - shift` and the real jaws land on the target."""
    r = float(np.hypot(x, z))
    if r < 1e-6:
        return np.zeros(3)
    return np.array([x, 0.0, z]) / r * _reach_error(r)


def _stored_jaw_offset() -> float:
    """Sideways offset of the real jaw midpoint in the gripper frame (m, along the closing direction)."""
    try:
        return float(json.loads(JAW_OFFSET_FILE.read_text(encoding="utf-8"))["local_x_m"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _save_jaw_offset(local_x: float, info: dict) -> None:
    try:
        old = json.loads(JAW_OFFSET_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    recent = (old.get("recent_local_x_m") or [])[-9:] + [round(local_x, 4)]
    data = {"local_x_m": round(float(np.median(recent)), 4), "recent_local_x_m": recent,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "last": info,
            "note": "real jaw midpoint minus the arm model's, along the jaw closing direction; "
                    "measured by pick with the depth camera (median of the last 10)"}
    JAW_OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    JAW_OFFSET_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def _jaw_shift(serial: str, live: bool = True) -> tuple[np.ndarray, dict]:
    """How far the real jaw midpoint is from the model's right now, as a horizontal world vector
    (real minus model). Aim the model point at `object - shift` and the real jaws end up over the object.

    Live: the depth camera finds the two jaw tips (jaws open, pointing down). Only the sideways part as the
    camera sees it is used: along the camera's viewing direction one camera can't place thin jaw tips to
    better than 1-2 cm (two methods disagreed by that much on the saved frames), sideways they agreed to
    2 mm. Fallback: the last saved offset, turned into the world frame at the current pose.
    """
    motors = (await _wait_settled(serial, timeout_s=1.5))[0][serial]["motors"]
    norm = _norm_pose(motors)
    m = _model(serial)
    T = m.end_transform(norm)
    info: dict = {"source": "none"}
    if live and m.approach_dir(norm)[1] <= JAW_LOOK_MAX_DIR_Y:
        try:
            cal = _calibration()
            frame = await D.fetch_frame(REALSENSE_URL)
            jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
            found = D.find_jaw_tips(frame, cal["camera_pose"], jaw_end)
            if found and not JAW_GAP_MIN_M <= found["gap_m"] <= JAW_GAP_MAX_M:
                # 2026-11-22: a 3.0 cm "gap" (dark clothes behind the jaws) gave a 3.1 cm shift and a miss.
                info = {"source": "none", "why": f"jaw gap {found['gap_m'] * 100:.1f} cm implausible: tips misread"}
            elif found:
                fwd = D.pose_matrix(cal["camera_pose"])[:3, :3] @ np.array([0.0, 0.0, -1.0])
                side = np.array([-fwd[2], 0.0, fwd[0]])
                side /= np.linalg.norm(side)
                amount = float((np.array(found["midpoint"]) - jaw_end) @ side)
                if abs(amount) <= JAW_CORRECT_MAX_M:
                    shift = side * amount
                    local_x = float((T[:3, :3].T @ shift)[0])
                    info = {"source": "camera", "sideways_cm": round(amount * 100, 1),
                            "jaw_gap_cm": round(found["gap_m"] * 100, 1), "local_x_cm": round(local_x * 100, 1)}
                    return shift, info
                info = {"source": "none", "why": f"jaw detection {amount * 100:.1f} cm off: ignored"}
            else:
                info = {"source": "none", "why": "jaw tips not found in the depth camera image"}
        except Exception as e:  # camera bridge down, no calibration: fall back to the saved offset
            info = {"source": "none", "why": f"{type(e).__name__}: {e}"}
    if live and "why" not in info:
        info = {"source": "none", "why": "jaws not pointing down steeply enough for the camera to find the tips"}
    local_x = _stored_jaw_offset()
    if local_x:
        shift = T[:3, :3] @ np.array([local_x, 0.0, 0.0])
        shift[1] = 0.0
        info = {**info, "source": "saved", "local_x_cm": round(local_x * 100, 1)}
        return shift, info
    return np.zeros(3), info


VERIFY_TOL_M = 0.008          # jaws this close to centred on the object: fine, close
VERIFY_ABORT_M = 0.015        # still further off than this after one correction: don't close
VERIFY_FULL_MIN_ELEV_DEG = 45.0   # below this camera look-down angle only the sideways error is trusted


async def _jaw_error_at_object(serial: str, object_xz: np.ndarray) -> dict:
    """With the open jaws down at an object: how far is the real midpoint between the jaw tips from where
    it should be for the object to sit between them (world x-z, m)? {"seen": False, "why"} when the camera
    can't tell. The tips are taken from the color image and placed on the plane at the jaw-end height;
    with the camera looking down less steeply than 45 deg, only the part across its view is kept (the
    part along its view moves ~2 cm per cm of unknown jaw height at 25 deg)."""
    motors = (await _wait_settled(serial, timeout_s=1.5))[0][serial]["motors"]
    norm = _norm_pose(motors)
    m = _model(serial)
    T = m.end_transform(norm)
    direction = m.approach_dir(norm)
    if direction[1] > -0.45:
        return {"seen": False, "why": "jaws too flat for the camera to find the tips"}
    jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
    try:
        cal = _calibration()
        frame = await D.fetch_frame(REALSENSE_URL)
    except Exception as e:
        return {"seen": False, "why": f"{type(e).__name__}: {e}"}
    _save_frame(frame, norm, "verify")
    found = D.find_jaw_tips(frame, cal["camera_pose"], jaw_end, plane_y=max(float(jaw_end[1]), 0.003))
    if not found:
        return {"seen": False, "why": "jaw tips not found in the image"}
    if not 0.035 <= found["gap_m"] <= 0.062:
        return {"seen": False, "why": f"jaw gap read as {found['gap_m'] * 100:.1f} cm: tips misread"}
    # The jaw ends sit past the aim point along the jaws; their midpoint should be that far beyond the object.
    past = (AM.JAW_END_M - AM.TIP_LOCAL[1] - GRASP_REACH_M) * direction[[0, 2]]
    err = (np.asarray(object_xz, dtype=float) + past) - np.array(found["midpoint"])[[0, 2]]
    Rc = D.pose_matrix(cal["camera_pose"])[:3, :3]
    fwd = Rc @ np.array([0.0, 0.0, -1.0])
    elev = float(np.degrees(np.arcsin(min(1.0, abs(fwd[1])))))
    full = elev >= VERIFY_FULL_MIN_ELEV_DEG
    if not full:
        side = np.array([-fwd[2], fwd[0]])
        side /= np.linalg.norm(side)
        err = side * float(err @ side)
    return {"seen": True, "error": err, "off_cm": round(float(np.linalg.norm(err)) * 100, 1),
            "checked": "position" if full else "sideways only (camera too low to judge forward/back)"}


async def _align_jaws(serial: str, above, step: str, **move_kw) -> tuple[np.ndarray, list]:
    """Hover over `above`, look at the real jaws, and move so their midpoint (not the model's) is there.
    Returns the shift that was applied (real minus model) and what each look found."""
    above = np.asarray(above, dtype=float)
    shift, looks = np.zeros(3), []
    for _ in range(2):
        new, info = await _jaw_shift(serial)
        looks.append(info)
        if info["source"] == "none" or np.linalg.norm(new - shift) < JAW_CORRECT_TOL_M:
            if info["source"] != "none":
                shift = new
            break
        shift = new
        await _move_tip(serial, above - shift, step, corrections=1, **move_kw)
        if info["source"] == "saved":
            break
    live = [i for i in looks if i["source"] == "camera"]
    if live:
        _save_jaw_offset(live[-1]["local_x_cm"] / 100, live[-1])
    return shift, looks


def _auto_roll_deg(serial: str, pose: dict[int, float], axis_deg: float) -> float:
    """Wrist roll (deg from straight) that makes the jaws close across an object's long axis."""
    m = _model(serial)
    long_axis = np.array([np.cos(np.radians(axis_deg)), 0.0, np.sin(np.radians(axis_deg))])
    best, best_cost = 0.0, 2.0
    for deg in np.arange(-WRIST_ROLL_MAX_DEG, WRIST_ROLL_MAX_DEG + 0.1, 5.0):
        c = m.jaw_close_dir(pose | {7: _wrist_roll_target(float(deg))})
        h = np.array([c[0], 0.0, c[2]])
        n = np.linalg.norm(h)
        if n < 0.5:
            continue
        cost = abs(float(h @ long_axis) / n) + abs(deg) * 1e-4      # ties: the smaller turn
        if cost < best_cost:
            best, best_cost = float(deg), cost
    return best


async def _gripper_goal(serial: str, motors: dict, norm: float) -> None:
    m = motors[GRIPPER_ID]
    raw = from_norm(norm, m["range_min"], m["range_max"])
    await send_commands(await client(), [
        _sync_write(serial, RAM_GOAL_SPEED, [(GRIPPER_ID, MOVE_SPEED.to_bytes(2, "little"))]),
        _sync_write(serial, RAM_TORQUE_ENABLE, [(GRIPPER_ID, b"\x01")]),
        _sync_write(serial, RAM_GOAL_POSITION, [(GRIPPER_ID, raw.to_bytes(2, "little"))]),
    ])


async def _open_gripper(serial: str, opening: float, timeout_s: float = 3.0) -> float:
    await guard_motion(serial)
    motors = (await read_buses())[serial]["motors"]
    await _gripper_goal(serial, motors, opening)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(0.15)
        pos = (await read_buses())[serial]["motors"][GRIPPER_ID]["position_norm"]
        if abs(pos - opening) < 0.02:
            break
    return pos


async def _close_gripper(serial: str, step: float = 0.06) -> dict:
    """Close in small steps until the current says we're squeezing something (or the jaws meet)."""
    await guard_motion(serial)
    # The current alone can stall just under GRASP_CURRENT (a minifigure stopped the jaws at 10-11 and this
    # looped forever): jaws that stop moving short of the floor are holding something too. Bounded in time.
    deadline = time.monotonic() + 10.0
    last_pos, stalled = None, 0
    while True:
        motors = (await read_buses())[serial]["motors"]
        m = motors[GRIPPER_ID]
        pos = m["position_norm"]
        stalled = stalled + 1 if last_pos is not None and last_pos - pos < 0.005 else 0
        last_pos = pos
        if stalled >= 3 and pos > GRIPPER_FLOOR + 0.04 and m["current"] >= GRASP_CURRENT / 2:
            hold_goal = max(GRIPPER_FLOOR, pos - 0.03)
            await _gripper_goal(serial, motors, hold_goal)
            return {"grasped": True, "position": pos, "current": m["current"], "hold_goal": hold_goal,
                    "note": "jaws stalled on the object"}
        if time.monotonic() > deadline:
            return {"grasped": False, "position": pos, "current": m["current"], "why": "close timed out"}
        if m["current"] >= GRASP_CURRENT:
            hold_goal = max(GRIPPER_FLOOR, m["position_norm"] - 0.03)   # keep a firm, bounded squeeze
            await _gripper_goal(serial, motors, hold_goal)
            return {"grasped": True, "position": m["position_norm"], "current": m["current"], "hold_goal": hold_goal}
        if m["position_norm"] <= GRIPPER_FLOOR + 0.01:
            return {"grasped": False, "position": m["position_norm"], "current": m["current"]}
        await _gripper_goal(serial, motors, max(GRIPPER_FLOOR, m["position_norm"] - step))
        await asyncio.sleep(0.35)


def _plan_grasp(serial: str, x: float, z: float, jaw_low_y: float, clear_top_m: float = 0.0) -> dict:
    """Steepest jaw angle at (x, z) that reaches both the grasp and a safe approach straight above it.
    Returns the grasp-point height that puts the lowest jaw point at jaw_low_y and the approach height.
    The approach keeps the jaw ends at least 1.5 cm above `clear_top_m` (the object's height): an
    approach below the object's top drives the jaws through it (2026-11-22: a 4.2 cm figure, approach
    2 cm above the grasp). (x, z) is where the MODEL must aim, i.e. after reach / jaw corrections."""
    m = _model(serial)
    start = _last_pose.get(serial, HOME_POSE)
    for ay in GRASP_TILTS:
        y = jaw_low_y + 0.02
        ok = False
        for _ in range(3):                       # the jaw depth below the grasp point depends on the pose
            ik = _ik(serial, [x, y, z], start, ay)
            if ik.error > 0.004:
                break
            q = ik.positions | {GRIPPER_ID: GRIPPER_OPEN}
            depth = m.grasp_point(q, GRASP_REACH_M)[1] - m.gripper_points(q)[:, 1].min()
            y_new = jaw_low_y + depth
            ok = abs(y_new - y) < 0.002
            y = y_new
            if ok:
                break
        if not ok:
            continue
        need = max(0.03, clear_top_m - jaw_low_y + 0.015)
        for clearance in sorted({c for c in (TRAVEL_CLEARANCE_M, 0.05, 0.03) if c >= need} | {need}, reverse=True):
            if _ik(serial, [x, y + clearance, z], start, ay).error < 0.006:
                return {"approach_y": ay, "grasp_y": y, "above_y": y + clearance, "pose": ik.positions}
        # No safe approach at this jaw angle: try a flatter one rather than coming in low.
    raise RuntimeError(f"({x:.3f}, {z:.3f}) can't be reached from above with room to clear the object")


def _marker_cfg() -> dict | None:
    try:
        cfg = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
        return cfg if cfg.get("offset_local") else None
    except (OSError, ValueError):
        return None


async def _measure_marker(serial: str, marker_id: int, size_m: float,
                          dictionary: str = MARKER_DICT) -> tuple[np.ndarray | None, dict]:
    frame = await D.fetch_frame(REALSENSE_URL)
    marks = [mk for mk in D.detect_markers(frame, _calibration()["camera_pose"], size_m, dictionary)
             if mk["id"] == marker_id]
    motors = (await read_buses())[serial]["motors"]
    return (np.array(marks[0]["position"]) if marks else None), motors


async def _marker_delta(serial: str) -> dict | None:
    """Real gripper position minus what the arm model predicts, measured with the gripper marker (world, m).
    Both the object and the marker are seen by the same depth camera, so camera calibration errors cancel."""
    cfg = _marker_cfg()
    if not cfg:
        return None
    measured, motors = await _measure_marker(serial, cfg["id"], cfg["size_m"], cfg.get("dictionary", "DICT_4X4_50"))
    if measured is None:
        return {"visible": False}
    Tend = _model(serial).end_transform(_norm_pose(motors))
    predicted = Tend[:3, :3] @ np.array(cfg["offset_local"]) + Tend[:3, 3]
    delta = measured - predicted
    return {"visible": True, "delta": delta, "delta_cm": np.round(delta * 100, 1).tolist()}


@mcp.tool()
async def find_markers(marker_size_cm: float = MARKER_SIZE_CM, dictionary: str = MARKER_DICT) -> list[dict]:
    """Fiducial markers the depth camera sees (default: the gripper's AprilTag 16h5, id 3), with position
    (robot frame, m), normal and x_axis. dictionary: any OpenCV name, e.g. "DICT_4X4_50". A diagnostic:
    nothing in the pick / place workflow depends on it."""
    frame = await D.fetch_frame(REALSENSE_URL)
    return D.detect_markers(frame, _calibration()["camera_pose"], marker_size_cm / 100, dictionary)


@mcp.tool()
async def calibrate_gripper_marker(marker_id: int = MARKER_ID, marker_size_cm: float = MARKER_SIZE_CM,
                                   reset: bool = False, dictionary: str = MARKER_DICT,
                                   bus: str | None = None) -> dict:
    """LEGACY, not used by pick any more (pick finds the real jaw tips in the camera image instead; the tag
    is hidden at grasp poses). Records where the ArUco marker sits on the gripper at the current pose, in
    the gripper's own frame; averages 2-3 calls. reset=True starts over."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await _wait_settled(serial, timeout_s=2.0)
    measured, motors = await _measure_marker(serial, marker_id, marker_size_cm / 100, dictionary)
    if measured is None:
        raise RuntimeError(f"marker {marker_id} not visible to the depth camera at this pose (find_markers to check)")
    Tend = _model(serial).end_transform(_norm_pose(motors))
    local = (Tend[:3, :3].T @ (measured - Tend[:3, 3])).tolist()
    fresh = {"id": marker_id, "size_m": marker_size_cm / 100, "dictionary": dictionary, "samples": []}
    cfg = (None if reset else _marker_cfg_raw()) or fresh
    if (cfg.get("id") != marker_id or abs(cfg.get("size_m", 0) - marker_size_cm / 100) > 1e-6
            or cfg.get("dictionary", "DICT_4X4_50") != dictionary):
        cfg = fresh
    cfg["samples"] = (cfg.get("samples") or [])[-9:] + [local]
    S = np.array(cfg["samples"])
    cfg["offset_local"] = S.mean(axis=0).tolist()
    spread = float(np.linalg.norm(S - S.mean(axis=0), axis=1).max()) if len(S) > 1 else 0.0
    cfg["spread_cm"] = round(spread * 100, 2)
    MARKER_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return {"samples": len(S), "offset_local_cm": np.round(S.mean(axis=0) * 100, 1).tolist(),
            "spread_cm": cfg["spread_cm"], "marker_world": np.round(measured, 4).tolist()}


def _marker_cfg_raw() -> dict | None:
    try:
        return json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def _correct_with_marker(serial: str, target, step: str, **move_kw) -> dict | None:
    """If the gripper marker shows the real grip point off from where the model put it, re-aim by the
    difference (up to twice). Returns what it did, or None without a calibrated marker."""
    if not _marker_cfg():
        return None
    target = np.asarray(target, dtype=float)
    aim = target.copy()
    log_m = []
    for _ in range(2):
        d = await _marker_delta(serial)
        if not d or not d["visible"]:
            log_m.append("marker not visible")
            break
        log_m.append(d["delta_cm"])
        n = float(np.linalg.norm(d["delta"]))
        if n < MARKER_CORRECT_MIN_M or n > MARKER_CORRECT_MAX_M:
            break
        aim = aim - d["delta"]
        await _move_tip(serial, aim, step, corrections=0, **move_kw)
    return {"marker_delta_cm": log_m}


def _object(object_id: int) -> dict:
    for o in _last_objects:
        if o["id"] == object_id:
            return o
    raise RuntimeError(f"no object {object_id} from the last find_objects; call find_objects again")


FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", str(Path(__file__).resolve().parent / "data" / "frames")))
FRAMES_KEEP = int(os.environ.get("FRAMES_KEEP", "60"))
# Jaw angle the arm can still manage at a distance from the base (m), steepest first: straight down reaches
# ~32 cm, 64 deg ~36 cm, 53 deg ~38 cm, 44 deg ~41 cm, 30 deg ~42 cm (GRASP_TILTS, measured on the model).
REACH_BY_JAW_ANGLE = ((0.32, 90), (0.36, 64), (0.38, 53), (0.41, 44), (0.42, 30))


def _save_frame(frame: D.Frame, norm: dict[int, float], tag: str) -> None:
    """Keep the last FRAMES_KEEP depth-camera frames (+ joint readings) on disk. On 2026-11-22 every miss
    had to be debugged from memory because the frames were gone; these make it an offline job."""
    try:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        stem = FRAMES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
        stem.with_suffix(".bin").write_bytes(D.encode_frame(frame))
        stem.with_suffix(".json").write_text(json.dumps({"norm": {str(k): v for k, v in norm.items()}, "tag": tag}))
        old = sorted(FRAMES_DIR.glob("*.bin"))[:-FRAMES_KEEP]
        for f in old:
            f.unlink(missing_ok=True)
            f.with_suffix(".json").unlink(missing_ok=True)
    except OSError as e:
        log.warning("could not save frame: %s", e)


def _reach_info(x: float, z: float, standing: bool) -> dict:
    """Can the arm grasp something at (x, z) from above, at what jaw angle, and which way to go about it."""
    r = float(np.hypot(x, z))
    need = r - _reach_error(r)          # the model must aim this far out for the real jaws to get to r
    angle = next((deg for limit, deg in REACH_BY_JAW_ANGLE if need <= limit), None)
    if angle is None:
        return {"reachable": False, "how": f"out of reach ({r * 100:.0f} cm from the base; the arm manages ~39 cm)"}
    if standing and angle >= 44:
        how = "pick(object_id)"
    else:
        how = "pick(object_id, close=False), check the photo, nudge_jaws if needed, close_and_lift()"
    return {"reachable": True, "jaw_angle_deg": angle, "how": how}


@mcp.tool()
async def find_objects(bus: str | None = None) -> list:
    """See what's on the table (not the table, not the arm). Call it with the arm at home, or after
    lift_and_look: anything joined to the arm in the image is treated as arm.

    Objects are found in the depth camera's COLOR image (things that differ from the table around them) and
    placed with the calibrated table plane; depth only gives their height. Returns a JSON list and the photo
    with numbered boxes, nearest first. Each object: id, color, center [x, y, z] (m, robot frame: base at
    origin, y up, arm faces +x), top_m (height), size_cm [length, width, height], standing (taller than
    ~2 cm), `ends` for long flat objects (a pen), and:
      reach: {reachable, jaw_angle_deg, how} - `how` is the tool call to use for this object;
      crowded_by: {id, gap_cm} when another object sits within a jaw's width - move that one first.
    Use the id with pick / place_near. Not seen: things the same color as the table.
    """
    global _last_objects
    frame = await D.fetch_frame(REALSENSE_URL, frames=DEPTH_FRAMES)
    cal = _calibration()
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    norm = _norm_pose(buses[serial]["motors"])
    arm = _model(serial).surface_points(norm)
    _last_objects = D.find_objects_rgb(frame, cal["camera_pose"], arm)
    for o in _last_objects:
        o["reach"] = _reach_info(o["center"][0], o["center"][2], o["standing"])
    _save_frame(frame, norm, "find")
    keys = ("id", "color", "center", "top_m", "size_cm", "standing", "distance_m", "ends", "reach", "crowded_by")
    summary = [{k: o[k] for k in keys if k in o} for o in _last_objects]
    return [json.dumps(summary), Image(data=D.annotate(frame, _last_objects), format="jpeg")]


@mcp.tool()
async def depth_point(fx: float, fy: float) -> dict:
    """3D point (robot frame, m) under a position in the RealSense color image, given as fractions of its
    width/height (0..1, from the top-left), e.g. a pen tip you see in find_objects' image."""
    frame = await D.fetch_frame(REALSENSE_URL, frames=DEPTH_FRAMES)
    p = D.pixel_to_world(frame, _calibration()["camera_pose"], fx, fy)
    if p is None:
        raise RuntimeError("no depth at that spot (try a point next to it)")
    return {"point": np.round(p, 4).tolist()}


@mcp.tool()
async def tip_position(bus: str | None = None) -> dict:
    """Where the gripper is now (robot frame, m), from the joint readings and the arm model."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    norm = _norm_pose(buses[serial]["motors"])
    m = _model(serial)
    return {"tip": np.round(m.tip(norm), 4).tolist(), "pointing": np.round(m.approach_dir(norm), 2).tolist(),
            "gripper": norm.get(GRIPPER_ID), "holding": _held is not None}


@mcp.tool()
async def pick(object_id: int | None = None, x: float | None = None, z: float | None = None,
               top_m: float = 0.03, wrist_roll_deg: float | None = None, grasp_height_m: float | None = None,
               dry_run: bool = False, close: bool = True, bus: str | None = None):
    """Pick up an object in ONE call: approach above it, open, look at the real jaws with the depth camera
    and centre them over the object, descend straight down, check the jaws are over it, close until the
    current says it's held, lift. If the check shows the jaws off the object after one correction it does
    NOT close: failed_step "verify" (then lift_and_look and pick again, or use close=False).
    If the check can't see the jaws at all it doesn't close either: it stops open like close=False
    (stopped_before_close) and returns the zoomed photo, so look, nudge_jaws, then close_and_lift.
    Give object_id from the last find_objects, or x, z (+ top_m, its height) in the robot frame.
    The open jaws are ~5 cm apart. wrist_roll_deg (±85, 0 = straight) turns them; leave it out and a long
    object from find_objects gets the turn that closes the jaws across its short side. grasp_height_m
    overrides the height of the point between the jaws (default from top_m).
    Reading the result: `jaw_look` says how far the real jaws were from the model sideways and whether that
    came from the camera or the saved value; `verify` says what the check before closing saw ("seen": false
    means the camera couldn't tell, and the grasp went ahead unchecked). `model_error_cm` in each step is encoder-vs-model only, not
    the real distance to the object. Every call is logged to data/runs.jsonl.
    dry_run=True moves nothing and returns the plan (jaw angle, heights, corrections) to check first.
    close=False stops with the jaws open around where the object should be and returns a zoomed photo:
    look at it, fix what you see with nudge_jaws, then call close_and_lift. Use this for small or flat
    objects and after a miss: the arm lands within ~1-1.5 cm, which is not enough for a lying minifigure.
    On failure it stops, holds, and says which step failed. After a failure do not go home: lift_and_look."""
    global _held
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    if object_id is not None:
        obj = _object(object_id)
        gx, gz, top = obj["center"][0], obj["center"][2], obj["top_m"]
    elif x is not None and z is not None:
        obj, gx, gz, top = None, x, z, top_m
    else:
        raise RuntimeError("give object_id, or x and z")
    _last_pose[serial] = _norm_pose(buses[serial]["motors"])
    reach = _reach_shift(gx, gz)
    plan = await asyncio.to_thread(_plan_grasp, serial, gx - reach[0], gz - reach[2], _jaw_low_y(top), top)
    if grasp_height_m is not None:
        plan["above_y"] += grasp_height_m - plan["grasp_y"]
        plan["grasp_y"] = grasp_height_m
    grasp_y, ay = plan["grasp_y"], plan["approach_y"]
    if wrist_roll_deg is None:
        # Turning the jaws only makes sense when they point (nearly) straight down.
        wrist_roll_deg = (_auto_roll_deg(serial, plan["pose"], obj["axis_deg"])
                          if obj and "ends" in obj and ay <= -0.8 else 0.0)
    above = np.array([gx, plan["above_y"], gz]) - reach
    kw = {"wrist_roll_deg": wrist_roll_deg, "approach_y": ay}
    t0 = time.monotonic()
    log_steps: dict = {}
    result: dict
    log_steps["plan"] = {"jaw_angle_deg_below_horizontal": round(float(np.degrees(np.arcsin(-ay))), 0),
                         "grasp_y": round(grasp_y, 4), "above_y": round(float(above[1]), 4),
                         "jaw_ends_clear_object_top_by_cm": round((float(above[1]) - grasp_y + _jaw_low_y(top) - top) * 100, 1),
                         "wrist_roll_deg": round(wrist_roll_deg, 0),
                         "reach_error_cm": round(_reach_error(float(np.hypot(gx, gz))) * 100, 1),
                         "saved_jaw_offset_cm": round(_stored_jaw_offset() * 100, 1)}
    if dry_run:
        return {"dry_run": True, "target": [round(gx, 4), round(gz, 4)], "top_m": top, "plan": log_steps["plan"]}
    try:
        # The wrist roll is part of the approach (up high): rolling near the table swings the jaws into it.
        log_steps["approach"] = await _move_tip(serial, above, "approach", corrections=2, **kw)
        log_steps["open"] = await _open_gripper(serial, GRIPPER_OPEN)
        shift, looks = await _align_jaws(serial, above, "centre jaws", gripper=GRIPPER_OPEN, **kw)
        log_steps["jaw_look"] = looks
        down = np.array([gx, grasp_y, gz]) - reach - shift
        log_steps["descend"] = await _move_tip(serial, down, "descend", corrections=2, gripper=GRIPPER_OPEN,
                                               steps=DESCENT_STEPS, **kw)
        # Before closing: is the object really between the jaws? One correction over the top if not; if it
        # still isn't, stop here instead of closing on air or on the object's edge.
        verify = []
        check = await _jaw_error_at_object(serial, np.array([gx, gz]))
        verify.append({k: v for k, v in check.items() if k != "error"})
        if check["seen"] and VERIFY_TOL_M < check["off_cm"] / 100 <= JAW_CORRECT_MAX_M:
            fix = np.array([check["error"][0], 0.0, check["error"][1]])
            await _move_tip(serial, above - shift, "verify (up)", corrections=1, gripper=GRIPPER_OPEN, **kw)
            await _move_tip(serial, above - shift + fix, "verify (across)", corrections=1, gripper=GRIPPER_OPEN, **kw)
            log_steps["descend_again"] = await _move_tip(serial, down + fix, "verify (down)", corrections=2,
                                                         gripper=GRIPPER_OPEN, steps=DESCENT_STEPS, **kw)
            shift = shift - fix
            check = await _jaw_error_at_object(serial, np.array([gx, gz]))
            verify.append({k: v for k, v in check.items() if k != "error"})
        log_steps["verify"] = verify
        if close and check["seen"] and check["off_cm"] / 100 > VERIFY_ABORT_M:
            await _move_tip(serial, above - shift, "retreat", corrections=1, gripper=GRIPPER_OPEN, **kw)
            result = {"success": False, "failed_step": "verify",
                      "why": f"the jaws are {check['off_cm']} cm from the object: not closing on nothing",
                      "steps": log_steps, "elapsed_s": round(time.monotonic() - t0, 1),
                      "next": "lift_and_look, then pick again (close=False to adjust by eye)"}
            _log_action("pick", {"target": [round(gx, 4), round(gz, 4)], "top_m": top}, result)
            return result
        # The check couldn't see the jaws: closing now is a guess (7 of 8 such picks missed on 2026-11-22,
        # knocking standing figures over). Stop open instead and let the caller look and nudge.
        unchecked = close and not check["seen"]
        if not close or unchecked:
            global _open_grasp
            _open_grasp = {"serial": serial, "approach_y": ay, "wrist_roll_deg": wrist_roll_deg, "object": obj,
                           "grasp_y": grasp_y, "clear_y": float(above[1])}
            result = {"success": None, "stopped_before_close": True, "steps": log_steps,
                      "next": "look at the photo; nudge_jaws(right_cm, toward_camera_cm) until the object is "
                              "between the jaw tips, then close_and_lift()",
                      **({"why": "the depth camera couldn't see the jaws to check them: not closing blind"}
                         if unchecked else {}),
                      "elapsed_s": round(time.monotonic() - t0, 1)}
            return [json.dumps(result), await _jaw_photo(serial)]
        grip = await _close_gripper(serial)
        log_steps["close"] = grip
        if not grip["grasped"]:
            await _open_gripper(serial, GRIPPER_OPEN)
            await _move_tip(serial, above - shift, "retreat", corrections=0, gripper=GRIPPER_OPEN, **kw)
            result = {"success": False, "failed_step": "close", "why": "jaws closed on nothing (current stayed low)"}
        else:
            log_steps["lift"] = await _move_tip(serial, above - shift, "lift", corrections=1,
                                                gripper=grip["hold_goal"], **kw)
            g = (await read_buses())[serial]["motors"][GRIPPER_ID]
            holding = g["position_norm"] > GRIPPER_FLOOR + 0.02 and g["current"] >= GRASP_CURRENT // 2
            if holding:
                _held = {"grasp_y": grasp_y, "hold_goal": grip["hold_goal"], "object": obj,
                         "wrist_roll_deg": wrist_roll_deg}
            result = {"success": holding, "failed_step": None if holding else "lift (dropped it?)",
                      "gripper": {"position": g["position_norm"], "current": g["current"]}}
    except MotionFailed as e:
        result = {"success": False, "failed_step": e.step, "why": e.result.get("abort_reason")}
    result |= {"steps": log_steps, "elapsed_s": round(time.monotonic() - t0, 1)}
    _log_action("pick", {"object": obj and {k: obj[k] for k in ("color", "center", "size_cm") if k in obj},
                         "target": [round(gx, 4), round(gz, 4)], "top_m": top}, result)
    return result


_open_grasp: dict | None = None      # set by pick(close=False): the jaws are open at an object, waiting


async def _jaw_photo(serial: str, half_px: int = 80, zoom: int = 5):
    """Zoomed depth-camera photo around the jaws, for judging by eye whether the object is between them.
    Without a camera it returns a line saying so: the move itself already happened and must be reported."""
    try:
        return await _jaw_photo_image(serial, half_px, zoom)
    except Exception as e:
        return f"(no photo: {type(e).__name__}: {e})"


async def _jaw_photo_image(serial: str, half_px: int, zoom: int) -> Image:
    import io

    from PIL import Image as PILImage

    frame = await D.fetch_frame(REALSENSE_URL)
    norm = _norm_pose((await read_buses())[serial]["motors"])
    T = _model(serial).end_transform(norm)
    jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
    ci = D.color_intrinsics(frame)
    Tinv = np.linalg.inv(D.pose_matrix(_calibration()["camera_pose"]))
    local = Tinv[:3, :3] @ jaw_end + Tinv[:3, 3]
    img = PILImage.open(io.BytesIO(frame.jpeg)).convert("RGB")
    u = int(ci["fx"] * local[0] / -local[2] + ci["ppx"])
    v = int(-ci["fy"] * local[1] / -local[2] + ci["ppy"])
    u = min(max(u, half_px), img.width - half_px)
    v = min(max(v, half_px), img.height - half_px)
    crop = img.crop((u - half_px, v - half_px, u + half_px, v + half_px)).resize(
        (2 * half_px * zoom, 2 * half_px * zoom), PILImage.LANCZOS)
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=88)
    return Image(data=out.getvalue(), format="jpeg")


def _camera_table_axes() -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors on the table for "right in the depth camera image" and "toward the camera"."""
    R = D.pose_matrix(_calibration()["camera_pose"])[:3, :3]
    right = R @ np.array([1.0, 0.0, 0.0])
    right[1] = 0.0
    toward = -(R @ np.array([0.0, 0.0, -1.0]))
    toward[1] = 0.0
    return right / np.linalg.norm(right), toward / np.linalg.norm(toward)


@mcp.tool()
async def nudge_jaws(right_cm: float = 0.0, toward_camera_cm: float = 0.0, up_cm: float = 0.0,
                     bus: str | None = None) -> list:
    """Move the open jaws a little, as seen in the depth camera photo: right_cm (+ = right in the image),
    toward_camera_cm (+ = down the image, nearer the camera), up_cm. At most 3 cm each. A sideways move
    always goes over the top: straight up until the jaws clear the object, across, straight down. (A
    1.5 cm hop pushed a standing figure 8 cm across the table, 2026-11-22.) Returns a new zoomed photo.
    A photo taken with the jaws low hides the object behind the gripper: if you can't tell where it is,
    call lift_and_look instead of guessing.
    Use after pick(close=False), then close_and_lift() when the object sits between the jaw tips."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    clip = lambda v: max(-3.0, min(3.0, float(v))) / 100
    right, toward = _camera_table_axes()
    norm = _norm_pose(buses[serial]["motors"])
    m = _model(serial)
    here = m.grasp_point(norm, GRASP_REACH_M)
    ay = (_open_grasp or {}).get("approach_y", float(m.approach_dir(norm)[1]))
    roll = (_open_grasp or {}).get("wrist_roll_deg")
    goal = here + right * clip(right_cm) + toward * clip(toward_camera_cm) + np.array([0.0, clip(up_cm), 0.0])
    kw = {"approach_y": ay, "wrist_roll_deg": roll, "gripper": norm.get(GRIPPER_ID)}
    steps = {}
    try:
        if abs(right_cm) > 0.05 or abs(toward_camera_cm) > 0.05:
            # Over the top, never sideways at table height: up, across, down.
            clear_y = max(here[1], goal[1], (_open_grasp or {}).get("clear_y", here[1] + 0.06))
            if clear_y > here[1] + 0.003:
                steps["up"] = await _move_tip(serial, np.array([here[0], clear_y, here[2]]), "nudge (up)",
                                              corrections=1, **kw)
            steps["across"] = await _move_tip(serial, np.array([goal[0], clear_y, goal[2]]), "nudge (across)",
                                              corrections=1, **kw)
        steps["settle"] = await _move_tip(serial, goal, "nudge", corrections=2, steps=DESCENT_STEPS, **kw)
        result = {"success": True, "moved_cm": {"right": right_cm, "toward_camera": toward_camera_cm, "up": up_cm},
                  "model_error_cm": steps["settle"]["model_error_cm"]}
    except MotionFailed as e:
        result = {"success": False, "failed_step": e.step, "why": e.result.get("abort_reason")}
    return [json.dumps(result), await _jaw_photo(serial)]


@mcp.tool()
async def lift_and_look(up_cm: float = 10.0, bus: str | None = None) -> list:
    """Raise the jaws straight up (default 10 cm) so the depth camera can see the table under them, then
    return what find_objects sees: the object list and the full photo. Use it whenever a photo with the
    jaws low doesn't show clearly where the object is, and after any miss, before deciding the next move.
    The jaws stay over the same spot, so nudge_jaws / pick can continue from there (no need to go home)."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    norm = _norm_pose(buses[serial]["motors"])
    m = _model(serial)
    here = m.grasp_point(norm, GRASP_REACH_M)
    info = _open_grasp or {}
    goal = here + np.array([0.0, max(2.0, min(15.0, float(up_cm))) / 100, 0.0])
    try:
        await _move_tip(serial, goal, "lift to look", corrections=1, gripper=norm.get(GRIPPER_ID),
                        approach_y=info.get("approach_y", float(m.approach_dir(norm)[1])),
                        wrist_roll_deg=info.get("wrist_roll_deg"))
    except (MotionFailed, RuntimeError) as e:      # e.g. out of reach that high: look from where it got to
        log.warning("lift_and_look: %s", e)
    return await find_objects(bus)


@mcp.tool()
async def close_and_lift(lift_cm: float = 6.0, bus: str | None = None) -> list:
    """Close the jaws where they are until they hold something, then lift straight up and check it is still
    held. For after pick(close=False) / nudge_jaws. If the jaws close on nothing they re-open and stay
    in place (look and nudge again). Returns a zoomed photo. A held object can then be placed with place."""
    global _held, _open_grasp
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    t0 = time.monotonic()
    grip = await _close_gripper(serial)
    if not grip["grasped"]:
        await _open_gripper(serial, GRIPPER_OPEN)
        result = {"success": False, "failed_step": "close", "why": "jaws closed on nothing; re-opened, still in place",
                  "close": grip}
    else:
        norm = _norm_pose((await read_buses())[serial]["motors"])
        m = _model(serial)
        here = m.grasp_point(norm, GRASP_REACH_M)
        info = _open_grasp or {}
        try:
            lift = await _move_tip(serial, here + np.array([0.0, max(1.0, min(10.0, lift_cm)) / 100, 0.0]), "lift",
                                   corrections=1, gripper=grip["hold_goal"],
                                   approach_y=info.get("approach_y", float(m.approach_dir(norm)[1])),
                                   wrist_roll_deg=info.get("wrist_roll_deg"))
            g = (await read_buses())[serial]["motors"][GRIPPER_ID]
            holding = g["position_norm"] > GRIPPER_FLOOR + 0.02 and g["current"] >= GRASP_CURRENT // 2
            if holding:
                _held = {"grasp_y": info.get("grasp_y", float(here[1])), "hold_goal": grip["hold_goal"],
                         "object": info.get("object"), "wrist_roll_deg": info.get("wrist_roll_deg") or 0.0}
                _open_grasp = None
            result = {"success": holding, "failed_step": None if holding else "lift (dropped it?)", "close": grip,
                      "gripper": {"position": g["position_norm"], "current": g["current"]}, "lift": lift}
        except MotionFailed as e:
            result = {"success": False, "failed_step": e.step, "why": e.result.get("abort_reason"), "close": grip}
    result["elapsed_s"] = round(time.monotonic() - t0, 1)
    _log_action("close_and_lift", {}, result)
    return [json.dumps(result), await _jaw_photo(serial)]


def _log_action(tool: str, request: dict, result: dict) -> None:
    """One line per pick/place in data/runs.jsonl, so success rates can be counted before and after a change."""
    try:
        _log_run({"tool": tool, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **request, **result})
    except (OSError, TypeError, ValueError) as e:
        log.warning("could not log %s: %s", tool, e)


@mcp.tool()
async def place(x: float, z: float, release_opening: float = 0.45, wrist_roll_deg: float | None = None,
                clear_m: float = 0.0, bus: str | None = None) -> dict:
    """Put the held object down at x, z (robot frame, m) in ONE call: carry above it, lower straight down to
    the height it was picked from, open, lift away. `release_opening` is the least it opens (small, so
    nearby things aren't pushed); it always opens wider than the grip on the object. The saved sideways jaw
    offset and the reach correction are applied. Logged to data/runs.jsonl.
    clear_m: height (m above the table) of anything the held object must pass over on the way, e.g. a
    box's wall: the carry then keeps the object's bottom ~3 cm above it, and it is lowered straight in."""
    global _held
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    grasp_y = _held["grasp_y"] if _held else MIN_GRASP_TIP_Y
    hold = _held["hold_goal"] if _held else None
    if wrist_roll_deg is None:
        wrist_roll_deg = _held.get("wrist_roll_deg", 0.0) if _held else 0.0
    _last_pose[serial] = _norm_pose(buses[serial]["motors"])
    reach = _reach_shift(x, z)
    held_top = ((_held or {}).get("object") or {}).get("top_m", 0.0) if _held else 0.0
    plan = await asyncio.to_thread(_plan_grasp, serial, x - reach[0], z - reach[2], JAW_MIN_Y, held_top)
    ay = plan["approach_y"]
    above = np.array([x, grasp_y + (plan["above_y"] - plan["grasp_y"]), z]) - reach
    if clear_m > 0:
        # The held object's bottom hangs grasp_y below the jaws; the loaded arm also settles ~2 cm low.
        above[1] = max(above[1], grasp_y + clear_m + PLACE_OVER_MARGIN_M)
    kw = {"wrist_roll_deg": wrist_roll_deg, "approach_y": ay}
    t0 = time.monotonic()
    log_steps: dict = {}
    try:
        # The loaded arm settles ~3 cm low without a correction, low enough to sweep things off the table.
        log_steps["carry"] = await _move_tip(serial, above, "carry", corrections=1, gripper=hold, **kw)
        shift, info = await _jaw_shift(serial, live=False)
        log_steps["jaw_offset"] = info
        log_steps["lower"] = await _move_tip(serial, np.array([x, grasp_y + 0.003, z]) - reach - shift, "lower",
                                             corrections=2, gripper=hold, steps=DESCENT_STEPS, **kw)
        # Open wider than the grip, whatever was asked: a wide object held at 0.56 was not let go by a
        # "release" to 0.45, which is narrower than the object (2026-11-22).
        if hold is not None:
            release_opening = min(0.9, max(release_opening, hold + 0.03 + 0.2))
        log_steps["release"] = await _open_gripper(serial, release_opening)
        log_steps["retreat"] = await _move_tip(serial, above - shift, "retreat", corrections=1,
                                               gripper=release_opening, **kw)
        _held = None
        # Then straight up, well clear of the table: from the low retreat the next pick's swing across the
        # table knocked over the object just placed (2026-11-22).
        clear = (above - shift).copy()
        clear[1] = max(clear[1], PLACE_CLEAR_Y)
        log_steps["rise"] = await _move_tip(serial, clear, "rise", corrections=1, gripper=release_opening, **kw)
        result = {"success": True, "placed_at": [round(x, 4), round(z, 4)]}
    except MotionFailed as e:
        result = {"success": False, "failed_step": e.step, "why": e.result.get("abort_reason")}
    result |= {"steps": log_steps, "elapsed_s": round(time.monotonic() - t0, 1)}
    _log_action("place", {"target": [round(x, 4), round(z, 4)]}, result)
    return result


@mcp.tool()
async def calibrate_reach(side: str = "right", distances_cm: list[float] | None = None,
                          bus: str | None = None) -> dict:
    """Measure how far short or long the real jaws land along the reach direction, and save it for
    pick / place (data/reach_map.json). Run it after calibrating the depth camera, with the gripper empty.

    The arm reaches to one side of the table (side: "right" or "left" as the robot sees it) at each
    distance, jaws open, 3 cm above the table, at the spot where its reach direction is square to the
    depth camera's line of sight. There the camera sees reach sideways, which it measures to ~2 mm, and
    an unknown jaw height barely matters. KEEP THAT SIDE OF THE TABLE CLEAR. Takes ~1.5 min.
    Returns one entry per distance: error_cm < 0 means the real jaws were short of the model.
    """
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    cal = _calibration()
    cam = D.pose_matrix(cal["camera_pose"])[:3, 3]
    cam_d = float(np.hypot(cam[0], cam[2]))
    cam_az = float(np.arctan2(cam[2], cam[0]))
    sign = 1.0 if side == "right" else -1.0          # the robot faces +x with y up, so its right is +z
    points, log_points = [], []
    try:
        for r_cm in distances_cm or [22.0, 27.0, 32.0, 37.0, 41.0]:
            r = r_cm / 100
            # Thales: on the circle whose diameter is base-camera, the base and the camera are seen at
            # a right angle, so the reach direction is square to the camera's line of sight.
            az = cam_az + sign * float(np.arccos(min(1.0, r / cam_d)))
            x, z = r * np.cos(az), r * np.sin(az)
            _last_pose[serial] = _norm_pose((await read_buses())[serial]["motors"])
            entry: dict = {"r_cm": r_cm, "at": [round(float(x), 3), round(float(z), 3)]}
            try:
                plan = await asyncio.to_thread(_plan_grasp, serial, float(x), float(z), REACH_PROBE_JAW_Y)
            except RuntimeError as e:
                entry["skipped"] = str(e)
                log_points.append(entry)
                continue
            ay = plan["approach_y"]
            if ay > -0.45:
                entry["skipped"] = "jaws too flat here for the camera to find the tips"
                log_points.append(entry)
                continue
            await _move_tip(serial, [x, plan["above_y"], z], "reach probe (above)", corrections=0,
                            gripper=GRIPPER_OPEN, approach_y=ay)
            await _move_tip(serial, [x, plan["grasp_y"], z], "reach probe", corrections=2,
                            gripper=GRIPPER_OPEN, approach_y=ay, steps=2)
            motors = (await _wait_settled(serial, timeout_s=2.0))[0][serial]["motors"]
            await asyncio.sleep(0.4)
            norm = _norm_pose(motors)
            T = _model(serial).end_transform(norm)
            jaw_end = T[:3, :3] @ (AM.JAW_AXIS_LOCAL * AM.JAW_END_M) + T[:3, 3]
            frame = await D.fetch_frame(REALSENSE_URL)
            found = D.find_jaw_tips(frame, cal["camera_pose"], jaw_end, plane_y=max(float(jaw_end[1]), 0.003))
            entry["jaw_angle_deg_below_horizontal"] = round(float(np.degrees(np.arcsin(-ay))), 0)
            if not found:
                entry["skipped"] = "jaw tips not found in the image"
            else:
                d = (np.array(found["midpoint"]) - jaw_end)[[0, 2]]
                radial = np.array([jaw_end[0], jaw_end[2]]) / float(np.hypot(jaw_end[0], jaw_end[2]))
                err = float(d @ radial)
                entry |= {"error_cm": round(err * 100, 1),
                          "across_cm": round(float(d @ np.array([-radial[1], radial[0]])) * 100, 1),
                          "jaw_gap_cm": round(found["gap_m"] * 100, 1)}
                if not JAW_GAP_MIN_M <= found["gap_m"] <= JAW_GAP_MAX_M:
                    # The jaws are open ~5 cm: any other gap means something else was taken for a tip.
                    entry["skipped"] = "jaw gap implausible: tips misread, not saved"
                elif abs(err) <= REACH_CORRECT_MAX_M:
                    points.append({"r_m": round(float(np.hypot(jaw_end[0], jaw_end[2])), 4), "error_m": round(err, 4),
                                   "tilt": ay})
                else:
                    entry["skipped"] = "error too large to be real: not saved"
            log_points.append(entry)
            await _move_tip(serial, [x, plan["above_y"], z], "reach probe (lift)", corrections=0,
                            gripper=GRIPPER_OPEN, approach_y=ay)
    except MotionFailed as e:
        log_points.append({"stopped": e.step, "why": e.result.get("abort_reason")})
    except Exception as e:      # never leave the arm stretched out over the table
        log_points.append({"stopped": "error", "why": f"{type(e).__name__}: {e}"})
    home_result = await home(bus)
    saved = False
    if len(points) >= 2:
        REACH_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        REACH_MAP_FILE.write_text(json.dumps({
            "points": points, "side": side, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "error_m = real minus model reach of the jaw ends (negative: lands short), by distance from "
                    "the base r_m. Measured by calibrate_reach with the depth camera, looking from the side."},
            indent=2), encoding="utf-8")
        saved = True
    return {"saved": saved, "points": log_points, "home": home_result.get("success")}


@mcp.tool()
async def place_near(object_id: int, end: int | None = None, gap_cm: float = 4.0,
                     release_opening: float = 0.45, bus: str | None = None) -> dict:
    """Put the held object next to another object from the last find_objects, `gap_cm` from its edge, on
    the side facing the robot. A long object (a pen) gets it beside the middle of its long side; pass
    end=0/1 to put it beside one of its `ends` instead."""
    obj = _object(object_id)
    c = np.array([obj["center"][0], obj["center"][2]])
    if end is None and "ends" in obj:
        # Beside the long side, not 4 cm past the far end of a 14 cm pen (which is what measuring from the
        # centre by half the LENGTH did).
        along = np.array(obj["ends"][1]["xz"]) - np.array(obj["ends"][0]["xz"])
        along /= max(np.linalg.norm(along), 1e-6)
        side = np.array([-along[1], along[0]])
        if side @ (-c) < 0:              # the side that faces the robot base
            side = -side
        target = c + side * (obj["size_cm"][1] / 200 + gap_cm / 100)
        return await place(float(target[0]), float(target[1]), release_opening, bus=bus)
    if end is not None:
        if "ends" not in obj:
            raise RuntimeError(f"object {object_id} is not elongated; omit end")
        e = np.array(obj["ends"][end]["xz"])
        along = e - c
        along /= max(np.linalg.norm(along), 1e-6)
        side = np.array([-along[1], along[0]])
        if side @ (-e) < 0:              # pick the side that faces the robot base
            side = -side
        target = e + side * (obj["size_cm"][1] / 200 + gap_cm / 100)
    else:
        toward = -c / max(np.linalg.norm(c), 1e-6)
        target = c + toward * (max(obj["size_cm"][:2]) / 200 + gap_cm / 100)
    return await place(float(target[0]), float(target[1]), release_opening, bus=bus)


@mcp.tool()
async def home(bus: str | None = None) -> dict:
    """Bring the arm back to its folded home pose safely from anywhere: shoulder back first (slowly, it
    lifts the arm away from the table), then roll joints straight, then fold the elbow last. ~30 s.
    Call it when the task is finished, and before find_objects if the arm is in the camera's way.
    NOT after a failed pick: use lift_and_look and try again from there."""
    global _held
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    await guard_motion(serial)
    t0 = time.monotonic()
    stages = []
    try:
        cur = _norm_pose(buses[serial]["motors"])
        while cur[2] > HOME_POSE[2] + 0.05:                 # 1. shoulder back, in chunks of <= 0.3
            tgt = dict(cur)
            tgt[2] = max(HOME_POSE[2], cur[2] - 0.3)
            stages.append((await _go(serial, tgt, "shoulder back"))["elapsed_s"])
            cur = _norm_pose((await read_buses())[serial]["motors"])
            if cur[2] > tgt[2] + 0.08:                       # couldn't lift: don't loop forever
                raise MotionFailed("shoulder back", {"abort_reason": f"shoulder stuck at {cur[2]:.2f}"})
        if any(abs(cur[j] - HOME_POSE[j]) > 0.03 for j in ROLL_IDS):   # 2. rolls straight
            tgt = dict(cur) | {j: HOME_POSE[j] for j in ROLL_IDS}
            stages.append((await _go(serial, tgt, "rolls straight"))["elapsed_s"])
        stages.append((await _go(serial, dict(HOME_POSE), "fold home"))["elapsed_s"])   # 3. the rest
        _held = None
        return {"success": True, "stages_s": stages, "elapsed_s": round(time.monotonic() - t0, 1)}
    except MotionFailed as e:
        return {"success": False, "failed_step": e.step, "why": e.result.get("abort_reason"),
                "elapsed_s": round(time.monotonic() - t0, 1)}


mcp.custom_route("/events", methods=["GET"])(A.events_endpoint)
mcp.custom_route("/events/clear", methods=["POST"])(A.clear_endpoint)


if __name__ == "__main__":
    log.info("starting MCP server (transport=%s, station=%s)", MCP_TRANSPORT, STATION_HOST)
    mcp.run(transport=MCP_TRANSPORT)
