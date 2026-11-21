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
"""

from __future__ import annotations

import asyncio
import logging
import os
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

from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

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
        buses[serial] = {"motors": dict(sorted(motors.items()))}
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
    """Discover camera queues from the station's inference-states index."""
    entries = await read_last("inference-states", 1)
    if not entries:
        return []
    rx = inference.InferenceRxReader(entries[-1])
    return sorted({e.get_queue() for e in rx.get_entries() or []
                   if e.get_type() == drivers.QueueDataType.QDT_USB_VIDEO_FRAMES})


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


async def latest_jpeg(queue_id: str) -> dict | None:
    """Most recent JPEG frame from a usbvideo queue.

    Returns {"jpeg", "label", "age_s" (None if unknown), "disconnected"}, where
    `disconnected` means the camera reported a disconnect after its last frame.
    """
    label = queue_id
    disconnected = False
    newest = True
    for raw in reversed(await read_last(queue_id, 8)):
        env = usbvideo.RxEnvelopeReader(raw)
        cam = env.get_camera()
        if cam and cam.get_product():
            label = f"{cam.get_product()} ({queue_id})"
        kind = env.get_type()
        if newest and kind == usbvideo.RxEnvelopeType.ET_DEVICE_DISCONNECTED:
            disconnected = True
        newest = False
        if kind != usbvideo.RxEnvelopeType.ET_FRAMES:
            continue
        pack = env.get_frames()
        frames = pack.get_frames_data() if pack else []
        if frames:
            return {"jpeg": bytes(frames[-1]), "label": label,
                    "age_s": _frame_age_s(env, pack), "disconnected": disconnected}
    return None


def frame_status(got: dict) -> tuple[str, bool]:
    """(text for the label, is_stale) for a frame from latest_jpeg."""
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

    camera: "all" (default), or a queue id / index from `list_cameras`.
    Returns one labeled JPEG per camera. Each label says how old the frame is;
    a frame marked STALE is NOT the current scene, so never plan moves from it.
    """
    queues = await camera_queues()
    if not queues:
        return ["No cameras found. Check the USB hub and that usb-video is enabled in station.yaml."]
    selected = list(enumerate(queues))  # keep each camera's index from list_cameras
    if camera != "all":
        if camera.isdigit():
            if int(camera) >= len(queues):
                return [f"No camera {camera}: only {len(queues)} found ({queues}). Use an index 0..{len(queues) - 1} or 'all'."]
            selected = [selected[int(camera)]]
        else:
            selected = [(i, q) for i, q in selected if camera in q]
            if not selected:
                return [f"No camera matching {camera!r}. Use list_cameras."]
    out: list = []
    for i, q in selected:
        got = await latest_jpeg(q)
        if got is None:
            out.append(f"camera {i}: {q} has no frames yet")
            continue
        status, stale = frame_status(got)
        text = f"camera {i}: {got['label']} - {status}"
        if stale:
            text += (". This is NOT the current scene: don't plan or verify moves with it. "
                     "Check the camera's USB connection and the station log.")
            log.warning("look: camera %s %s", q, status)
        out.append(text)
        out.append(Image(data=got["jpeg"], format="jpeg"))
    return out


@mcp.tool()
async def list_cameras() -> list[str]:
    """List camera queue ids known to the station."""
    return await camera_queues()


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
    """Torque on, holding the current pose."""
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


if __name__ == "__main__":
    log.info("starting MCP server (transport=%s, station=%s)", MCP_TRANSPORT, STATION_HOST)
    mcp.run(transport=MCP_TRANSPORT)
