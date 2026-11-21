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

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import usbvideo  # noqa: E402
from target.gen_python.protobuf.station import commands as station_commands  # noqa: E402
from target.gen_python.protobuf.station import drivers, inference  # noqa: E402

STATION_HOST = os.environ.get("STATION_HOST", "localhost")
FOLLOWER_BUS = os.environ.get("FOLLOWER_BUS", "").strip()
MAX_STEP_NORM = float(os.environ.get("MAX_STEP_NORM", "0.15"))
MOVE_SPEED = int(os.environ.get("MOVE_SPEED", "600"))
READ_TIMEOUT_S = 5.0

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
            calibrated = rmax > rmin
            motors[int(m.get_id())] = {
                "joint": ELROBOT_JOINTS.get(int(m.get_id()), f"motor_{m.get_id()}"),
                "position_raw": pos,
                "position_norm": round((pos - rmin) / (rmax - rmin), 3) if calibrated else None,
                "range_min": rmin,
                "range_max": rmax,
                "calibrated": calibrated,
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


async def latest_jpeg(queue_id: str) -> tuple[bytes, str] | None:
    """Most recent JPEG frame from a usbvideo queue, plus a camera label."""
    label = queue_id
    for raw in reversed(await read_last(queue_id, 8)):
        env = usbvideo.RxEnvelopeReader(raw)
        cam = env.get_camera()
        if cam and cam.get_product():
            label = f"{cam.get_product()} ({queue_id})"
        if env.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES:
            continue
        pack = env.get_frames()
        frames = pack.get_frames_data() if pack else []
        if frames:
            return bytes(frames[-1]), label
    return None


# ----------------------------------------------------------------------------
# tools

@mcp.tool()
async def look(camera: str = "all") -> list:
    """Grab the latest frame from the station cameras (gripper cam and desk cam).

    camera: "all" (default), or a queue id / index from `list_cameras`.
    Returns one labeled JPEG per camera.
    """
    queues = await camera_queues()
    if not queues:
        return ["No cameras found. Check the USB hub and that usb-video is enabled in station.yaml."]
    if camera != "all":
        queues = [queues[int(camera)]] if camera.isdigit() else [q for q in queues if camera in q]
    out: list = []
    for i, q in enumerate(queues):
        got = await latest_jpeg(q)
        if got is None:
            out.append(f"camera {i}: {q} has no frames yet")
            continue
        jpeg, label = got
        out.append(f"camera {i}: {label}")
        out.append(Image(data=jpeg, format="jpeg"))
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
    return {"follower_bus": follower, "buses": buses}


@mcp.tool()
async def move_joints(targets: dict[str, float], bus: str | None = None) -> dict:
    """Move joints to absolute normalized positions (0..1 in calibrated range).

    targets: {"<motor id or joint name>": position_norm}, e.g. {"1": 0.5, "gripper": 0.2}
    Each joint moves at most MAX_STEP_NORM per call (larger requests are clamped
    and reported), so call repeatedly and `look` in between.
    """
    buses = await read_buses()
    serial = pick_bus(buses, bus)
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
        raw = int(round(m["range_min"] + tgt * (m["range_max"] - m["range_min"])))
        raw = max(m["range_min"], min(m["range_max"], raw))
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


@mcp.tool()
async def hold(bus: str | None = None) -> str:
    """Torque on, holding the current pose."""
    buses = await read_buses()
    serial = pick_bus(buses, bus)
    motors = buses[serial]["motors"]
    goals = [(i, m["position_raw"].to_bytes(2, "little")) for i, m in motors.items()]
    await send_commands(await client(), [
        _sync_write(serial, RAM_GOAL_POSITION, goals),
        _sync_write(serial, RAM_TORQUE_ENABLE, [(i, b"\x01") for i in motors]),
    ])
    return f"holding pose on {serial}"


if __name__ == "__main__":
    log.info("starting MCP server (transport=%s, station=%s)", MCP_TRANSPORT, STATION_HOST)
    mcp.run(transport=MCP_TRANSPORT)
