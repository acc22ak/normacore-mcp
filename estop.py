"""Emergency stop for the normacore-mcp arm: a separate process with its own Station connection.

Run it in its own terminal next to the robot and keep that window focused:

    NORMA_CORE_REPO=~/norma-core STATION_HOST=localhost .venv/bin/python estop.py

    SPACE        HOLD : freeze every powered arm where it is (torque stays on)
    X  or  ESC   RELAX: torque off on every arm (it goes limp and DROPS, support it!)
    R  R         release the latch (press R twice within 3 s)
    Q            quit (a latch stays set until someone releases it)

One-shot commands (scripts, a second terminal, a foot pedal mapped to a key...):

    estop.py hold | relax | release | status

How it stops Claude, not just the servos:
  1. It writes HOLD/RELAX straight to the servos (sync-write, same as Station's own teleop),
     repeated a few times over ~0.3 s so it wins against a motion tick already in flight.
  2. It sets a LATCH: an entry in Station's `estop/state` queue. server.py and autopilot.py read
     it before every move and on every motion tick, so a running go_to / play_motion / run_skill /
     move_until stops within one tick, and every new move is refused until a human releases it.
     The latch lives in Station, so it works when the e-stop and the MCP server run on different
     laptops, and it survives restarting either of them.

Limits (be honest about them):
  - It needs a working connection to Station. If Station or the network is down, it can't reach
    the servos: the real last resort is the arm's power switch / unplugging its 12 V brick.
  - Station's own leader->follower mirroring (web-UI teleop or teach_mode) doesn't read the latch,
    so every press also sends Station "stop mirroring" for any mirrored follower. Station answers
    by turning that follower's torque off a moment later; HOLD keeps re-holding until that has
    happened, so the arm may sag for ~0.1 s before it holds. Mirroring stays off after a release:
    turn it back on deliberately (teach_mode(on=True) or the web UI).

Env vars: NORMA_CORE_REPO (required for station I/O), STATION_HOST (default localhost),
FOLLOWER_BUS (if set, HOLD only targets that bus), LEADER_BUS (never HOLD this bus).
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# latch protocol (pure, no station imports: server.py and autopilot.py import these)

QUEUE = "estop/state"
MODES = ("hold", "relax")


def encode_latch(latched: bool, mode: str = "hold", by: str = "", note: str = "") -> bytes:
    if latched and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return json.dumps({"latched": bool(latched), "mode": mode if latched else None,
                       "by": by or socket.gethostname(), "t": round(time.time(), 3),
                       "note": note}).encode()


def decode_latch(raw) -> dict | None:
    """Latest queue entry -> latch dict if latched, else None.

    Fails closed: an entry we can't parse counts as a HOLD latch, so garbage in the queue
    stops the arm instead of being ignored."""
    try:
        d = json.loads(bytes(raw).decode())
        if not isinstance(d, dict):
            raise ValueError("not an object")
    except (ValueError, UnicodeDecodeError):
        return {"latched": True, "mode": "hold", "by": "?", "t": None, "note": "unreadable estop entry"}
    if not d.get("latched"):
        return None
    mode = d.get("mode") if d.get("mode") in MODES else "hold"
    return {**d, "latched": True, "mode": mode}


def refusal(latch: dict) -> str:
    who = latch.get("by") or "?"
    return (f"E-STOP is latched ({latch['mode'].upper()}, set by {who}). Refusing to move. "
            "A human must check the robot and release it (estop.py: press R twice, or `estop.py release`).")


# ----------------------------------------------------------------------------
# station I/O (imports resolved lazily so the protocol above has no dependencies)

RAM_TORQUE_ENABLE = 0x28
RAM_GOAL_POSITION = 0x2A
RAM_PRESENT_POSITION = 0x38
RAM_PRESENT_CURRENT = 0x45
REPEATS = 4            # first write + 3 repeats
REPEAT_INTERVAL_S = 0.1

_mods = None


def _station_modules():
    global _mods
    if _mods is None:
        repo = Path(os.environ.get("NORMA_CORE_REPO", "")).expanduser()
        if not (repo / "software" / "station" / "shared" / "station_py").is_dir():
            sys.exit(f"NORMA_CORE_REPO must point at a norma-core checkout (got {str(repo)!r})")
        sys.path.insert(0, str(repo / "software" / "station" / "shared"))
        sys.path.insert(0, str(repo))
        import station_py
        from target.gen_python.protobuf.drivers.st3215 import st3215
        from target.gen_python.protobuf.station import commands, drivers
        _mods = (station_py, st3215, commands, drivers)
    return _mods


def _mirroring_module():
    _station_modules()
    from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
    return mirroring


MIRRORING_QUEUE = "inference/mirroring"
MIRROR_SETTLE_S = 2.5      # how long HOLD keeps re-holding after stopping Station's mirroring


def parse_mirrored_followers(raw) -> list[str]:
    """One inference/mirroring entry -> serials of buses that are currently mirroring targets."""
    st = _mirroring_module().RxEnvelopeReader(raw).get_state()
    out = []
    for link in (st.get_mirroring() or []) if st is not None else []:
        for t in link.get_targets() or []:
            if t.get_id() is not None:
                out.append(t.get_id().get_unique_id())
    return out


def stop_mirroring_commands(followers: list[str]) -> list:
    """CT_STOP_MIRROR per follower, same shape as Station's web UI / server.teach_mode(on=False):
    source = the follower bus, no targets. Station then unlinks it and (async) turns its torque off."""
    mirroring = _mirroring_module()
    _, _, commands, drivers = _station_modules()
    out = []
    for f in followers:
        cmd = mirroring.Command(type=mirroring.CommandType.CT_STOP_MIRROR,
                                source=mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=f),
                                targets=[])
        out.append(commands.DriverCommand(type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND,
                                          body=cmd.encode()))
    return out


def _pos(raw: int) -> int:
    # same sign handling as server._pos (ST3215 position register, bit 15 = sign)
    if raw & 0x8000:
        return (4096 - (raw & 0x0FFF)) & 0x0FFF
    return raw & 0x0FFF


def parse_buses(raw) -> dict[str, dict[int, dict]]:
    """One st3215/inference entry -> {bus serial: {motor id: {pos, torque, current}}}."""
    _, st3215, _, _ = _station_modules()
    out: dict[str, dict[int, dict]] = {}
    for bus in st3215.InferenceStateReader(raw).get_buses() or []:
        info = bus.get_bus()
        serial = info.get_serial_number() if info else "?"
        motors = {}
        for m in bus.get_motors() or []:
            s = bytes(m.get_state())
            motors[int(m.get_id())] = {
                "pos": _pos(struct.unpack_from("<H", s, RAM_PRESENT_POSITION)[0]) if len(s) >= RAM_PRESENT_POSITION + 2 else 0,
                "torque": len(s) > RAM_TORQUE_ENABLE and s[RAM_TORQUE_ENABLE] != 0,
                "current": struct.unpack_from("<H", s, RAM_PRESENT_CURRENT)[0] if len(s) >= RAM_PRESENT_CURRENT + 2 else 0,
            }
        out[serial] = dict(sorted(motors.items()))
    return out


def _sync_write(bus: str, address: int, writes: list[tuple[int, bytes]]):
    _, st3215, commands, drivers = _station_modules()
    cmd = st3215.Command(
        target_bus_serial=bus,
        sync_write=st3215.ST3215SyncWriteCommand(
            address=address,
            motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=i, value=v) for i, v in writes]))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


def hold_commands(buses: dict[str, dict[int, dict]], follower: str = "", leader: str = "",
                  only: list[str] | None = None) -> tuple[list, list[str]]:
    """Goal = present position, then torque on. Targets `only` if given (the buses chosen at the
    first press), else FOLLOWER_BUS if set, else every bus that is powered right now (a leader arm
    is limp, so it is skipped and never stiffened)."""
    if only is not None:
        targets = [b for b in only if b in buses]
    elif follower:
        targets = [follower] if follower in buses else []
    else:
        targets = [b for b, ms in buses.items() if b != leader and any(m["torque"] for m in ms.values())]
    cmds = []
    for b in targets:
        ms = buses[b]
        cmds.append(_sync_write(b, RAM_GOAL_POSITION, [(i, m["pos"].to_bytes(2, "little")) for i, m in ms.items()]))
        cmds.append(_sync_write(b, RAM_TORQUE_ENABLE, [(i, b"\x01") for i in ms]))
    return cmds, targets


def relax_commands(buses: dict[str, dict[int, dict]]) -> tuple[list, list[str]]:
    """Torque off on every motor of every bus (always safe for the servos; the arm drops)."""
    return [_sync_write(b, RAM_TORQUE_ENABLE, [(i, b"\x00") for i in ms]) for b, ms in buses.items()], list(buses)


class Station:
    """Thin wrapper so tests can swap the transport."""

    def __init__(self, host: str):
        self.host = host
        self.client = None

    async def connect(self):
        station_py, _, _, _ = _station_modules()
        import logging
        self.client = await station_py.new_station_client(self.host, logging.getLogger("estop"))

    async def _read_last(self, queue: str, timeout: float = 2.0):
        import asyncio
        station_py = _station_modules()[0]
        # tail offset 0 = the newest entry (offset 1 would be the one before it)
        qr = self.client.read_from_tail(queue, offset=b"\x00", limit=1, step=1, buf_size=2)
        try:
            entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        except asyncio.TimeoutError:
            entry = None
        if entry is None:
            if qr.err and (qr.err is station_py.ErrQueueNotFound or isinstance(qr.err, station_py.ErrQueueNotFound)):
                return None
            if qr.err:
                raise RuntimeError(f"reading {queue}: {qr.err}")
            return None
        return bytes(entry.Data)

    async def read_buses(self) -> dict[str, dict[int, dict]]:
        raw = await self._read_last("st3215/inference")
        if raw is None:
            raise RuntimeError("no data in st3215/inference: is an arm plugged in and powered?")
        return parse_buses(raw)

    async def send(self, cmds: list) -> None:
        if cmds:
            await _station_modules()[0].send_commands(self.client, cmds)

    async def read_mirrored_followers(self) -> list[str]:
        raw = await self._read_last(MIRRORING_QUEUE)
        return [] if raw is None else parse_mirrored_followers(raw)

    async def read_latch(self) -> dict | None:
        raw = await self._read_last(QUEUE)
        return None if raw is None else decode_latch(raw)

    async def write_latch(self, data: bytes) -> None:
        await self.client.enqueue(QUEUE, data)


async def trigger(st, mode: str, follower: str = "", leader: str = "", note: str = "",
                  repeats: int = REPEATS, interval: float = REPEAT_INTERVAL_S) -> dict:
    """Stop the arm NOW, then latch. Returns what was done (and any problem).

    If Station's leader->follower mirroring (teleop / teach_mode) is driving a follower, it is
    stopped too: otherwise the next mirrored goal would overwrite a HOLD. Station answers a stop
    by turning that follower's torque off a moment later, so HOLD keeps re-holding for up to
    MIRROR_SETTLE_S until that torque-off has landed (the arm may sag for ~0.1 s, then holds)."""
    import asyncio
    result = {"mode": mode, "buses": [], "latched": False, "errors": [], "mirroring_stopped": []}
    held: list[str] | None = None          # HOLD targets, fixed at the first press

    async def fire(first: bool = False):
        nonlocal held
        buses = await st.read_buses()
        if mode == "relax":
            cmds, targets = relax_commands(buses)
        else:
            cmds, targets = hold_commands(buses, follower, leader, only=held)
            if first:
                held = targets + [b for b in result["mirroring_stopped"] if b not in targets and b in buses]
                cmds, targets = hold_commands(buses, follower, leader, only=held)
        await st.send(cmds)
        return targets

    try:                                   # 0. stop Station's mirroring, so it can't overwrite us
        mirrored = await st.read_mirrored_followers()
        if mirrored:
            await st.send(stop_mirroring_commands(mirrored))
            result["mirroring_stopped"] = mirrored
    except Exception as e:                 # noqa: BLE001 - report, still stop the servos
        result["errors"].append(f"could not stop Station mirroring: {e}")
    try:                                   # 1. servos: that's what actually stops motion
        result["buses"] = await fire(first=True)
    except Exception as e:                 # noqa: BLE001 - report, still try to latch
        result["errors"].append(f"servo command failed: {e}")
    try:                                   # 2. latch, so the MCP server stops sending goals
        await st.write_latch(encode_latch(True, mode, note=note))
        result["latched"] = True
    except Exception as e:                 # noqa: BLE001
        result["errors"].append(f"LATCH NOT SET ({e}): Claude may keep moving the arm")
    for _ in range(repeats - 1):           # 3. repeat to win against a goal write already in flight
        await asyncio.sleep(interval)
        try:
            result["buses"] = await fire() or result["buses"]
        except Exception as e:             # noqa: BLE001
            result["errors"].append(f"repeat failed: {e}")
            break
    if mode == "hold" and result["mirroring_stopped"] and not any("repeat failed" in e for e in result["errors"]):
        # 4. Station turns a freed follower's torque off asynchronously: keep holding until we've
        #    seen it unlinked AND that torque-off has happened (then our hold is the last word).
        stopped = set(result["mirroring_stopped"])
        deadline = time.monotonic() + MIRROR_SETTLE_S
        saw_limp, unlinked = False, False
        while time.monotonic() < deadline:
            try:
                buses = await st.read_buses()
                unlinked = not (stopped & set(await st.read_mirrored_followers()))
                if any(not any(m["torque"] for m in buses.get(b, {}).values()) for b in stopped):
                    saw_limp = True
                if unlinked and saw_limp and all(any(m["torque"] for m in buses.get(b, {}).values()) for b in stopped):
                    break
                await fire()
            except Exception as e:         # noqa: BLE001
                result["errors"].append(f"re-hold after stopping mirroring failed: {e}")
                break
            await asyncio.sleep(interval)
        else:
            if not unlinked:
                result["errors"].append("Station mirroring still active after stop: turn it off in the web UI")
            elif not saw_limp:
                result["note"] = "didn't observe Station's torque-off; arm is held"
    if mode == "hold" and not result["buses"] and not result["errors"]:
        result["errors"].append("no powered arm found to hold (nothing had torque on)")
    return result


async def release(st, note: str = "") -> None:
    await st.write_latch(encode_latch(False, note=note))


# ----------------------------------------------------------------------------
# CLI

def _fmt_status(buses: dict | None, latch: dict | None, err: str | None) -> str:
    if err:
        return f"!! NO STATION: {err}  (last resort: arm power switch)"
    lt = f"LATCHED {latch['mode'].upper()} by {latch.get('by', '?')}" if latch else "armed (not latched)"
    parts = []
    for b, ms in (buses or {}).items():
        on = sum(m["torque"] for m in ms.values())
        cur = max((m["current"] for m in ms.values()), default=0)
        parts.append(f"{b}: torque {on}/{len(ms)}, max current {cur}")
    return f"[{lt}]  " + " | ".join(parts)


async def interactive(st, follower: str, leader: str) -> None:
    import asyncio
    import termios
    import tty

    print(__doc__.split("\n\n")[1])
    print("Connected to", st.host, "- keep this window focused.\n")
    loop = asyncio.get_running_loop()
    keys: asyncio.Queue = asyncio.Queue()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def on_key():
        ch = os.read(fd, 1)
        loop.call_soon_threadsafe(keys.put_nowait, ch.decode(errors="ignore"))

    async def status_loop():
        while True:
            err, buses, latch = None, None, None
            try:
                buses = await st.read_buses()
                latch = await st.read_latch()
            except Exception as e:  # noqa: BLE001
                err = str(e)
            sys.stdout.write("\r\x1b[2K" + _fmt_status(buses, latch, err)[:200])
            sys.stdout.flush()
            await asyncio.sleep(0.5)

    def say(msg: str):
        sys.stdout.write("\r\x1b[2K" + msg + "\n")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        loop.add_reader(fd, on_key)
        status = asyncio.create_task(status_loop())
        last_r = 0.0
        while True:
            ch = await keys.get()
            if ch == " ":
                r = await trigger(st, "hold", follower, leader, note="key SPACE")
                mir = f"  stopped mirroring on {r['mirroring_stopped']}" if r["mirroring_stopped"] else ""
                say(f">>> HOLD on {r['buses'] or 'no bus'}  latched={r['latched']}{mir}  {'; '.join(r['errors'])}")
            elif ch in ("x", "X", "\x1b"):
                r = await trigger(st, "relax", follower, leader, note="key RELAX")
                mir = f"  stopped mirroring on {r['mirroring_stopped']}" if r["mirroring_stopped"] else ""
                say(f">>> RELAX (torque off) on {r['buses']}  latched={r['latched']}{mir}  {'; '.join(r['errors'])}")
            elif ch in ("r", "R"):
                now = time.monotonic()
                if now - last_r <= 3.0:
                    await release(st, note="key R R")
                    say(">>> latch RELEASED: Claude may move the arm again")
                    last_r = 0.0
                else:
                    last_r = now
                    say("press R again within 3 s to release the latch")
            elif ch in ("q", "Q", "\x03", "\x04"):
                say("quitting (latch state unchanged)")
                break
        status.cancel()
    finally:
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def main_async(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="E-stop for the normacore-mcp arm")
    p.add_argument("action", nargs="?", default="interactive",
                   choices=["interactive", "hold", "relax", "release", "status"])
    p.add_argument("--host", default=os.environ.get("STATION_HOST", "localhost"),
                   help="Station host[:port] (default $STATION_HOST or localhost; port 8888)")
    p.add_argument("--bus", default=os.environ.get("FOLLOWER_BUS", "").strip(),
                   help="only HOLD this bus (default $FOLLOWER_BUS; else every powered bus)")
    p.add_argument("--leader", default=os.environ.get("LEADER_BUS", "").strip(),
                   help="never HOLD this bus (default $LEADER_BUS)")
    a = p.parse_args(argv)

    st = Station(a.host)
    await st.connect()
    if a.action == "interactive":
        await interactive(st, a.bus, a.leader)
        return 0
    if a.action in ("hold", "relax"):
        r = await trigger(st, a.action, a.bus, a.leader, note="cli")
        print(json.dumps(r))
        return 0 if r["latched"] and not r["errors"] else 1
    if a.action == "release":
        await release(st, note="cli")
        print("released")
        return 0
    err, buses, latch = None, None, None
    try:
        buses = await st.read_buses()
    except Exception as e:  # noqa: BLE001
        err = str(e)
    latch = await st.read_latch()
    print(_fmt_status(buses, latch, None) if not err else f"[{'LATCHED' if latch else 'armed'}] {err}")
    return 0


def main() -> None:
    import asyncio
    sys.exit(asyncio.run(main_async(sys.argv[1:])))


if __name__ == "__main__":
    main()
