"""The only module that talks to Station. Thin wrapper over ~/normacore-mcp/server.py, imported
as a library (no MCP transport), so the motion engine, mirroring guard, current/stall aborts and
camera roles are the same code the MCP server runs.

Two implementations with the same async interface:
  RealIO  -> Station over TCP :8888 (needs the follower arm and cameras plugged in)
  SimIO   -> no hardware: a saved frame and a simulated arm that moves instantly-ish
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

MCP_DIR = Path(os.environ.get("NORMACORE_MCP_DIR", str(Path(__file__).resolve().parents[2])))
os.environ.setdefault("NORMA_CORE_REPO", str(Path.home() / "norma-core"))
os.environ.setdefault("FOLLOWER_BUS", "5B8E115950")
os.environ.setdefault("LEADER_BUS", "5B91043428")
REAL_FOLLOWER = "5B8E115950"
LEADER_SERIAL = "5B91043428"
# Driving the LEADER arm (a test override): its weak 5 V servos need ~100+ current units just to
# lift the forearm, which trips the default limit of 100 (Station's own default). Raise it for the
# shoulder, upper_arm_roll and elbow only (motors 2, 3, 4), and only on the leader. Approved by the
# user on 2026-11-22 after watching the motion. The servos' own overload protection still applies.
LEADER_CURRENT_LIMITS = os.environ.get("LEADER_CURRENT_LIMITS", "2:160,3:160,4:160")
# The weak elbow also falls up to ~0.16 behind its path while lifting the forearm, which trips the
# default "fell behind" stop (0.10). On the leader, allow 0.35 (user's request, 2026-11-22): the
# replay sags instead of stopping, but a joint that's truly stuck still stops it.
LEADER_TRACK_ERR_MAX = os.environ.get("LEADER_TRACK_ERR_MAX", "0.35")
if os.environ["FOLLOWER_BUS"] == LEADER_SERIAL:
    os.environ.setdefault("CURRENT_LIMITS", LEADER_CURRENT_LIMITS)
    os.environ.setdefault("TRACK_ERR_MAX", LEADER_TRACK_ERR_MAX)

log = logging.getLogger("station_io")


class StationError(RuntimeError):
    pass


class RealIO:
    name = "station"

    def __init__(self):
        sys.path.insert(0, str(MCP_DIR))
        import server  # noqa: E402  (importing does not start the MCP server)
        self.S = server
        self.follower = os.environ["FOLLOWER_BUS"]

    async def connect(self) -> None:
        """Fail early if the follower isn't live: Station keeps the last joint snapshot after an arm
        is unplugged, so a stale bus would otherwise look connected."""
        a = await self._bus()
        await asyncio.sleep(0.15)
        b = await self._bus()
        if b["stamp_ns"] == a["stamp_ns"]:
            raise StationError(f"follower {self.follower} is not sending data (unplugged or unpowered?)")
        ms = await self.S.mirroring_state()
        if self.S._is_mirrored_follower(ms, self.follower):
            raise StationError("Station teleop is mirroring onto the follower; turn it off in the web UI first")

    async def _bus(self) -> dict:
        # Station briefly has no arm data while a USB link drops and re-enumerates: retry ~1 s
        for attempt in range(5):
            try:
                buses = await self.S.read_buses()
                if self.follower in buses:
                    break
            except RuntimeError:
                if attempt == 4:
                    raise
                if attempt >= 1:
                    self.reconnect()
            await asyncio.sleep(0.25)
        if self.follower not in buses:
            raise StationError(f"follower bus {self.follower} not found (have {list(buses)})")
        return buses[self.follower]

    def reconnect(self) -> None:
        """Drop the Station connection; the next read opens a fresh one. A long history read that
        times out half-way can leave its stream unread, which stalls every later read on the same
        connection ("no data in st3215/inference" while Station itself is fine)."""
        log.warning("Station reads keep coming back empty: opening a fresh connection")
        self.S._client = None

    async def mirrored_from(self) -> str | None:
        """The leader bus if Station teleop is driving the follower right now, else None."""
        return self.S._is_mirrored_follower(await self.S.mirroring_state(), self.follower)

    async def read_joints(self) -> dict[str, dict]:
        """joint name -> {position_norm, range_min, range_max, current, torque_on, error_status}"""
        motors = (await self._bus())["motors"]
        return {m["joint"]: m for m in motors.values()}

    async def read_frame(self, role: str = "desk") -> tuple[bytes, float]:
        """Newest JPEG from the camera with this role, and its age in seconds."""
        cams, gone = await self.S.cameras()
        roles, problems = self.S.camera_roles(cams, gone)
        cam = roles.get(role)
        if cam is None or cam["jpeg"] is None:
            raise StationError(f"no '{role}' camera frame ({problems or 'role not set'})")
        return cam["jpeg"], float(cam["age_s"] if cam["age_s"] is not None else 1e9)

    async def play(self, waypoints: list[dict], speed_scale: float) -> dict:
        """Keyframes [{"joints": {name: norm}, "duration_s": s}, ...] through the MCP motion engine.
        Aborts and holds on overcurrent, tracking error, stale state or stop()."""
        return await self.S._run_motion(waypoints, speed_scale, self.follower)

    def stop(self) -> None:
        """Ask a running motion to stop and hold. Returns immediately; safe from any thread via
        loop.call_soon_threadsafe."""
        self.S._abort.set()

    async def hold(self) -> None:
        await self.S._hold_shielded(self.follower)

    async def relax(self) -> str:
        """Torque off on every motor of this bus: the arm goes limp."""
        return await self.S.relax(bus=self.follower)

    async def stamp_ns(self) -> int:
        return int((await self._bus())["stamp_ns"])

    async def history_since(self, since_ns: int) -> list[tuple[float, dict[str, float]]]:
        """Joint positions (name -> norm) logged by Station since a stamp, ~20 Hz, t from 0."""
        serial, samples, extra = await self.S._read_history(self.follower, since_ns=since_ns)
        names = {mid: m["joint"] for mid, m in extra["last"]["motors"].items()}
        return [(t, {names.get(j, j): v for j, v in norm.items()}) for t, norm, _raw in samples]


class SimIO:
    """No hardware. Joints start at the given pose; play() glides there over the planned time
    (scaled down by sim_speed) and honours stop()."""
    name = "sim"
    RANGE = (1025, 3071)  # a plausible calibrated range, same for every joint

    def __init__(self, frame_path: Path, start_pose: dict[str, float], sim_speed: float = 4.0):
        self.frame = Path(frame_path).read_bytes()
        self.pose = dict(start_pose)
        self.sim_speed = sim_speed
        self._abort = asyncio.Event()
        self.sent: list[dict] = []    # every waypoint list handed to play(), for tests
        self._torque = {n: True for n in self.pose}
        self.surface: dict[str, tuple[float, int]] = {}   # joint -> (norm where it is blocked, +1/-1 side)

    async def connect(self) -> None:
        return None

    def _blocked(self, n: str, v: float) -> tuple[float, bool]:
        if n not in self.surface:
            return v, False
        at, side = self.surface[n]
        return (at, True) if (v - at) * side > 0 else (v, False)

    async def read_joints(self) -> dict[str, dict]:
        return {n: {"joint": n, "position_norm": round(v, 3), "range_min": self.RANGE[0],
                    "range_max": self.RANGE[1], "current": getattr(self, "_load", {}).get(n, 2), "torque_on": self._torque.get(n, True), "error_status": 0}
                for n, v in self.pose.items()}

    async def read_frame(self, role: str = "desk") -> tuple[bytes, float]:
        return self.frame, 0.1

    async def play(self, waypoints: list[dict], speed_scale: float) -> dict:
        self.sent.append({"waypoints": waypoints, "speed_scale": speed_scale})
        self._abort.clear()
        t0 = time.monotonic()
        prev_t = 0.0
        for w in waypoints:
            if "t" in w:   # recorded samples: original spacing
                dur, prev_t = max(0.0, float(w["t"]) - prev_t), float(w["t"])
                dur = max(0.001, dur / max(speed_scale, 0.1) / self.sim_speed)
            else:
                dur = max(0.05, float(w.get("duration_s", 0.5))) / max(speed_scale, 0.1) / self.sim_speed
            try:
                await asyncio.wait_for(self._abort.wait(), timeout=dur)
                return {"completed": False, "abort_reason": "stopped", "final": dict(self.pose),
                        "elapsed_s": round(time.monotonic() - t0, 2)}
            except asyncio.TimeoutError:
                self._load = {}
                for n, v in (w.get("joints") or {}).items():
                    self.pose[n], hit = self._blocked(n, v)
                    if hit:   # pushing on the surface: the servo draws more and can't get there
                        self._load[n] = 2 + int(400 * abs(v - self.pose[n]))
                self._torque = {n: True for n in self.pose}
        return {"completed": True, "abort_reason": None, "settled": True, "final": dict(self.pose),
                "elapsed_s": round(time.monotonic() - t0, 2)}

    def stop(self) -> None:
        self._abort.set()

    async def hold(self) -> None:
        return None

    async def relax(self) -> str:
        for m in self._torque:
            self._torque[m] = False
        return "torque disabled (sim)"

    async def stamp_ns(self) -> int:
        return time.monotonic_ns()

    async def mirrored_from(self) -> str | None:
        return getattr(self, "mirror_leader", None)

    async def history_since(self, since_ns: int) -> list[tuple[float, dict[str, float]]]:
        """Tests put hand-made samples in self.fake_history."""
        return list(getattr(self, "fake_history", []))
