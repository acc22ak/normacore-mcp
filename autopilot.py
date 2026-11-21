"""Autopilot mode: a SEPARATE, OPTIONAL MCP server that adds a guarded "move until
it stops" probe on top of the main normacore server WITHOUT changing any of its code.

Why a second server? It lets us add tools without editing server.py. autopilot.py
imports server.py and motion.py as libraries and reuses their station I/O, safety
monitor and normalization; server.py never imports autopilot.py. So the whole feature
is this one file plus data/control_mode.json and tests/test_autopilot.py -- delete them
and the main server is byte-for-byte unchanged. (Same pattern the roadmap plans for
estop.py: a separate process with its own station connection.)

Run it alongside the main server:
  stdio (Claude Code spawns it):   claude mcp add --transport stdio autopilot -- \
                                     python /path/to/autopilot.py
  own http port (watch its logs):  AUTOPILOT_MCP_TRANSPORT=streamable-http python autopilot.py
                                     (default 127.0.0.1:8791, so it does not clash with 8790)
Needs the same env as server.py (NORMA_CORE_REPO, STATION_HOST, FOLLOWER_BUS, ...).

What it adds
------------
- set_control_mode("step"|"autopilot"): selects, at runtime (no restart), whether the
  autopilot tools are usable. Stored in data/control_mode.json; missing/invalid -> "step"
  (fail-safe). A human can flip it out-of-band by editing/deleting that file.
- move_until(joint, target, ...): steps a single joint toward `target` in small
  increments and STOPS when it meets resistance (over-current = contact), stalls, hits a
  soft limit, or reaches `target`. This is the "if unsure how far it can go, test it out"
  primitive. A fault on any OTHER joint, a servo error bit, or stale state aborts + holds,
  exactly like the engine. Returns where it stopped and why, plus a suggested_limit.
- autopilot_stop(): stop a running probe (for a second client / e-stop script).

Safety note: switching the mode to "step" is NOT an emergency stop -- it only changes
what the NEXT command does. To halt a moving arm, use the main server's relax (torque
off, always allowed) or cut power. Thresholds here are untuned guesses until measured on
hardware; set_control_mode("autopilot") requires confirm_thresholds_tuned=True so nobody
probes into contact on guessed numbers by accident.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import server as S          # the main MCP server, imported as a library (not modified)
import motion as M          # pure motion/safety logic, reused as-is

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("normacore-autopilot")   # server.py already sent logging to stderr

# --- tunables (ALL untuned guesses: MEASURE ON HARDWARE, then set via env) ----------
# Probe velocity is deliberately low: probing means driving toward possible contact, and
# the main server has no current protection outside this loop, so go gently.
PROBE_VEL_NORM = float(os.environ.get("PROBE_VEL_NORM", "0.10"))     # norm/s for the probed joint
PROBE_STEP_NORM = float(os.environ.get("PROBE_STEP_NORM", "0.03"))   # commanded increment per tick
PROBE_MAX_TRAVEL = float(os.environ.get("PROBE_MAX_TRAVEL", "0.50")) # hard cap on total travel, norm
PROBE_STALL_TICKS = int(os.environ.get("PROBE_STALL_TICKS", "4"))    # ticks of no movement -> stall
PROBE_STALL_EPS = float(os.environ.get("PROBE_STALL_EPS", "0.004"))  # norm; movement below this = stopped
PROBE_SETTLE_TOL = float(os.environ.get("PROBE_SETTLE_TOL", "0.02")) # norm; reached-target tolerance

CONTROL_MODE_FILE = Path(os.environ.get(
    "CONTROL_MODE_FILE", str(Path(__file__).resolve().parent / "data" / "control_mode.json")))

AUTOPILOT_MCP_TRANSPORT = os.environ.get("AUTOPILOT_MCP_TRANSPORT", "stdio")
AUTOPILOT_MCP_HOST = os.environ.get("AUTOPILOT_MCP_HOST", "127.0.0.1")
AUTOPILOT_MCP_PORT = int(os.environ.get("AUTOPILOT_MCP_PORT", "8791"))

_probe_lock = asyncio.Lock()
_abort = asyncio.Event()

ap = FastMCP("normacore-autopilot", host=AUTOPILOT_MCP_HOST, port=AUTOPILOT_MCP_PORT, instructions=(
    "EXPERIMENTAL autopilot mode for the normacore arm, as a separate server. Turn it on "
    "with set_control_mode('autopilot', confirm_thresholds_tuned=True), then use move_until "
    "to drive a joint until it meets resistance instead of guessing the distance. It aborts "
    "and holds on any fault. relax/power on the main server are still the real emergency stop."
))


# ----------------------------------------------------------------------------
# control mode: a tiny file re-read on every move_until call (no restart needed).

def read_control_mode() -> str:
    """Current mode, 'step' (safe default) or 'autopilot'. Missing/invalid file -> 'step'."""
    try:
        data = json.loads(CONTROL_MODE_FILE.read_text(encoding="utf-8"))
        mode = str(data.get("mode", "step")).strip().lower()
        return mode if mode in ("step", "autopilot") else "step"
    except FileNotFoundError:
        return "step"
    except (OSError, ValueError, TypeError) as e:
        log.warning("control mode file unreadable (%s); defaulting to 'step'", e)
        return "step"


def _write_control_mode(mode: str, by: str, note: str) -> None:
    CONTROL_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "set_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "set_by": by, "note": note}
    tmp = CONTROL_MODE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CONTROL_MODE_FILE)


def _thresholds() -> dict:
    cfg = S._safety_config()
    return {
        "current_limit": cfg.current_limit,
        "per_motor_current": cfg.per_motor_current,
        "track_err_max": cfg.track_err_max,
        "track_err_ticks": cfg.track_err_ticks,
        "stale_s": cfg.stale_s,
        "probe_vel_norm": PROBE_VEL_NORM,
        "probe_step_norm": PROBE_STEP_NORM,
        "probe_max_travel": PROBE_MAX_TRAVEL,
        "probe_stall_ticks": PROBE_STALL_TICKS,
        # True if nothing has overridden the documented defaults -> still unmeasured.
        "looks_like_defaults": (cfg.current_limit == 100 and cfg.track_err_max == 0.10
                                and not cfg.per_motor_current),
    }


# ----------------------------------------------------------------------------
# pure probe decision (unit-testable; no I/O). Reuses motion.SafetyMonitor for the
# fault logic, with gripper_id set to the probed joint so that over-current on THAT
# joint is read as "contact" (success), not a crash (abort).

@dataclasses.dataclass
class ProbeConfig:
    joint: int
    stall_ticks: int = PROBE_STALL_TICKS
    stall_eps: float = PROBE_STALL_EPS


@dataclasses.dataclass
class Prober:
    cfg: ProbeConfig
    monitor: M.SafetyMonitor
    _stall: int = 0
    _last: float | None = None

    def update(self, motors: dict[int, dict], commanded: dict[int, float],
               reached_target: bool, state_age_s: float) -> tuple[str | None, str | None]:
        """One tick. Returns (stop_reason, abort_reason).

        stop_reason set  -> the probe finished successfully (reached a limit or the target);
        abort_reason set -> a fault; the caller must hold and report failure;
        both None        -> keep probing.
        """
        abort = self.monitor.check(motors, commanded, state_age_s)
        if abort is not None:
            return None, abort
        if self.monitor.gripper_blocked:            # over-current on the probed joint
            return f"contact: over-current on joint {self.cfg.joint}", None
        pos = motors[self.cfg.joint]["position_norm"]
        if not reached_target and self._last is not None and abs(pos - self._last) < self.cfg.stall_eps:
            self._stall += 1
            if self._stall >= self.cfg.stall_ticks:
                return f"stall: joint {self.cfg.joint} stopped moving before the target", None
        else:
            self._stall = 0
        self._last = pos
        if reached_target:
            return "reached target without contact", None
        return None, None


def _advance(cur: float, target: float, step: float) -> float:
    if cur < target:
        return min(cur + step, target)
    if cur > target:
        return max(cur - step, target)
    return target


# ----------------------------------------------------------------------------
# tools

@ap.tool()
async def set_control_mode(mode: str, confirm_thresholds_tuned: bool = False, note: str = "") -> dict:
    """Select the control mode at runtime (no restart). Fail-safe default is 'step'.

    mode="autopilot": unlocks move_until. Because probing drives toward contact and the
      thresholds here are untuned guesses until measured on hardware, this REQUIRES
      confirm_thresholds_tuned=True (a human acknowledgement).
    mode="step": re-locks move_until and stops any probe running in this process.

    NOTE: switching to 'step' is not an emergency stop -- it only affects the next command.
    To halt a moving arm use the main server's relax, or cut power.
    """
    mode = mode.strip().lower()
    if mode not in ("step", "autopilot"):
        raise RuntimeError("mode must be 'step' or 'autopilot'")
    th = _thresholds()
    if mode == "autopilot" and not confirm_thresholds_tuned:
        raise RuntimeError(
            "refusing to enable autopilot: its current/tracking thresholds are untuned guesses "
            f"({th}). Measure them on the real arm first (run go_to/play_motion and read the "
            "reports), then call set_control_mode('autopilot', confirm_thresholds_tuned=True).")
    _write_control_mode(mode, by="tool", note=note)
    if mode == "step":
        _abort.set()        # stop any in-flight probe in this process
    return {
        "mode": mode,
        "thresholds": th,
        "file": str(CONTROL_MODE_FILE),
        "reminder": "relax/power on the main server are the real emergency stop; mode != stop.",
    }


@ap.tool()
async def get_autopilot_status() -> dict:
    """Current control mode, whether a probe is running, and the (untuned) thresholds."""
    return {
        "mode": read_control_mode(),
        "probe_running": _probe_lock.locked(),
        "thresholds": _thresholds(),
        "file": str(CONTROL_MODE_FILE),
    }


@ap.tool()
async def autopilot_stop() -> str:
    """Stop a running move_until: it holds where it is. (Claude Code runs one tool at a time,
    so this is mainly for a second client / e-stop script; the main server's hold/relax also
    stop the arm.)"""
    _abort.set()
    return "stop requested; a running probe will hold its current pose"


@ap.tool()
async def move_until(joint: str, target: float, step: float = PROBE_STEP_NORM,
                     max_travel: float = PROBE_MAX_TRAVEL, bus: str | None = None) -> dict:
    """Drive ONE joint toward `target` (0..1 normalized) and stop at the first limit it meets.

    Use this instead of guessing a distance: it steps the joint in small increments and stops
    when it meets resistance (over-current = contact), stalls, reaches a soft limit, or reaches
    `target`. The other joints are held in place. A fault on any other joint, a servo error bit,
    or stale robot data aborts and holds.

    joint: motor id or name (e.g. "elbow" or "4"). target: where to probe toward; the direction
      is simply toward that value from where the joint is now. step: commanded increment per tick
      (clamped to the main server's MAX_STEP_NORM). max_travel: hard cap on total travel (norm).

    Requires autopilot mode (set_control_mode). Returns where it stopped, why, how far it
    travelled, and a suggested_limit you can later save to data/limits.json (not written here).
    """
    if read_control_mode() != "autopilot":
        raise RuntimeError("autopilot mode is off; call set_control_mode('autopilot', "
                           "confirm_thresholds_tuned=True) first (or a human can set data/control_mode.json).")
    if _probe_lock.locked():
        raise RuntimeError("another probe is already running; call autopilot_stop first")
    async with _probe_lock:
        _abort.clear()
        buses = await S.read_buses()
        serial = S.pick_bus(buses, bus)
        await S.guard_motion(serial)                 # refuse while mirrored / on a leader bus
        motors = buses[serial]["motors"]
        names = {v["joint"]: k for k, v in motors.items()}
        mid = int(joint) if str(joint).isdigit() else names.get(str(joint))
        if mid is None or mid not in motors:
            raise RuntimeError(f"unknown joint {joint!r}; known: {list(names)}")
        # Same precondition as the engine: every joint calibrated and no (non-voltage) error bit,
        # so holding the others is well-defined and positions are all known.
        bad = {m["joint"]: ("not calibrated" if not m["calibrated"] else f"error 0x{m['error_status']:02X}")
               for m in motors.values()
               if not m["calibrated"] or (m["error_status"] & ~M.IGNORED_STATUS_BITS)}
        if bad:
            raise RuntimeError(f"refusing to probe: {bad}")

        ids = sorted(motors)
        start = {j: motors[j]["position_norm"] for j in ids}
        ranges = {j: (motors[j]["range_min"], motors[j]["range_max"]) for j in ids}
        limits = S._soft_limits()

        # Clamp the target to [0,1], the joint's soft limit, and the max-travel cap.
        tgt = max(0.0, min(1.0, float(target)))
        lo, hi = limits.get(mid, (0.0, 1.0))
        tgt_lim = min(max(tgt, lo), hi)
        capped = start[mid] + max(-max_travel, min(max_travel, tgt_lim - start[mid]))
        target_total = round(capped, 6)
        clamp_notes = {}
        if abs(tgt_lim - tgt) > 1e-9:
            clamp_notes["soft_limit"] = f"{tgt:.3f} -> {tgt_lim:.3f} (limits [{lo}, {hi}])"
        if abs(capped - tgt_lim) > 1e-9:
            clamp_notes["max_travel"] = f"{tgt_lim:.3f} -> {target_total:.3f} (cap {max_travel})"

        # Advance the command at the probe velocity so it never races ahead of the servo
        # (which would trip a false tracking-error abort). `step` and MAX_STEP_NORM are caps.
        per_tick = min(PROBE_VEL_NORM / S.MOTION_RATE_HZ, abs(float(step)), S.MAX_STEP_NORM)
        per_tick = max(per_tick, 1e-4)

        # Probe config: SafetyMonitor with gripper_id = probed joint -> over-current there is
        # "contact" (handled by Prober), not an abort. All other faults behave as in the engine.
        safety = dataclasses.replace(S._safety_config(), gripper_id=mid)
        prober = Prober(ProbeConfig(joint=mid), M.SafetyMonitor(safety))

        # Speed/accel/torque once up front (like _run_motion). Probed joint at PROBE_VEL; the
        # others only need to hold, so a low speed is fine.
        speeds = [(j, M.steps_per_s(PROBE_VEL_NORM if j == mid else 0.0,
                                    S.range_size(*ranges[j])).to_bytes(2, "little")) for j in ids]
        c = await S.client()
        await S.send_commands(c, [
            S._sync_write(serial, S.RAM_GOAL_SPEED, speeds),
            S._sync_write(serial, S.RAM_ACC, [(j, bytes([S.MOTION_ACCEL])) for j in ids]),
            S._sync_write(serial, S.RAM_TORQUE_ENABLE, [(j, b"\x01") for j in ids]),
        ])

        period = 1.0 / S.MOTION_RATE_HZ
        cmd = start[mid]
        done_cmd = cmd == target_total
        stop_reason = abort_reason = None
        ticks = late = 0
        last_stamp, last_change = None, time.monotonic()
        latest = motors
        # Generous deadline so a stuck loop can never run forever.
        deadline = time.monotonic() + abs(target_total - start[mid]) / max(PROBE_VEL_NORM, 1e-6) + 5.0
        estop_mode = None
        t0 = time.monotonic()
        try:
            while True:
                if _abort.is_set():
                    abort_reason = "stopped by autopilot_stop / mode set to step"
                    break
                latch = await S.estop_latch()          # e-stop pressed in estop.py
                if latch:
                    estop_mode = latch["mode"]
                    abort_reason = f"E-STOP {estop_mode.upper()} pressed by {latch.get('by') or '?'}"
                    break
                if time.monotonic() > deadline:
                    stop_reason = "timeout: no contact and target not reached"
                    break
                if not done_cmd:
                    cmd = _advance(cmd, target_total, per_tick)
                    done_cmd = abs(cmd - target_total) < 1e-9
                commanded = {j: (cmd if j == mid else start[j]) for j in ids}
                await S.send_commands(c, [S._sync_write(serial, S.RAM_GOAL_POSITION, [
                    (j, S.from_norm(commanded[j], *ranges[j]).to_bytes(2, "little")) for j in ids])])
                state = (await S.read_buses())[serial]
                latest = state["motors"]
                if state["stamp_ns"] != last_stamp:
                    last_stamp, last_change = state["stamp_ns"], time.monotonic()
                reached = done_cmd and abs(latest[mid]["position_norm"] - target_total) <= PROBE_SETTLE_TOL
                stop_reason, abort_reason = prober.update(
                    latest, commanded, reached, time.monotonic() - last_change)
                if stop_reason or abort_reason:
                    break
                ticks += 1
                delay = t0 + ticks * period - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    late += 1
        except BaseException:
            await S._hold_shielded(serial)
            raise

        if estop_mode != "relax":                    # after a RELAX e-stop the arm must stay limp
            await S._hold_shielded(serial)           # hold wherever we ended up
        if abort_reason:
            log.warning("probe aborted: %s", abort_reason)

        final_pos = latest[mid]["position_norm"]
        contacted = stop_reason is not None and stop_reason.startswith(("contact", "stall"))
        return {
            "completed": abort_reason is None,
            "stop_reason": stop_reason,
            "abort_reason": abort_reason,
            "bus": serial,
            "joint": motors[mid]["joint"],
            "start": round(start[mid], 3),
            "final": round(final_pos, 3),
            "target": target_total,
            "travelled": round(abs(final_pos - start[mid]), 3),
            # Where a future soft limit could go, IF this was a real obstacle (NOT auto-saved).
            "suggested_limit": round(final_pos, 3) if contacted else None,
            "clamped": clamp_notes or None,
            "ticks": ticks,
            "late_ticks": late,
            "max_tracking_error": {motors[j]["joint"]: round(e, 3)
                                   for j, e in prober.monitor.max_track_err.items()},
        }


if __name__ == "__main__":
    log.info("starting autopilot MCP server (transport=%s, station=%s)",
             AUTOPILOT_MCP_TRANSPORT, S.STATION_HOST)
    ap.run(transport=AUTOPILOT_MCP_TRANSPORT)
