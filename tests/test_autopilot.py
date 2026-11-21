"""Tests for the SEPARATE autopilot server (move_until probe + control mode).

usage: python tests/test_autopilot.py <normacore-mcp dir> <norma-core repo>

Self-contained: it brings its own tiny simulated station (a bus whose servos glide
toward their goal and spike their current when they hit a configurable "wall"), so it
does not touch the main offline_test.py. Delete autopilot.py and this file to remove the
feature with no trace.
"""
import asyncio, os, sys, tempfile

sys.path.insert(0, sys.argv[1])
os.environ["NORMA_CORE_REPO"] = sys.argv[2]
os.environ["FOLLOWER_BUS"] = "FOLLOW"
# fast + deterministic: high tick rate, high probe velocity so a probe is a handful of ticks.
os.environ.update({"MOTION_RATE_HZ": "100", "PROBE_VEL_NORM": "3.0", "STALE_STATE_S": "0.5"})
_TMP = tempfile.mkdtemp(prefix="normacore-ap-")
os.environ["CONTROL_MODE_FILE"] = os.path.join(_TMP, "control_mode.json")
os.environ["LIMITS_FILE"] = os.path.join(_TMP, "limits.json")   # default: no soft limits

import server as S
import motion as M
import autopilot as A
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.station import commands, drivers

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ---- tiny simulated station ------------------------------------------------
# One follower bus. Each read advances every torqued motor up to STEP raw toward its goal;
# a motor with a "wall" clamps at it and spikes its current when crossed (either direction).
STEP = 220
MODEL = {}
SIM = {"stamp": 1, "freeze": False, "reads": 0, "inject": None}

def reset(walls=None):
    walls = walls or {}
    MODEL.clear()
    for j in range(1, 9):
        MODEL[j] = {"pos": 2000, "goal": 2000, "torque": 1, "cur": 10, "err": 0,
                    "rmin": 1000, "rmax": 3000, "frozen": True, "wall": walls.get(j)}
    SIM.update(stamp=1, freeze=False, reads=0, inject=None)

def motor_state(pos, torque, current, err):
    b = bytearray(0x50)
    b[0x38:0x3A] = pos.to_bytes(2, "little"); b[0x28] = torque
    b[0x45:0x47] = current.to_bytes(2, "little"); b[0x40] = err
    return bytes(b)

def enc_st3215():
    SIM["reads"] += 1
    if not SIM["freeze"]:
        SIM["stamp"] += 1
    for m in MODEL.values():
        m["cur"] = 10
        if not m["torque"]:
            continue
        d = max(-STEP, min(STEP, m["goal"] - m["pos"]))
        new = m["pos"] + d
        w = m["wall"]
        if w is not None and (m["pos"] <= w <= new or new <= w <= m["pos"]):
            new, m["cur"] = w, 400          # hit the wall: clamp + current spike
        m["pos"] = new
    inj = SIM["inject"]                      # delayed fault on some motor, mid-probe
    if inj and SIM["reads"] >= inj["after"]:
        MODEL[inj["motor"]][inj["field"]] = inj["value"]
    ms = [st3215.InferenceState_MotorState(id=j, state=motor_state(m["pos"], m["torque"], m["cur"], m["err"]),
                                           range_min=m["rmin"], range_max=m["rmax"], range_freezed=m["frozen"])
          for j, m in MODEL.items()]
    bus = st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number="FOLLOW"), motors=ms,
                                         monotonic_stamp_ns=SIM["stamp"])
    return st3215.InferenceState(buses=[bus]).encode()

MIRROR_PRESENT = {"v": False}      # True -> FOLLOW is mirrored (teach mode on)

async def fake_read_last(q, n=1, timeout=None):
    if q == "st3215/inference":
        return [memoryview(enc_st3215())]
    if q == S.MIRRORING_QUEUE:
        if not MIRROR_PRESENT["v"]:
            raise S.QueueMissing(q)
        from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
        def mb(s): return mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=s)
        st = mirroring.InferenceState(
            modes=[mirroring.InferenceState_Bus(id=mb("LEAD"), mode=mirroring.BusMode.BR_LEADER)],
            mirroring=[mirroring.InferenceState_Mirroring(
                source=mirroring.InferenceState_Bus(id=mb("LEAD"), mode=mirroring.BusMode.BR_LEADER),
                targets=[mirroring.InferenceState_Bus(id=mb("FOLLOW"), mode=mirroring.BusMode.BR_FOLLOWER)])])
        return [memoryview(mirroring.RxEnvelope(state=st).encode())]
    raise S.QueueMissing(q)

async def fake_send(_c, cmds):
    for c in cmds:
        r = commands.DriverCommandReader(memoryview(c.encode()))
        if r.get_type() != drivers.StationCommandType.STC_ST3215_COMMAND:
            continue
        cmd = st3215.CommandReader(r.get_body()); sw = cmd.get_sync_write()
        if not sw:
            continue
        if sw.get_address() == S.RAM_TORQUE_ENABLE:
            for mw in sw.get_motors():
                MODEL[mw.get_motor_id()]["torque"] = bytes(mw.get_value())[0]
        if sw.get_address() == S.RAM_GOAL_POSITION:
            for mw in sw.get_motors():
                MODEL[mw.get_motor_id()]["goal"] = int.from_bytes(bytes(mw.get_value()), "little")

async def fake_client(): return None
S.read_last, S.send_commands, S.client = fake_read_last, fake_send, fake_client

async def expect_error(name, coro, needle):
    try:
        await coro
        check(name, False, "no error raised")
    except RuntimeError as e:
        check(name, needle in str(e), str(e))

def norm(raw):  # for this bus (1000..3000): raw -> 0..1
    return (raw - 1000) / 2000


# ---- pure-logic tests (no station) -----------------------------------------
def pure_tests():
    print("--- pure logic ---")
    check("_advance moves up toward target", A._advance(0.5, 0.8, 0.1) == 0.6)
    check("_advance moves down toward target", A._advance(0.5, 0.2, 0.1) == 0.4)
    check("_advance never overshoots", A._advance(0.78, 0.8, 0.1) == 0.8)

    cfg = M.SafetyConfig(current_limit=100, gripper_id=4, track_err_max=0.1, track_err_ticks=3, stale_s=0.5)
    def motors(**over):
        base = {j: {"joint": f"j{j}", "error_status": 0, "current": 10, "position_norm": 0.5} for j in (2, 4, 8)}
        for k, v in over.items():
            j, field = k.split("_", 1)
            base[int(j[1:])][field] = v
        return base
    cmd = {2: 0.5, 4: 0.5, 8: 0.5}

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    check("keeps probing when all is well", p.update(motors(), cmd, False, 0.0) == (None, None))

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    stop, abort = p.update(motors(j4_current=300), cmd, False, 0.0)
    check("over-current on PROBED joint = contact (success)", stop and "contact" in stop and abort is None, (stop, abort))

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    stop, abort = p.update(motors(j2_current=300), cmd, False, 0.0)
    check("over-current on ANOTHER joint = abort", abort and "current" in abort and stop is None, (stop, abort))

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    stop, abort = p.update(motors(j4_error_status=0x20), cmd, False, 0.0)
    check("servo error bit on probed joint = abort (not contact)", abort and "0x20" in abort, (stop, abort))

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    check("stale state aborts", "old" in (p.update(motors(), cmd, False, 0.9)[1] or ""))

    # first observation sets the baseline; stall_ticks=2 then needs two more still ticks.
    p = A.Prober(A.ProbeConfig(joint=4, stall_ticks=2, stall_eps=0.004), M.SafetyMonitor(cfg))
    res = [p.update(motors(j4_position_norm=0.60), cmd, False, 0.0) for _ in range(3)]
    check("stall detected only after N still ticks", res[0] == (None, None) and res[1] == (None, None)
          and res[2][0] and "stall" in res[2][0], res)

    p = A.Prober(A.ProbeConfig(joint=4), M.SafetyMonitor(cfg))
    stop, abort = p.update(motors(), cmd, True, 0.0)
    check("reached target without contact", stop and "reached target" in stop and abort is None, (stop, abort))

    # control-mode file: fail-safe default + round-trip
    import json, pathlib
    f = pathlib.Path(os.environ["CONTROL_MODE_FILE"])
    f.unlink(missing_ok=True)
    check("missing mode file -> step", A.read_control_mode() == "step")
    f.write_text("{ not json", encoding="utf-8")
    check("garbage mode file -> step", A.read_control_mode() == "step")
    A._write_control_mode("autopilot", "test", "")
    check("written mode is read back", A.read_control_mode() == "autopilot")
    check("file has expected shape", set(json.loads(f.read_text())) >= {"mode", "set_at", "set_by"})
    f.unlink(missing_ok=True)


# ---- integration tests (simulated station) ---------------------------------
async def integration_tests():
    print("\n--- integration (simulated servos) ---")
    import pathlib
    mode_file = pathlib.Path(os.environ["CONTROL_MODE_FILE"])

    # mode gating: move_until refused in step mode, and autopilot needs the ack flag
    mode_file.unlink(missing_ok=True)
    reset()
    await expect_error("move_until refused in step mode", A.move_until("elbow", 0.9), "autopilot mode is off")
    await expect_error("autopilot needs threshold ack", A.set_control_mode("autopilot"), "untuned")
    r = await A.set_control_mode("autopilot", confirm_thresholds_tuned=True)
    check("autopilot enabled with ack", r["mode"] == "autopilot" and A.read_control_mode() == "autopilot", str(r))

    # contact: probe elbow (id 4) up toward 1.0; a wall at raw 2600 (norm 0.8) stops it
    reset({4: 2600})
    r = await A.move_until("elbow", 1.0)
    check("probe stops at contact", r["completed"] and r["stop_reason"] and "contact" in r["stop_reason"], str(r))
    check("stopped near the wall", abs(r["final"] - norm(2600)) <= 0.03, r["final"])
    check("suggests the discovered limit", r["suggested_limit"] is not None and abs(r["suggested_limit"] - 0.8) <= 0.03, r)
    check("other joints were held", all(abs(MODEL[j]["pos"] - 2000) <= STEP for j in MODEL if j != 4),
          {j: MODEL[j]["pos"] for j in MODEL})
    check("ended holding (torque on, goal≈pos)", all(m["torque"] == 1 for m in MODEL.values())
          and abs(MODEL[4]["goal"] - MODEL[4]["pos"]) <= STEP, MODEL[4])

    # no wall: it simply reaches the requested target
    reset()
    r = await A.move_until("elbow", 0.7)
    check("no obstacle -> reaches target", r["completed"] and "reached target" in r["stop_reason"], str(r))
    check("reached target has no suggested limit", r["suggested_limit"] is None, r)
    check("landed on target", abs(r["final"] - 0.7) <= 0.03, r["final"])

    # max_travel cap: target far away but cap stops it short (no wall)
    reset()
    r = await A.move_until("elbow", 1.0, max_travel=0.1)
    check("max_travel caps the probe", r["travelled"] <= 0.12 and r["clamped"] and "max_travel" in r["clamped"], str(r))

    # soft limit clamps the target
    pathlib.Path(os.environ["LIMITS_FILE"]).write_text('{"4": [0.0, 0.65]}', encoding="utf-8")
    reset()
    r = await A.move_until("elbow", 1.0)
    check("soft limit clamps target", abs(r["target"] - 0.65) <= 1e-6 and r["clamped"] and "soft_limit" in r["clamped"], str(r))
    pathlib.Path(os.environ["LIMITS_FILE"]).unlink(missing_ok=True)

    # pre-existing fault anywhere -> refused upfront (like the engine)
    reset()
    MODEL[2]["err"] = 0x20
    await expect_error("pre-existing fault refuses the probe", A.move_until("elbow", 0.9), "refusing to probe")

    # fault appearing on ANOTHER joint DURING the probe aborts and holds
    reset()
    SIM["inject"] = {"motor": 2, "field": "err", "value": 0x20, "after": 4}
    r = await A.move_until("elbow", 0.9)
    check("mid-probe fault on another joint aborts", not r["completed"] and "0x20" in r["abort_reason"], str(r))
    check("held after abort", all(m["torque"] == 1 for m in MODEL.values()), MODEL)

    # refused while mirrored (teach mode on)
    reset()
    MIRROR_PRESENT["v"] = True
    await expect_error("probe refused while mirrored", A.move_until("elbow", 0.9), "being mirrored")
    MIRROR_PRESENT["v"] = False

    # unknown joint
    reset()
    await expect_error("unknown joint rejected", A.move_until("nope", 0.5), "unknown joint")

    # set_control_mode('step') aborts a running probe
    reset({4: 3000})   # wall at the very end so the probe runs a while
    task = asyncio.create_task(A.move_until("elbow", 1.0, max_travel=0.5))
    await asyncio.sleep(0.05)
    await A.set_control_mode("step")
    r = await task
    check("mode->step stops a running probe", not r["completed"] and "step" in (r["abort_reason"] or ""), str(r))

    # status + tool registration
    await A.set_control_mode("autopilot", confirm_thresholds_tuned=True)
    st = await A.get_autopilot_status()
    check("status reports mode + thresholds", st["mode"] == "autopilot" and "current_limit" in st["thresholds"], str(st))
    tools = sorted(t.name for t in await A.ap.list_tools())
    check("autopilot tools registered", {"set_control_mode", "move_until", "autopilot_stop", "get_autopilot_status"} <= set(tools), tools)
    check("does NOT register the main server's tools", "go_to" not in tools and "look" not in tools, tools)


async def main():
    pure_tests()
    await integration_tests()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)

asyncio.run(main())
