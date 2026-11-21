"""Offline test: runs every MCP tool against a simulated Station (no robot needed).

usage: python tests/offline_test.py <normacore-mcp dir> <norma-core repo>

The fake station reacts to mirroring commands the way motors-mirroring does in the
norma-core source: START -> link + torque on targets (inference/mod.rs:68-88),
STOP -> unlink + torque off on the freed bus (inference/mod.rs:91-131).
"""
import asyncio, io, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["NORMA_CORE_REPO"] = sys.argv[2]
os.environ["FOLLOWER_BUS"] = "FOLLOW"
# fast settings so the motion-engine tests run in a few seconds
os.environ.update({"MOTION_RATE_HZ": "100", "MAX_VEL_NORM": "2.0", "SETTLE_S": "0.5", "TRACK_ERR_TICKS": "5"})
import tempfile
_TMP = tempfile.mkdtemp(prefix="normacore-test-")
os.environ["CAMERAS_FILE"] = os.path.join(_TMP, "cameras.json")  # never write into the repo's data/
import server as S
from PIL import Image as PI
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
from target.gen_python.protobuf.station import inference, drivers, commands

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ---- simulated station model ------------------------------------------------
MODEL = {
    "FOLLOW": {i: {"pos": 2048 + 10 * i, "torque": 0, "rmin": 1000, "rmax": 3000, "frozen": True} for i in range(1, 9)},
    "LEAD":   {i: {"pos": 2048 + 10 * i, "torque": 0, "rmin": 1000, "rmax": 3000, "frozen": i != 3} for i in range(1, 9)},
}
MIRROR = {"present": True, "links": {}, "modes": {}}  # links: follower -> leader

def motor_state(pos, torque, current=0, err=0):
    b = bytearray(0x50)
    b[0x38:0x3A] = pos.to_bytes(2, "little"); b[0x28] = torque
    b[0x45:0x47] = current.to_bytes(2, "little"); b[0x40] = err
    return bytes(b)

# Simulated follower servos (off until the motion-engine tests): each st3215 read moves every
# torqued follower motor up to SIM["step"] steps toward its goal and advances the bus stamp.
SIM = {"on": False, "step": 300, "stamp": 1, "freeze": False, "reads": 0,
       "spike": None, "err": None, "blocked": set(), "grip_stop": None}

def sim_tick():
    if not SIM["on"]:
        return
    SIM["reads"] += 1
    if not SIM["freeze"]:
        SIM["stamp"] += 1
    for j, m in MODEL["FOLLOW"].items():
        m["cur"], m["err"] = 10, 0
        goal = m.get("goal")
        if goal is None or not m["torque"] or j in SIM["blocked"]:
            continue
        d = max(-SIM["step"], min(SIM["step"], goal - m["pos"]))
        new = m["pos"] + d
        gs = SIM["grip_stop"]
        if j == 8 and gs is not None and goal < gs and new <= gs:   # jaws closed on an object
            new, m["cur"] = gs, 300
        m["pos"] = new
    for key in ("spike", "err"):
        ev = SIM[key]
        if ev and SIM["reads"] >= ev["after"]:
            MODEL["FOLLOW"][ev["motor"]]["cur" if key == "spike" else "err"] = ev["value"]

def enc_st3215():
    sim_tick()
    buses = []
    for serial, motors in MODEL.items():
        ms = [st3215.InferenceState_MotorState(id=i, state=motor_state(m["pos"], m["torque"], m.get("cur", 0), m.get("err", 0)),
                                               range_min=m["rmin"], range_max=m["rmax"], range_freezed=m["frozen"]) for i, m in motors.items()]
        buses.append(st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number=serial), motors=ms,
                                                    monotonic_stamp_ns=SIM["stamp"]))
    return st3215.InferenceState(buses=buses).encode()

def mb(s): return mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=s)

def enc_mirroring():
    by_leader = {}
    for f, l in MIRROR["links"].items():
        by_leader.setdefault(l, []).append(f)
    st = mirroring.InferenceState(
        modes=[mirroring.InferenceState_Bus(id=mb(s), mode=m) for s, m in MIRROR["modes"].items()],
        mirroring=[mirroring.InferenceState_Mirroring(
            source=mirroring.InferenceState_Bus(id=mb(l), mode=mirroring.BusMode.BR_LEADER),
            targets=[mirroring.InferenceState_Bus(id=mb(f), mode=mirroring.BusMode.BR_FOLLOWER) for f in fs])
            for l, fs in by_leader.items()])
    return mirroring.RxEnvelope(state=st).encode()

buf = io.BytesIO(); PI.new("RGB", (64, 48), (200, 30, 30)).save(buf, "JPEG"); JPEG = buf.getvalue()
STATIC = {
    "inference-states": inference.InferenceRx(entries=[
        inference.InferenceRx_Entry(queue="usbvideo/abc", type=drivers.QueueDataType.QDT_USB_VIDEO_FRAMES),
        inference.InferenceRx_Entry(queue="st3215/inference", type=drivers.QueueDataType.QDT_ST3215_INFERENCE)]).encode(),
    "usbvideo/abc": usbvideo.RxEnvelope(type=usbvideo.RxEnvelopeType.ET_FRAMES,
        camera=usbvideo.Camera(product="C270 HD WEBCAM"),
        frames=frame.FramesPack(frames_data=[JPEG])).encode(),
}

async def fake_read_last(q, n=1):
    if q == "st3215/inference":
        return [memoryview(enc_st3215())]
    if q == S.MIRRORING_QUEUE:
        if not MIRROR["present"]:
            raise S.QueueMissing(q)
        return [memoryview(enc_mirroring())]
    v = STATIC[q]
    return [memoryview(x) for x in v] if isinstance(v, list) else [memoryview(v)]  # list = several entries, oldest first

def cam_frame(age_s, product="C270 HD WEBCAM"):
    """usbvideo frames envelope whose frame was captured age_s seconds ago (station wall clock)."""
    import time
    stamp = frame.FrameStamp(local_stamp_ns=time.time_ns() - int(age_s * 1e9))
    return usbvideo.RxEnvelope(type=usbvideo.RxEnvelopeType.ET_FRAMES, camera=usbvideo.Camera(product=product),
                               stamp=stamp, frames=frame.FramesPack(frames_data=[JPEG], stamps=[stamp])).encode()

def cam_event(kind):
    return usbvideo.RxEnvelope(type=kind, camera=usbvideo.Camera(product="C270 HD WEBCAM")).encode()

SENT = []
async def fake_send(_c, cmds):
    for c in cmds:
        r = commands.DriverCommandReader(memoryview(c.encode()))
        SENT.append(r)
        if r.get_type() == drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND:
            m = mirroring.CommandReader(r.get_body())
            src = m.get_source().get_unique_id()
            if m.get_type() == mirroring.CommandType.CT_START_MIRROR:
                MIRROR["modes"][src] = mirroring.BusMode.BR_LEADER
                for t in m.get_targets():
                    f = t.get_unique_id()
                    MIRROR["links"][f] = src; MIRROR["modes"][f] = mirroring.BusMode.BR_FOLLOWER
                    for mm in MODEL[f].values(): mm["torque"] = 1
            else:  # STOP: bus may be source or target; freed bus goes limp
                freed = [src] + [f for f, l in MIRROR["links"].items() if l == src]
                for f in list(MIRROR["links"]):
                    if f == src or MIRROR["links"][f] == src:
                        del MIRROR["links"][f]
                for b in freed:
                    for mm in MODEL[b].values(): mm["torque"] = 0
        elif r.get_type() == drivers.StationCommandType.STC_ST3215_COMMAND:
            cmd = st3215.CommandReader(r.get_body()); sw = cmd.get_sync_write()
            if sw and sw.get_address() == S.RAM_TORQUE_ENABLE:
                for mw in sw.get_motors():
                    MODEL[cmd.get_target_bus_serial()][mw.get_motor_id()]["torque"] = bytes(mw.get_value())[0]
            if sw and sw.get_address() == S.RAM_GOAL_POSITION:
                for mw in sw.get_motors():
                    MODEL[cmd.get_target_bus_serial()][mw.get_motor_id()]["goal"] = int.from_bytes(bytes(mw.get_value()), "little")

async def fake_client(): return None
S.read_last, S.send_commands, S.client = fake_read_last, fake_send, fake_client

def last_goal_write():
    for r in reversed(SENT):
        if r.get_type() == drivers.StationCommandType.STC_ST3215_COMMAND:
            cmd = st3215.CommandReader(r.get_body()); sw = cmd.get_sync_write()
            if sw and sw.get_address() == S.RAM_GOAL_POSITION:
                return cmd.get_target_bus_serial(), [(m.get_motor_id(), int.from_bytes(bytes(m.get_value()), "little")) for m in sw.get_motors()]

async def expect_error(name, coro, needle):
    try:
        await coro
        check(name, False, "no error raised")
    except RuntimeError as e:
        check(name, needle in str(e), str(e))

async def main():
    # --- normalization matches Station's mirroring math (normalize.rs / mirror.rs) ---
    check("norm plain range", S.to_norm(2000, 1000, 3000) == 0.5)
    check("norm clamps below range", S.to_norm(500, 1000, 3000) == 0.0)
    # wrapping range 3900 -> 200 (size 396): pos 0 is 196 steps in
    check("wrap range size", S.range_size(3900, 200) == 396)
    check("wrap norm", abs(S.to_norm(0, 3900, 200) - 196 / 396) < 1e-9)
    check("wrap from_norm wraps past 4095", S.from_norm(196 / 396, 3900, 200) == 0)
    check("wrap from_norm end", S.from_norm(1.0, 3900, 200) == 200)

    # --- perception + state ---
    lc = await S.list_cameras()
    check("cameras discovered", [c["queue"] for c in lc["cameras"]] == ["usbvideo/abc"] and lc["cameras"][0]["product"] == "C270 HD WEBCAM", str(lc))
    out = await S.look()
    check("look returns label + image", isinstance(out[0], str) and hasattr(out[1], "data"))
    bad = await S.look("5")
    check("look bad index gives message, not IndexError", len(bad) == 1 and "only 1 found" in bad[0], str(bad))
    check("look unknown name gives message", "No camera matching" in (await S.look("nope"))[0])

    # --- frame freshness: camera queues persist, so the "latest" frame can be old ---
    saved = dict(STATIC)
    STATIC["usbvideo/abc"] = cam_frame(0.2)
    out = await S.look()
    check("fresh frame: age shown, not stale", "0.2 s old" in out[0] and "STALE" not in out[0], out[0])
    STATIC["usbvideo/abc"] = cam_frame(3 * 3600)
    out = await S.look()
    check("old frame flagged STALE with age", "STALE" in out[0] and "3.0 h old" in out[0] and "NOT the current scene" in out[0], out[0])
    check("stale frame still returned for diagnosis", hasattr(out[1], "data"))
    STATIC["usbvideo/abc"] = [cam_frame(0.5), cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_DISCONNECTED)]
    out = await S.look()
    check("disconnect after last frame flagged STALE", "STALE: camera reported disconnected" in out[0], out[0])
    STATIC["usbvideo/abc"] = [cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_DISCONNECTED), cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_CONNECTED), cam_frame(0.1)]
    out = await S.look()
    check("reconnected camera with fresh frame is fine", "STALE" not in out[0], out[0])
    STATIC["usbvideo/abc"] = cam_frame(-60)
    out = await S.look()
    check("future timestamp reported as clock mismatch, not stale", "clocks disagree" in out[0] and "STALE" not in out[0], out[0])
    STATIC["usbvideo/abc"] = cam_frame(-1.0)
    out = await S.look()
    check("small clock skew still reads as fresh", "0.0 s old" in out[0], out[0])

    # --- labels keep the index from list_cameras ---
    STATIC["usbvideo/def"] = cam_frame(0.1, product="GRIPPER CAM")
    STATIC["inference-states"] = inference.InferenceRx(entries=[
        inference.InferenceRx_Entry(queue=q, type=drivers.QueueDataType.QDT_USB_VIDEO_FRAMES)
        for q in ("usbvideo/abc", "usbvideo/def")]).encode()
    out = await S.look("1")
    check("look('1') labels it camera 1", out[0].startswith("camera 1: GRIPPER CAM"), out[0])
    out = await S.look("def")
    check("look by name keeps real index", out[0].startswith("camera 1:"), out[0])
    out = await S.look()
    check("look all returns both cameras in order", out[0].startswith("camera 0:") and out[2].startswith("camera 1:"), str(out[::2]))

    # --- camera roles ---
    r = await S.set_camera_role("desk", "0")
    check("set_camera_role by index", r["camera"] == "usbvideo/abc" and os.path.exists(os.environ["CAMERAS_FILE"]), str(r))
    r = await S.set_camera_role("gripper", "def")
    check("set_camera_role by hash prefix", r["camera"] == "usbvideo/def", str(r))
    out = await S.look("desk")
    check("look('desk') returns the desk camera with role in label", len(out) == 2 and out[0].startswith("camera 0 [desk]:"), str(out[::2]))
    out = await S.look("GRIPPER")
    check("roles are case-insensitive", out[0].startswith("camera 1 [gripper]: GRIPPER CAM"), out[0])
    lc = await S.list_cameras()
    check("list_cameras shows roles", [c["role"] for c in lc["cameras"]] == ["desk", "gripper"] and lc["role_problems"] == [], str(lc))
    r = await S.set_camera_role("desk", "usbvideo/def")
    check("one role per camera: reassigning drops the old role", r["saved_roles"] == {"desk": "usbvideo/def"}, str(r))
    out = await S.look("gripper")
    check("unknown role gives a clear message", "No camera matching 'gripper'" in out[0] and "desk" in out[0], out[0])
    r = await S.set_camera_role("desk", "none")
    check("'none' removes a role", r["saved_roles"] == {}, str(r))
    await expect_error("role 'all' rejected", S.set_camera_role("all", "0"), "role must be")
    await expect_error("ambiguous prefix rejected", S.set_camera_role("desk", "usbvideo/"), "matches no single camera")
    os.environ["CAMERAS"] = "desk=abc,top=ffff"
    lc = await S.list_cameras()
    check("CAMERAS env works as default", lc["cameras"][0]["role"] == "desk", str(lc))
    check("env role for missing camera reported", any("'top'" in p for p in lc["role_problems"]), str(lc["role_problems"]))
    await S.set_camera_role("main", "abc")
    lc = await S.list_cameras()
    check("two roles on one camera reported", any("several roles" in p for p in lc["role_problems"]), str(lc["role_problems"]))
    out = await S.look()
    check("look surfaces role problems", out[0].startswith("camera role problems:"), out[0])
    del os.environ["CAMERAS"]; await S.set_camera_role("main", "none")

    STATIC.clear(); STATIC.update(saved)
    st = await S.get_robot_state()
    check("state has mirroring info", st["mirroring"]["available"] and st["mirroring"]["links"] == [])
    check("follower picked from env", st["follower_bus"] == "FOLLOW")
    check("warns about unfrozen calibration", len(st["warnings"]) == 1 and "LEAD" in st["warnings"][0] and "upper_arm_roll" in st["warnings"][0], str(st["warnings"]))
    check("range_frozen exposed", st["buses"]["FOLLOW"]["motors"][1]["range_frozen"] is True)

    # --- motion with no mirroring ---
    r = await S.move_joints({"1": 0.9, "elbow": 0.5})
    check("move clamps to MAX_STEP_NORM", r["result"]["base_yaw"]["clamped"] and r["result"]["base_yaw"]["to"] == round(0.529 + 0.15, 3))
    bus, goals = last_goal_write()
    check("goal written to follower", bus == "FOLLOW" and goals[0][0] == 1)
    await expect_error("refuses explicit leader bus once roles known",
                       (lambda: (MIRROR["modes"].__setitem__("LEAD", mirroring.BusMode.BR_LEADER), S.move_joints({"1": 0.5}, bus="LEAD"))[1])(),
                       "LEADER")

    # --- teach mode on: refused when poses differ ---
    MODEL["LEAD"][2]["pos"] = 2800
    r = await S.teach_mode(True)
    check("teach_mode refuses big pose diff", r["teach_mode"] == "refused" and "shoulder" in r["problems"], str(r))
    MODEL["LEAD"][2]["pos"] = MODEL["FOLLOW"][2]["pos"]
    r = await S.teach_mode(True)
    check("teach_mode on confirmed", r["teach_mode"] == "on" and r["leader"] == "LEAD", str(r))
    start = [mirroring.CommandReader(x.get_body()) for x in SENT
             if x.get_type() == drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND][-1]
    check("START command shape", start.get_type() == mirroring.CommandType.CT_START_MIRROR
          and start.get_source().get_unique_id() == "LEAD"
          and [t.get_unique_id() for t in start.get_targets()] == ["FOLLOW"])

    # --- guard while mirrored ---
    await expect_error("move refused while mirrored", S.move_joints({"1": 0.5}), "teach_mode(on=False)")
    await expect_error("nudge refused while mirrored", S.nudge_joints({"elbow": 0.05}), "teach_mode(on=False)")
    await expect_error("hold refused while mirrored", S.hold(), "teach_mode(on=False)")
    check("relax still allowed while mirrored", "torque disabled" in await S.relax())
    for mm in MODEL["FOLLOW"].values(): mm["torque"] = 1

    # --- teach mode off: stop, wait for station torque-off, then hold ---
    r = await S.teach_mode(False)
    check("teach_mode off + holding", r["teach_mode"] == "off" and r["holding"] and r["note"] is None, str(r))
    stop = [mirroring.CommandReader(x.get_body()) for x in SENT
            if x.get_type() == drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND][-1]
    check("STOP uses follower as source (like web UI)", stop.get_type() == mirroring.CommandType.CT_STOP_MIRROR
          and stop.get_source().get_unique_id() == "FOLLOW")
    check("follower torque back on after hold", all(m["torque"] == 1 for m in MODEL["FOLLOW"].values()))
    r = await S.move_joints({"1": 0.5})
    check("move allowed after teach off", r["bus"] == "FOLLOW")

    # --- mirroring driver absent: motion allowed, teach_mode errors ---
    MIRROR["present"] = False; MIRROR["modes"].clear()
    r = await S.move_joints({"1": 0.5})
    check("move allowed when mirroring queue missing", r["bus"] == "FOLLOW")
    await expect_error("teach_mode needs mirroring driver", S.teach_mode(True), "not found")

    await motion_engine_tests()

    tools = sorted(t.name for t in await S.mcp.list_tools())
    print("tools:", tools)
    check("teach_mode tool registered", "teach_mode" in tools)
    check("motion tools registered", {"go_to", "play_motion", "stop_motion"} <= set(tools))
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)

def reset_follower():
    for m in MODEL["FOLLOW"].values():
        m.update(pos=2000, torque=1, goal=2000, cur=10, err=0)
    SIM.update(on=True, freeze=False, reads=0, spike=None, err=None, blocked=set(), grip_stop=None)

def held_at_present():
    """Last goal write to FOLLOW equals current positions, and torque is on (= hold)."""
    _, goals = last_goal_write()
    return all(MODEL["FOLLOW"][j]["goal"] == MODEL["FOLLOW"][j]["pos"] for j, _ in goals) and \
        all(m["torque"] == 1 for m in MODEL["FOLLOW"].values())

async def motion_engine_tests():
    import anyio, time as _t
    print("\n--- motion engine (simulated servos) ---")
    MIRROR.update(present=False); MIRROR["links"].clear(); MIRROR["modes"].clear()

    reset_follower()
    r = await S.go_to({"shoulder": 0.7})
    check("go_to completes and settles", r["completed"] and r["settled"], str(r))
    check("go_to reaches target", abs(r["final"]["shoulder"] - 0.7) <= 0.02, r["final"])
    check("go_to streams many small goals", r["ticks"] >= 5, r["ticks"])
    check("go_to timing matches plan", abs(r["elapsed_s"] - r["planned_s"]) < 0.6, (r["planned_s"], r["elapsed_s"]))

    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800); SIM["grip_stop"] = 2000
    r = await S.play_motion([{"joints": {"elbow": 0.6}, "duration_s": 0.3},
                             {"joints": {"gripper": 0.1}, "duration_s": 0.5}])
    check("grasp detected, motion still completes", r["completed"] and r["grasp_detected"], str(r))
    check("gripper stops squeezing at the object", MODEL["FOLLOW"][8]["goal"] >= 1990, MODEL["FOLLOW"][8])

    reset_follower(); SIM["spike"] = {"motor": 2, "after": 10, "value": 500}
    r = await S.play_motion([{"joints": {"shoulder": 0.9}, "duration_s": 1.0}])
    check("collision (over-current) aborts", not r["completed"] and "current 500" in r["abort_reason"], str(r))
    check("after abort the arm holds where it is", held_at_present())

    reset_follower(); SIM["err"] = {"motor": 4, "after": 10, "value": 0x20}
    r = await S.play_motion([{"joints": {"elbow": 0.9}, "duration_s": 1.0}])
    check("servo overload bit aborts", not r["completed"] and "0x20" in r["abort_reason"], str(r))

    reset_follower(); SIM["err"] = {"motor": 4, "after": 3, "value": 0x01}
    r = await S.go_to({"elbow": 0.6})
    check("voltage bit alone does not abort", r["completed"], str(r))

    reset_follower(); SIM["blocked"] = {3}
    r = await S.play_motion([{"joints": {"upper_arm_roll": 0.9}, "duration_s": 0.5}])
    check("blocked joint aborts on tracking error", not r["completed"] and "away from its target" in r["abort_reason"], str(r))

    reset_follower(); SIM["freeze"] = True
    r = await S.play_motion([{"joints": {"shoulder": 0.8}, "duration_s": 1.5}])
    check("frozen robot data aborts (no moving blind)", not r["completed"] and "old" in r["abort_reason"], str(r))

    reset_follower()
    task = asyncio.create_task(S.play_motion([{"joints": {"shoulder": 0.9}, "duration_s": 1.5}]))
    await asyncio.sleep(0.2)
    await expect_error("second motion refused while one runs", S.go_to({"elbow": 0.6}), "another motion")
    await S.stop_motion()
    r = await task
    check("stop_motion stops a running motion", not r["completed"] and "stopped" in r["abort_reason"], str(r))
    check("stopped motion holds", held_at_present())

    reset_follower()
    t0 = _t.monotonic()
    async with anyio.create_task_group() as tg:
        scope = anyio.CancelScope()
        async def run():
            with scope:
                await S.play_motion([{"joints": {"shoulder": 0.9}, "duration_s": 1.5}])
        tg.start_soon(run)
        await anyio.sleep(0.2)
        scope.cancel()        # what the MCP library does when the client cancels the call
    check("cancelled call ends quickly", _t.monotonic() - t0 < 1.0)
    check("cancelled call still holds the arm (shielded)", held_at_present())
    r = await S.go_to({"shoulder": 0.5})
    check("engine usable again after cancel (lock released)", r["completed"], str(r))

    reset_follower()
    rec = [{"t": i * 0.05, "joints": {"shoulder": 0.5 + 0.01 * i}} for i in range(21)]
    r1 = await S.play_motion(rec)
    reset_follower()
    r2 = await S.play_motion(rec, speed_scale=2.0)
    check("dense recording plays with its timing", r1["completed"] and abs(r1["planned_s"] - 1.0) < 0.05, str(r1))
    check("speed_scale 2 halves the plan", abs(r2["planned_s"] - 0.5) < 0.05 and r2["completed"], str(r2))

    reset_follower(); MIRROR.update(present=True); MIRROR["links"]["FOLLOW"] = "LEAD"
    await expect_error("fast moves refused while teach mode is on", S.go_to({"shoulder": 0.6}), "teach_mode(on=False)")
    MIRROR["links"].clear(); MIRROR.update(present=False)
    SIM["on"] = False

asyncio.run(main())
