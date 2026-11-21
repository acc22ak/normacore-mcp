"""Offline test: runs every MCP tool against a simulated Station (no robot needed).

usage: python tests/offline_test.py <normacore-mcp dir> <norma-core repo>

The fake station reacts to mirroring commands the way motors-mirroring does in the
norma-core source: START -> link + torque on targets (inference/mod.rs:68-88),
STOP -> unlink + torque off on the freed bus (inference/mod.rs:91-131).
"""
import asyncio, io, json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["NORMA_CORE_REPO"] = sys.argv[2]
os.environ["FOLLOWER_BUS"] = "FOLLOW"
# fast settings so the motion-engine tests run in a few seconds
os.environ.update({"MOTION_RATE_HZ": "100", "MAX_VEL_NORM": "2.0", "SETTLE_S": "0.5", "TRACK_ERR_TICKS": "5"})
import tempfile
_TMP = tempfile.mkdtemp(prefix="normacore-test-")
os.environ["CAMERAS_FILE"] = os.path.join(_TMP, "cameras.json")  # never write into the repo's data/
os.environ["SKILLS_DIR"] = os.path.join(_TMP, "skills")
os.environ["RUNS_FILE"] = os.path.join(_TMP, "runs.jsonl")
os.environ["ATTEMPT_FILE"] = os.path.join(_TMP, "attempt.json")
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

async def fake_read_last(q, n=1, timeout=None, step=1):
    if q == "st3215/inference":
        if HISTORY and n > 1:                      # robot history for save_attempt_as_skill
            READS_N.append(n)
            return [memoryview(x) for x in HISTORY[-n:][::step]]  # Station: every step-th of the last n
        return [memoryview(enc_st3215())]
    if q == S.MIRRORING_QUEUE:
        if not MIRROR["present"]:
            raise S.QueueMissing(q)
        return [memoryview(enc_mirroring())]
    if q not in STATIC:                            # e.g. estop/state before anyone pressed the e-stop
        raise S.QueueMissing(q)
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
    check("look bad index gives message, not IndexError", len(bad) == 1 and "only 1 connected" in bad[0], str(bad))
    check("look unknown name gives message", "No camera matching" in (await S.look("nope"))[0])

    # --- frame freshness: camera queues persist, so the "latest" frame can be old ---
    saved = dict(STATIC)
    STATIC["usbvideo/abc"] = cam_frame(0.2)
    out = await S.look()
    check("fresh frame: age shown, not stale", "0.2 s old" in out[0] and "STALE" not in out[0], out[0])
    STATIC["usbvideo/abc"] = cam_frame(30)
    out = await S.look()
    check("old frame flagged STALE with age", "STALE" in out[0] and "30.0 s old" in out[0] and "NOT the current scene" in out[0], out[0])
    check("stale frame still returned for diagnosis", hasattr(out[1], "data"))
    # past CAMERA_GONE_S (60 s) the camera counts as unplugged: hidden, but reachable by id
    STATIC["usbvideo/abc"] = cam_frame(3 * 3600)
    out = await S.look()
    check("long-dead camera hidden from look('all')", len(out) == 1 and "No cameras connected (1 unplugged" in out[0], out[0])
    out = await S.look("abc")
    check("unplugged camera still viewable by id, flagged STALE", out[0].startswith("unplugged camera:") and "3.0 h old" in out[0] and hasattr(out[1], "data"), out[0])
    STATIC["usbvideo/abc"] = [cam_frame(0.5), cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_DISCONNECTED)]
    check("disconnected camera hidden", (await S.list_cameras())["cameras"] == [])
    out = await S.look("abc")
    check("disconnect after last frame flagged STALE", "STALE: camera reported disconnected" in out[0], out[0])
    # what the real Station writes last on unplug ("Capture session ... ended")
    STATIC["usbvideo/abc"] = [cam_frame(0.5), cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_RECORDING_END)]
    lc = await S.list_cameras(include_gone=True)
    check("capture-end counts as unplugged", lc["cameras"] == [] and lc["unplugged_hidden"] == 1 and lc["unplugged"][0]["queue"] == "usbvideo/abc", str(lc))
    STATIC["usbvideo/abc"] = cam_event(usbvideo.RxEnvelopeType.ET_DEVICE_CONNECTED)
    lc = await S.list_cameras()
    check("just-connected camera without frames is listed", [c["frame"] for c in lc["cameras"]] == ["no frames yet"], str(lc))
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

    # --- roles follow the physical camera across replugs (real Linux queue ids) ---
    # On Linux Station's unique_id ends in the USB device number, which changes per replug,
    # so a replugged camera shows up under a new "/<station>/usbvideo/<md5>" queue.
    def usb_cam(age_s, product, vid, pid, serial):
        import time
        stamp = frame.FrameStamp(local_stamp_ns=time.time_ns() - int(age_s * 1e9))
        cam = usbvideo.Camera(product=product, vendor_id=vid, product_id=pid, serial_number=serial)
        return usbvideo.RxEnvelope(type=usbvideo.RxEnvelopeType.ET_FRAMES, camera=cam, stamp=stamp,
                                   frames=frame.FramesPack(frames_data=[JPEG], stamps=[stamp])).encode()
    def index(*qs):
        STATIC["inference-states"] = inference.InferenceRx(entries=[
            inference.InferenceRx_Entry(queue=q, type=drivers.QueueDataType.QDT_USB_VIDEO_FRAMES) for q in qs]).encode()
    DESK1, DESK2, GRIP = "/st1/usbvideo/17f9aa", "/st1/usbvideo/e606bb", "/st1/usbvideo/43c6cc"
    STATIC[DESK1] = usb_cam(0.1, "Web Camera", 0x32e6, 0x9228, "202604081837")
    STATIC[GRIP] = usb_cam(0.1, "FHD C3 Camera", 0x32e6, 0x9251, "2025090142113")
    index(DESK1, GRIP)
    r = await S.set_camera_role("desk", "17f9")
    check("hash prefix matches station-prefixed queue ids", r["camera"] == DESK1, str(r))
    check("role saved by stable camera_id", r["saved_roles"] == {"desk": "32e6:9228:202604081837"}, str(r))
    await S.set_camera_role("gripper", "32e6:9251:2025090142113")
    # unplug the desk cam: its queue ends with capture-end
    STATIC[DESK1] = [usb_cam(0.5, "Web Camera", 0x32e6, 0x9228, "202604081837"),
                     usbvideo.RxEnvelope(type=usbvideo.RxEnvelopeType.ET_DEVICE_RECORDING_END,
                                         camera=usbvideo.Camera(product="Web Camera", vendor_id=0x32e6, product_id=0x9228,
                                                                serial_number="202604081837")).encode()]
    out = await S.look("desk")
    check("unplugged role gives a clear message", len(out) == 1 and "is unplugged" in out[0] and "Web Camera" in out[0], out[0])
    # replug: same camera, new queue
    STATIC[DESK2] = usb_cam(0.1, "Web Camera", 0x32e6, 0x9228, "202604081837")
    index(DESK1, DESK2, GRIP)
    out = await S.look("desk")
    check("role follows the camera to its new queue", len(out) == 2 and "[desk]" in out[0] and DESK2 in out[0], out[0])
    lc = await S.list_cameras()
    check("old queue hidden, roles intact", [(c["queue"], c["role"]) for c in lc["cameras"]] == [(GRIP, "gripper"), (DESK2, "desk")]
          and lc["role_problems"] == [] and lc["unplugged_hidden"] == 1, str(lc))
    out = await S.look()
    check("look('all') shows only the 2 connected cameras", len(out) == 4, str(out[::2]))
    # legacy roles saved as queue ids still work
    await S.set_camera_role("desk", "none"); await S.set_camera_role("gripper", "none")
    os.environ["CAMERAS"] = f"top={GRIP}"
    check("legacy queue-id role still resolves", (await S.list_cameras())["cameras"][0]["role"] == "top")
    del os.environ["CAMERAS"]
    for q in (DESK1, DESK2, GRIP):
        STATIC.pop(q)

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

    # --- mirroring guard: someone switched Station teleop on in the web UI ---
    MIRROR["links"]["FOLLOW"] = "LEAD"
    MIRROR["modes"].update({"LEAD": mirroring.BusMode.BR_LEADER, "FOLLOW": mirroring.BusMode.BR_FOLLOWER})
    await expect_error("move refused while mirrored", S.move_joints({"1": 0.5}), "turn mirroring off")
    await expect_error("nudge refused while mirrored", S.nudge_joints({"elbow": 0.05}), "turn mirroring off")
    await expect_error("hold refused while mirrored", S.hold(), "turn mirroring off")
    check("relax still allowed while mirrored", "torque disabled" in await S.relax())
    for mm in MODEL["FOLLOW"].values(): mm["torque"] = 1
    MIRROR["links"].clear()
    r = await S.move_joints({"1": 0.5})
    check("move allowed once mirroring is off", r["bus"] == "FOLLOW")

    # --- mirroring driver absent: motion allowed ---
    MIRROR["present"] = False; MIRROR["modes"].clear()
    r = await S.move_joints({"1": 0.5})
    check("move allowed when mirroring queue missing", r["bus"] == "FOLLOW")

    await motion_engine_tests()
    await learning_tests()
    await self_learning_tests()

    tools = sorted(t.name for t in await S.mcp.list_tools())
    print("tools:", tools)
    check("teach mode + recording tools removed (17 tools)",
          not ({"teach_mode", "skill_from_history"} & set(tools)) and len(tools) == 17, len(tools))
    check("motion tools registered", {"go_to", "play_motion", "stop_motion"} <= set(tools))
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)

HISTORY: list = []     # encoded st3215/inference entries, oldest first
READS_N: list = []     # how many entries each history read asked for

def history_entry(stamp_ns, follow_raw: dict):
    motors = [st3215.InferenceState_MotorState(id=i, state=motor_state(follow_raw.get(i, 2000), 1),
                                               range_min=1000, range_max=3000, range_freezed=True) for i in range(1, 9)]
    lead = [st3215.InferenceState_MotorState(id=i, state=motor_state(2000, 0), range_min=1000, range_max=3000,
                                             range_freezed=True) for i in range(1, 9)]
    return st3215.InferenceState(buses=[
        st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number="FOLLOW"), motors=motors, monotonic_stamp_ns=stamp_ns),
        st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number="LEAD"), motors=lead, monotonic_stamp_ns=stamp_ns),
    ]).encode()

def make_demo(hz=25, idle_before=10.0, move=3.0, idle_after=5.0):
    """Follower history: still, then shoulder 2000->2400 and gripper closes 2800->2200, then still."""
    out, n = [], int((idle_before + move + idle_after) * hz)
    for i in range(n):
        t = i / hz
        u = min(max((t - idle_before) / move, 0.0), 1.0) if move > 0 else 0.0
        out.append(history_entry(10**12 + int(t * 1e9), {2: 2000 + int(400 * u), 8: 2800 - int(600 * u)}))
    return out

async def learn_from(history, name, since_ns, overwrite=False):
    """Create a skill the self-learning way: start_attempt at station time since_ns, then the
    follower history of the attempt, then save_attempt_as_skill."""
    SIM.update(on=False, stamp=since_ns)
    await S.start_attempt(f"test task {name}")
    HISTORY[:] = history
    try:
        return await S.save_attempt_as_skill(name, overwrite=overwrite)
    finally:
        HISTORY[:] = []

async def learning_tests():
    print("\n--- skills -> practice ladder ---")
    MIRROR.update(present=False); MIRROR["links"].clear(); MIRROR["modes"].clear()
    r = await learn_from(make_demo(), "pick_a", 10**12)
    check("skill learned from the attempt's history", r["kept_poses"] >= 2 and r["warning"] is None, str(r))
    check("history rate measured (~25 Hz), not assumed", abs(r["history"]["rate_hz"] - 25) < 1, r["history"])
    check("asked for >255 entries (multi-byte offset)", max(READS_N) > 255, READS_N)
    sk = json.load(open(os.path.join(os.environ["SKILLS_DIR"], "pick_a.json")))
    check("skill file has keyframes with raw, calibration, practice 0.5",
          sk["keyframes"] and "raw" in sk["keyframes"][0] and sk["calibration"]["ranges"]["2"] == [1000, 3000]
          and sk["practice"]["speed_scale"] == 0.5)
    SIM.update(on=False, stamp=10**12); await S.start_attempt("again")
    HISTORY[:] = make_demo()
    await expect_error("refuses to overwrite without overwrite=True", S.save_attempt_as_skill("pick_a"), "exists")
    await expect_error("bad skill name rejected", S.save_attempt_as_skill("Pick A!"), "skill name")
    HISTORY[:] = []

    ls = await S.list_skills()
    check("list_skills shows it", ls and ls[0]["name"] == "pick_a" and ls[0]["speed_scale"] == 0.5, str(ls))

    reset_follower()
    for m in MODEL["FOLLOW"].values():
        m.update(pos=2000, goal=2000)
    MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_a")
    check("run_skill plays the recording fast in one call", r["completed"] and r["speed_scale"] == 0.5, str(r))
    check("run_skill reached the demo's end pose", abs(MODEL["FOLLOW"][2]["pos"] - 2400) <= 25, MODEL["FOLLOW"][2])
    res = await S.report_skill_result(r["run_id"], True)
    check("1 success: still 0.5", res["practice"]["speed_scale"] == 0.5 and res["status"] == "ok", str(res))
    await expect_error("run_id can't be reported twice", S.report_skill_result(r["run_id"], True), "unknown")
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_a")
    res = await S.report_skill_result(r["run_id"], True)
    check("2 successes in a row: speed up to 0.75", res["practice"]["speed_scale"] == 0.75, str(res))
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_a")
    check("next run uses 0.75 and is quicker", r["speed_scale"] == 0.75 and r["completed"], str(r))
    res = await S.report_skill_result(r["run_id"], False, "missed the cube")
    check("failure: back to 0.5, status flaky", res["practice"]["speed_scale"] == 0.5 and res["status"] == "flaky", str(res))

    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    SIM["spike"] = {"motor": 3, "after": 5, "value": 500}
    r = await S.run_skill("pick_a")
    check("engine abort counts as failure automatically", not r["completed"] and r["counted_as"] == "failure", str(r))
    SIM["spike"] = None

    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 1100                                 # re-calibrated since the demo
    r = await S.run_skill("pick_a", speed_scale=1.5)
    check("calibration changed: replay from raw, capped at 0.5", r["speed_scale"] == 0.5 and r["calibration_changed"], str(r))
    check("calibration changed: still reaches the same RAW end pose", abs(MODEL["FOLLOW"][2]["pos"] - 2400) <= 25, MODEL["FOLLOW"][2])
    sk = json.load(open(os.path.join(os.environ["SKILLS_DIR"], "pick_a.json")))
    check("calibration changed: status needs_check", sk["status"] == "needs_check", sk["status"])
    res = await S.report_skill_result(r["run_id"], True)
    sk = json.load(open(os.path.join(os.environ["SKILLS_DIR"], "pick_a.json")))
    check("success after re-calibration stores the re-based skill", res["rebased_to_new_calibration"]
          and sk["calibration"]["ranges"]["2"] == [1100, 3000] and sk["last_good"]["calibration"]["ranges"]["2"] == [1000, 3000]
          and sk["version"] == 2 and sk["status"] == "ok", str(res))
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_a")
    check("re-based skill no longer needs the raw rebuild or the 0.5 cap", r["completed"] and not r["calibration_changed"], str(r))
    check("re-based skill still reaches the same RAW end pose", abs(MODEL["FOLLOW"][2]["pos"] - 2400) <= 25, MODEL["FOLLOW"][2])
    await S.report_skill_result(r["run_id"], True)
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 2300                                 # new range no longer contains the recorded raw positions
    await expect_error("raw positions outside today's range refuse the run (no clamping)", S.run_skill("pick_a"), "outside today's range")
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 1000

    runs = [json.loads(l) for l in open(os.environ["RUNS_FILE"], encoding="utf-8")]
    check("every run logged in runs.jsonl", len(runs) == 6 and all("practice_after" in x for x in runs), len(runs))

    # --- speed override can't skip the practice ladder ---
    await learn_from(make_demo(), "pick_b", 10**12)
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_b", speed_scale=2.0)
    check("new skill: speed_scale override capped to practice speed", r["speed_scale"] == 0.5 and "capped" in (r["speed_note"] or ""), str(r))
    await S.report_skill_result(r["run_id"], True)
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("pick_b", speed_scale=1.5)
    check("after a watched success the override is allowed", r["speed_scale"] == 1.5 and r["speed_note"] is None, str(r))
    await S.report_skill_result(r["run_id"], True)
    await expect_error("speed_scale 0 rejected", S.run_skill("pick_b", speed_scale=0), "> 0")

    # --- long, fast history (Station writes ~400 entries/s): strided read covers the whole attempt ---
    long_demo = make_demo(hz=200, idle_before=40.0, move=3.0, idle_after=5.0)
    r = await learn_from(long_demo, "long_a", 10**12)
    check("fast history read with a stride", r["history"]["read_step"] > 1 and r["attempt_s"] >= 47.5
          and "note" not in r["history"], r["history"])
    cap = S.HISTORY_MAX_ENTRIES
    S.HISTORY_MAX_ENTRIES = 300
    r = await learn_from(long_demo, "long_b", 10**12)
    check("entry cap raises the stride instead of cutting the attempt short",
          r["history"]["entries_read"] <= 300 and r["attempt_s"] >= 47.5, r["history"])
    S.HISTORY_MAX_ENTRIES = cap
    r = await learn_from(long_demo, "long_c", 10**12 - 60 * 10**9)
    check("attempt older than Station's history: warns with real coverage",
          "starts 60.0 s after start_attempt" in r["history"].get("note", ""), r["history"])

    # --- hand-written keyframe skills in the documented format ---
    hand = {"name": "wave", "bus": "FOLLOW", "status": "ok",
            "calibration": {"ranges": {str(j): [1000, 3000] for j in range(1, 9)}},
            "keyframes": [{"joints": {"2": 0.6}, "duration_s": 0.3, "label": "up"},
                          {"gripper": "close", "label": "grasp"},
                          {"joints": {"2": 0.5}, "duration_s": 0.3}]}
    with open(os.path.join(os.environ["SKILLS_DIR"], "wave.json"), "w") as f:
        json.dump(hand, f)
    await expect_error("gripper open/close needs GRIPPER_OPEN_NORM", S.run_skill("wave"), "GRIPPER_OPEN_NORM")
    S.GRIPPER_OPEN_NORM = "1"
    reset_follower()
    r = await S.run_skill("wave")
    check("{'gripper': 'close'} keyframe runs (close = 0.0 when open = 1)", r["completed"] and MODEL["FOLLOW"][8]["goal"] == 1000, str(r))
    await S.report_skill_result(r["run_id"], True)
    S.GRIPPER_OPEN_NORM = ""
    hand["keyframes"].append({"skill": "pick_a"})
    with open(os.path.join(os.environ["SKILLS_DIR"], "wave.json"), "w") as f:
        json.dump(hand, f)
    await expect_error("composite step gives a clear error", S.run_skill("wave"), "composite")
    SIM["on"] = False

def claude_attempt_history(t0_ns, hz=25):
    """Follower history of a Claude-style attempt: 5 s of unrelated movement BEFORE the attempt,
    then reach in 6 small steps (2 s thinking each), close gripper in 2 steps, lift in 3 steps."""
    out = []
    t = [-5.0]
    def emit(p):
        out.append(history_entry(t0_ns + int(t[0] * 1e9), dict(p))); t[0] += 1 / hz
    while t[0] < 0:                                     # unrelated wiggle before start_attempt
        emit({2: 2000 + int(300 * abs(((t[0] * 2) % 2) - 1)), 8: 2800})
    pose = {2: 2000, 8: 2800}
    def hold(sec):
        for _ in range(int(sec * hz)): emit(pose)
    def move(j, d, steps=6):
        for _ in range(steps):
            pose[j] += d // steps; emit(pose)
    hold(2)
    for _ in range(6): move(2, 60); hold(2)
    for _ in range(2): move(8, -300); hold(2)
    for _ in range(3): move(2, -60); hold(2)
    return out

async def self_learning_tests():
    print("\n--- self-learning: Claude's own attempt -> fast skill ---")
    MIRROR.update(present=False); MIRROR["links"].clear(); MIRROR["modes"].clear()
    await expect_error("save without start_attempt is refused", S.save_attempt_as_skill("x"), "start_attempt")
    t0 = 2 * 10**12
    SIM.update(on=False, stamp=t0)                      # "now" for start_attempt = t0
    r = await S.start_attempt("move cube from A to B")
    check("start_attempt marks the station time", r["since_ns"] == t0, str(r))
    HISTORY[:] = claude_attempt_history(t0)
    r = await S.save_attempt_as_skill("a_to_b")
    check("attempt saved as a skill", r["kept_poses"] == 4, str(r))
    check("movement before start_attempt ignored", r["attempt_s"] < 30, r["attempt_s"])
    check("minutes of thinking -> seconds of skill", r["attempt_s"] > 25 and r["skill_s"]["1.0"] < 8 and r["skill_s"]["2.0"] < 4, str(r))
    sk = json.load(open(os.path.join(os.environ["SKILLS_DIR"], "a_to_b.json")))
    check("self skill: source=self, keyframes with raw, start pose, ladder at 0.5",
          sk["source"]["type"] == "self" and all("raw" in k for k in sk["keyframes"])
          and sk["start_raw"]["2"] == 2000 and sk["practice"]["speed_scale"] == 0.5)
    await expect_error("attempt is consumed after saving", S.save_attempt_as_skill("a_to_b2"), "start_attempt")
    SIM.update(on=False, stamp=3 * 10**12); await S.start_attempt("nothing")
    HISTORY[:] = [history_entry(3 * 10**12 + i * 40_000_000, {2: 2000, 8: 2800}) for i in range(100)]
    await expect_error("attempt without movement is refused", S.save_attempt_as_skill("still"), "didn't move")
    HISTORY[:] = []

    reset_follower()
    MODEL["FOLLOW"][2].update(pos=2300, goal=2300)      # arm NOT at the learned start pose
    MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    r = await S.run_skill("a_to_b")
    check("self skill runs fast in one call", r["completed"] and r["speed_scale"] == 0.5, str(r))
    check("it first glides to the learned start, then ends at the learned end pose",
          abs(MODEL["FOLLOW"][2]["pos"] - 2180) <= 25 and abs(MODEL["FOLLOW"][8]["pos"] - 2200) <= 25,
          (MODEL["FOLLOW"][2]["pos"], MODEL["FOLLOW"][8]["pos"]))
    res = await S.report_skill_result(r["run_id"], True)
    check("practice ladder applies to self skills", res["status"] == "ok" and res["practice"]["streak"] == 1, str(res))
    ls = {s["name"]: s for s in await S.list_skills()}
    check("list_skills marks it self-learned", ls["a_to_b"]["source"] == "self", str(ls["a_to_b"]))

    # re-calibration after learning: start pose + keyframes replay from raw, then get re-based on success
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 1100
    r = await S.run_skill("a_to_b")
    check("self skill after re-calibration: replays from raw, ends at same raw pose",
          r["completed"] and r["calibration_changed"] and abs(MODEL["FOLLOW"][2]["pos"] - 2180) <= 25, str(r))
    res = await S.report_skill_result(r["run_id"], True)
    sk = json.load(open(os.path.join(os.environ["SKILLS_DIR"], "a_to_b.json")))
    check("success re-bases start pose + keyframes to the new calibration",
          res["rebased_to_new_calibration"] and sk["calibration"]["ranges"]["2"][0] == 1100
          and abs(float(sk["start_pose"]["2"]) - (2000 - 1100) / 1900) < 0.01
          and sk["last_good"]["start_pose"] is not None, (sk["start_pose"].get("2"), sk["calibration"]["ranges"]["2"]))
    reset_follower(); MODEL["FOLLOW"][8].update(pos=2800, goal=2800)
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 1100
    r = await S.run_skill("a_to_b")
    check("next run: no rebuild needed, still reaches the learned pose",
          r["completed"] and not r["calibration_changed"] and abs(MODEL["FOLLOW"][2]["pos"] - 2180) <= 25, str(r))
    await S.report_skill_result(r["run_id"], True)
    for m in MODEL["FOLLOW"].values():
        m["rmin"] = 1000
    SIM["on"] = False


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
    await expect_error("fast moves refused while mirroring is on", S.go_to({"shoulder": 0.6}), "turn mirroring off")
    MIRROR["links"].clear(); MIRROR.update(present=False)
    SIM["on"] = False

asyncio.run(main())
