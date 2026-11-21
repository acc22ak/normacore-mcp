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

def motor_state(pos, torque):
    b = bytearray(0x50)
    b[0x38:0x3A] = pos.to_bytes(2, "little"); b[0x28] = torque
    return bytes(b)

def enc_st3215():
    buses = []
    for serial, motors in MODEL.items():
        ms = [st3215.InferenceState_MotorState(id=i, state=motor_state(m["pos"], m["torque"]),
                                               range_min=m["rmin"], range_max=m["rmax"], range_freezed=m["frozen"]) for i, m in motors.items()]
        buses.append(st3215.InferenceState_BusState(bus=st3215.ST3215Bus(serial_number=serial), motors=ms))
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
    return [memoryview(STATIC[q])]

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
    check("cameras discovered", await S.list_cameras() == ["usbvideo/abc"])
    out = await S.look()
    check("look returns label + image", isinstance(out[0], str) and hasattr(out[1], "data"))
    bad = await S.look("5")
    check("look bad index gives message, not IndexError", len(bad) == 1 and "only 1 found" in bad[0], str(bad))
    check("look unknown name gives message", "No camera matching" in (await S.look("nope"))[0])
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

    tools = sorted(t.name for t in await S.mcp.list_tools())
    print("tools:", tools)
    check("teach_mode tool registered", "teach_mode" in tools)
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)

asyncio.run(main())
