"""Tests for the e-stop (estop.py) and the latch checks in server.py / autopilot.py.

usage: python tests/test_estop.py <normacore-mcp dir> <norma-core repo> [station host:port]

Self-contained simulated station: a FOLLOW bus whose torqued servos glide toward their goal, and
a limp LEAD bus (torque off, like a real leader arm). The e-stop's own Station wrapper is swapped
for one that talks to the same simulation, so the real command encoding is exercised end to end.

With the optional 3rd argument it also checks the latch round trip against a REAL Station
(writes to its estop/state queue, then releases). Use a scratch/test Station or make sure
nobody is relying on a latch there.
"""
import asyncio, os, sys, tempfile

sys.path.insert(0, sys.argv[1])
os.environ["NORMA_CORE_REPO"] = sys.argv[2]
os.environ["FOLLOWER_BUS"] = "FOLLOW"
os.environ.update({"MOTION_RATE_HZ": "100", "MAX_VEL_NORM": "0.5", "SETTLE_S": "0.3",
                   "STALE_STATE_S": "0.5", "PROBE_VEL_NORM": "0.5"})
_TMP = tempfile.mkdtemp(prefix="normacore-estop-")
os.environ["CONTROL_MODE_FILE"] = os.path.join(_TMP, "control_mode.json")
os.environ["LIMITS_FILE"] = os.path.join(_TMP, "limits.json")
os.environ["SKILLS_DIR"] = os.path.join(_TMP, "skills")
os.environ["RUNS_FILE"] = os.path.join(_TMP, "runs.jsonl")

import server as S
import autopilot as A
import estop as E
from target.gen_python.protobuf.drivers.st3215 import st3215
from target.gen_python.protobuf.station import commands, drivers

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ---- tiny simulated station ------------------------------------------------
STEP = 15                      # raw steps per read: slow enough that motions last ~1 s
MODEL, LATCH = {}, []          # LATCH = entries of the estop/state queue (oldest first)
SIM = {"stamp": 1}
# Station's motors-mirroring, simulated: while FOLLOW is linked, every read copies the leader's
# pose into the follower's goal; a STOP unlinks it and turns its torque off TORQUE_OFF_DELAY_S
# later (async, like inference/mod.rs:91-131).
MIRROR = {"links": {}, "pending_off": []}   # links: follower -> leader; pending_off: (time, bus)
TORQUE_OFF_DELAY_S = 0.15

def reset():
    MODEL.clear(); LATCH.clear(); MIRROR["links"].clear(); MIRROR["pending_off"].clear()
    MODEL["FOLLOW"] = {i: {"pos": 2000, "goal": 2000, "torque": 1} for i in range(1, 9)}
    MODEL["LEAD"] = {i: {"pos": 2100, "goal": 2100, "torque": 0} for i in range(1, 9)}

def sim_mirroring():
    import time as _t
    now = _t.monotonic()
    for t, bus in list(MIRROR["pending_off"]):
        if now >= t:
            for m in MODEL[bus].values(): m["torque"] = 0
            MIRROR["pending_off"].remove((t, bus))
    for f, l in MIRROR["links"].items():
        for i, m in MODEL[f].items():
            m["goal"] = MODEL[l][i]["pos"]

def enc_mirroring():
    from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
    def mb(s): return mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=s)
    by_leader = {}
    for f, l in MIRROR["links"].items():
        by_leader.setdefault(l, []).append(f)
    st = mirroring.InferenceState(mirroring=[mirroring.InferenceState_Mirroring(
        source=mirroring.InferenceState_Bus(id=mb(l), mode=mirroring.BusMode.BR_LEADER),
        targets=[mirroring.InferenceState_Bus(id=mb(f), mode=mirroring.BusMode.BR_FOLLOWER) for f in fs])
        for l, fs in by_leader.items()])
    return mirroring.RxEnvelope(state=st).encode()

def motor_state(pos, torque):
    b = bytearray(0x50)
    b[0x38:0x3A] = pos.to_bytes(2, "little"); b[0x28] = torque; b[0x45:0x47] = (10).to_bytes(2, "little")
    return bytes(b)

def enc_st3215():
    SIM["stamp"] += 1
    sim_mirroring()
    for m in MODEL["FOLLOW"].values():
        if m["torque"]:
            m["pos"] += max(-STEP, min(STEP, m["goal"] - m["pos"]))
    buses = [st3215.InferenceState_BusState(
        bus=st3215.ST3215Bus(serial_number=serial), monotonic_stamp_ns=SIM["stamp"],
        motors=[st3215.InferenceState_MotorState(id=i, state=motor_state(m["pos"], m["torque"]),
                                                 range_min=1000, range_max=3000, range_freezed=True)
                for i, m in motors.items()])
        for serial, motors in MODEL.items()]
    return st3215.InferenceState(buses=buses).encode()

async def fake_read_last(q, n=1, timeout=None, step=1):
    if q == "st3215/inference":
        return [memoryview(enc_st3215())]
    if q == E.QUEUE and LATCH:
        return [memoryview(LATCH[-1])]
    if q == S.MIRRORING_QUEUE and MIRROR["links"]:
        return [memoryview(enc_mirroring())]
    raise S.QueueMissing(q)          # mirroring off/absent, estop queue not created yet

SENT, MIRROR_CMDS = [], []
async def fake_send(_c, cmds):
    import time as _t
    from target.gen_python.protobuf.drivers.motors_mirroring import mirroring
    for c in cmds:
        r = commands.DriverCommandReader(memoryview(c.encode()))
        if r.get_type() == drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND:
            m = mirroring.CommandReader(r.get_body())
            src = m.get_source().get_unique_id()
            MIRROR_CMDS.append((m.get_type(), src, [t.get_unique_id() for t in m.get_targets() or []]))
            if m.get_type() == mirroring.CommandType.CT_STOP_MIRROR:   # source may be follower or leader
                for f in [f for f, l in MIRROR["links"].items() if src in (f, l)]:
                    del MIRROR["links"][f]
                    MIRROR["pending_off"].append((_t.monotonic() + TORQUE_OFF_DELAY_S, f))
            continue
        if r.get_type() != drivers.StationCommandType.STC_ST3215_COMMAND:
            continue
        cmd = st3215.CommandReader(r.get_body()); sw = cmd.get_sync_write()
        bus = cmd.get_target_bus_serial()
        SENT.append((bus, sw.get_address()))
        for mw in sw.get_motors():
            v = bytes(mw.get_value())
            if sw.get_address() == S.RAM_TORQUE_ENABLE:
                MODEL[bus][mw.get_motor_id()]["torque"] = v[0]
            elif sw.get_address() == S.RAM_GOAL_POSITION:
                MODEL[bus][mw.get_motor_id()]["goal"] = int.from_bytes(v, "little")

async def fake_client(): return None
REAL_READ_LAST, REAL_CLIENT = S.read_last, S.client          # for the optional real-Station test
S.read_last, S.send_commands, S.client = fake_read_last, fake_send, fake_client


class FakeStation:
    """estop.Station stand-in backed by the same simulation (real encode/parse paths)."""
    host = "sim"
    fail_latch = False
    async def read_buses(self): return E.parse_buses(enc_st3215())
    async def send(self, cmds): await fake_send(None, cmds)
    async def read_mirrored_followers(self):
        return E.parse_mirrored_followers(enc_mirroring()) if MIRROR["links"] else []
    async def read_latch(self): return E.decode_latch(LATCH[-1]) if LATCH else None
    async def write_latch(self, data):
        if self.fail_latch:
            raise ConnectionError("station write failed")
        LATCH.append(data)


async def expect_error(name, coro, needle):
    try:
        await coro
        check(name, False, "no error raised")
    except RuntimeError as e:
        check(name, needle in str(e), str(e))

def torque(bus): return {i: m["torque"] for i, m in MODEL[bus].items()}


# ---- pure protocol ---------------------------------------------------------
def protocol_tests():
    print("--- latch protocol ---")
    d = E.decode_latch(E.encode_latch(True, "relax", by="pit"))
    check("latch round trip", d and d["mode"] == "relax" and d["by"] == "pit", d)
    check("released entry -> not latched", E.decode_latch(E.encode_latch(False)) is None)
    g = E.decode_latch(b"\xff garbage")
    check("garbage fails closed as HOLD", g and g["mode"] == "hold", g)
    check("non-object JSON fails closed", E.decode_latch(b"[1,2]") is not None)
    check("unknown mode treated as hold", E.decode_latch(b'{"latched":true,"mode":"boom"}')["mode"] == "hold")
    try:
        E.encode_latch(True, "explode"); check("bad mode rejected", False)
    except ValueError:
        check("bad mode rejected", True)
    check("refusal tells the human how to release", "release" in E.refusal({"mode": "hold", "by": "x"}))


# ---- estop.py against the simulation ----------------------------------------
async def estop_tests():
    print("--- estop.py ---")
    st = FakeStation()

    reset(); SENT.clear()
    r = await E.trigger(st, "hold", repeats=2, interval=0.01)
    check("HOLD targets only powered buses (leader skipped)", r["buses"] == ["FOLLOW"], r)
    check("HOLD never stiffens the limp leader", all(v == 0 for v in torque("LEAD").values()), torque("LEAD"))
    check("HOLD sets goal = present position", all(m["goal"] == m["pos"] for m in MODEL["FOLLOW"].values()))
    check("HOLD writes goal before torque", SENT[:2] == [("FOLLOW", 0x2A), ("FOLLOW", 0x28)], SENT[:2])
    check("HOLD repeated", sum(1 for s in SENT if s == ("FOLLOW", 0x28)) == 2, SENT)
    check("HOLD latched", r["latched"] and (await st.read_latch())["mode"] == "hold", r)

    reset()
    r = await E.trigger(st, "relax", repeats=1)
    check("RELAX torque off on every bus", all(m["torque"] == 0 for b in MODEL.values() for m in b.values()))
    check("RELAX latched", (await st.read_latch())["mode"] == "relax")
    await E.release(st)
    check("release clears the latch", await st.read_latch() is None)

    reset()
    r = await E.trigger(st, "hold", follower="FOLLOW", leader="", repeats=1)
    check("HOLD with FOLLOWER_BUS targets just that bus", r["buses"] == ["FOLLOW"], r)
    reset()
    r = await E.trigger(st, "hold", leader="FOLLOW", repeats=1)
    check("HOLD never targets LEADER_BUS", r["buses"] == [] and r["errors"], r)

    reset(); st.fail_latch = True
    r = await E.trigger(st, "relax", repeats=1)
    st.fail_latch = False
    check("latch failure is reported loudly", not r["latched"] and any("LATCH NOT SET" in e for e in r["errors"]), r)
    check("...but the servos were still stopped", all(m["torque"] == 0 for m in MODEL["FOLLOW"].values()))


# ---- server.py honours the latch ---------------------------------------------
async def server_tests():
    print("--- server.py ---")
    st = FakeStation()

    reset(); LATCH.append(E.encode_latch(True, "hold", by="tester"))
    await expect_error("move_joints refused while latched", S.move_joints({"shoulder": 0.6}), "E-STOP is latched")
    await expect_error("go_to refused while latched", S.go_to({"shoulder": 0.6}), "E-STOP is latched")
    await expect_error("hold refused while latched (no surprise torque)", S.hold(), "E-STOP is latched")
    r = await S.relax()
    check("relax still allowed while latched", "torque disabled" in r and all(v == 0 for v in torque("FOLLOW").values()), r)

    await E.release(st); reset()
    r = await S.go_to({"shoulder": 0.55})
    check("moves work again after release", r["completed"], r)

    # e-stop pressed WHILE a fast motion runs: HOLD -> motion aborts and the arm stays powered
    reset()
    task = asyncio.create_task(S.go_to({"shoulder": 0.9, "elbow": 0.1}))
    await asyncio.sleep(0.15)
    e = await E.trigger(st, "hold", repeats=2, interval=0.02)
    r = await task
    check("HOLD aborts a running go_to", not r["completed"] and "E-STOP HOLD" in (r["abort_reason"] or ""), r)
    check("arm still powered after HOLD", all(v == 1 for v in torque("FOLLOW").values()), torque("FOLLOW"))
    sh = MODEL["FOLLOW"][2]
    check("shoulder stopped short of the target", S.to_norm(sh["pos"], 1000, 3000) < 0.85, sh)
    pos_then = {i: m["pos"] for i, m in MODEL["FOLLOW"].items()}
    for _ in range(10):
        enc_st3215()
    check("arm does not creep on after HOLD", all(abs(MODEL["FOLLOW"][i]["pos"] - p) <= STEP for i, p in pos_then.items()),
          {i: (p, MODEL["FOLLOW"][i]["pos"]) for i, p in pos_then.items()})

    # RELAX during a motion: the server must NOT re-enable torque when it aborts
    await E.release(st); reset()
    task = asyncio.create_task(S.go_to({"shoulder": 0.9}))
    await asyncio.sleep(0.15)
    await E.trigger(st, "relax", repeats=2, interval=0.02)
    r = await task
    check("RELAX aborts a running go_to", not r["completed"] and "E-STOP RELAX" in (r["abort_reason"] or ""), r)
    check("arm stays limp after RELAX (server did not hold)", all(v == 0 for v in torque("FOLLOW").values()), torque("FOLLOW"))


# ---- autopilot.py honours the latch -----------------------------------------
async def autopilot_tests():
    print("--- autopilot.py ---")
    st = FakeStation()
    await E.release(st); reset()
    await A.set_control_mode("autopilot", confirm_thresholds_tuned=True)
    task = asyncio.create_task(A.move_until("elbow", 0.95))
    await asyncio.sleep(0.15)
    await E.trigger(st, "relax", repeats=2, interval=0.02)
    r = await task
    check("RELAX aborts move_until", "E-STOP RELAX" in str(r.get("abort_reason")), r)
    check("arm stays limp after RELAX in autopilot", all(v == 0 for v in torque("FOLLOW").values()), torque("FOLLOW"))
    await E.release(st); reset()
    LATCH.append(E.encode_latch(True, "hold"))
    await expect_error("move_until refused while latched", A.move_until("elbow", 0.9), "E-STOP is latched")


# ---- e-stop while Station mirroring (teleop / teach_mode) drives the follower ----
def start_mirroring(leader_pos=2600):
    """Follower linked to the leader; the human holds the leader at leader_pos."""
    MIRROR["links"]["FOLLOW"] = "LEAD"
    for m in MODEL["LEAD"].values(): m["pos"] = leader_pos
    for _ in range(5): enc_st3215()               # follower starts gliding toward the leader

def human_moves_leader(to, reads=40):
    for m in MODEL["LEAD"].values(): m["pos"] = to
    for _ in range(reads): enc_st3215()

async def mirroring_tests():
    print("--- e-stop vs Station mirroring ---")
    st = FakeStation()

    # proof of the old problem: HOLD without stopping mirroring gets overwritten
    await E.release(st); reset(); start_mirroring()
    real = FakeStation.read_mirrored_followers
    async def blind(self): return []
    FakeStation.read_mirrored_followers = blind
    try:
        await E.trigger(st, "hold", repeats=1)
    finally:
        FakeStation.read_mirrored_followers = real
    before = MODEL["FOLLOW"][2]["pos"]
    human_moves_leader(2900)
    check("(old behaviour) HOLD alone is overwritten by mirroring", MODEL["FOLLOW"][2]["pos"] > before + 100,
          (before, MODEL["FOLLOW"][2]["pos"]))

    # HOLD now stops mirroring and still ends up holding, after Station's async torque-off
    await E.release(st); reset(); start_mirroring(); MIRROR_CMDS.clear()
    r = await E.trigger(st, "hold", repeats=2, interval=0.02)
    check("HOLD stops Station mirroring", r["mirroring_stopped"] == ["FOLLOW"] and not MIRROR["links"], r)
    check("stop sent in the web-UI shape (source = follower, no targets)",
          [(c[1], c[2]) for c in MIRROR_CMDS] == [("FOLLOW", [])], MIRROR_CMDS)
    check("Station's delayed torque-off has landed", not MIRROR["pending_off"])
    check("follower held after the torque-off (torque on)", all(v == 1 for v in torque("FOLLOW").values()),
          torque("FOLLOW"))
    check("leader never stiffened", all(v == 0 for v in torque("LEAD").values()), torque("LEAD"))
    check("HOLD reported no errors", not r["errors"], r)
    held = {i: m["pos"] for i, m in MODEL["FOLLOW"].items()}
    human_moves_leader(2900)
    check("follower no longer copies the leader", all(abs(MODEL["FOLLOW"][i]["pos"] - p) <= STEP
                                                      for i, p in held.items()),
          {i: (p, MODEL["FOLLOW"][i]["pos"]) for i, p in held.items()})
    check("latched HOLD", (await st.read_latch())["mode"] == "hold")

    # RELAX also stops mirroring and stays limp
    await E.release(st); reset(); start_mirroring()
    r = await E.trigger(st, "relax", repeats=2, interval=0.02)
    await asyncio.sleep(TORQUE_OFF_DELAY_S + 0.05); enc_st3215()
    check("RELAX stops Station mirroring", r["mirroring_stopped"] == ["FOLLOW"] and not MIRROR["links"], r)
    check("RELAX leaves every arm limp", all(m["torque"] == 0 for b in MODEL.values() for m in b.values()))
    p = MODEL["FOLLOW"][2]["pos"]
    human_moves_leader(2900)
    check("limp follower doesn't follow the leader", MODEL["FOLLOW"][2]["pos"] == p, (p, MODEL["FOLLOW"][2]["pos"]))

    # no mirroring: nothing extra is sent
    await E.release(st); reset(); MIRROR_CMDS.clear()
    r = await E.trigger(st, "hold", repeats=1)
    check("no mirroring -> no stop command sent", not MIRROR_CMDS and r["mirroring_stopped"] == [], MIRROR_CMDS)

    # teach_mode refuses while latched (Askar's check on top of the latch)
    await E.release(st); reset()
    LATCH.append(E.encode_latch(True, "hold"))
    await expect_error("teach_mode refused while latched", S.teach_mode(True), "E-STOP is latched")
    await E.release(st)


# ---- optional: real Station round trip --------------------------------------
async def real_station_test(host):
    print(f"--- real Station at {host} ---")
    st = E.Station(host)
    await st.connect()
    await st.write_latch(E.encode_latch(True, "hold", note="test_estop"))
    l1 = await st.read_latch()
    check("real Station: latch written and read back", l1 and l1["mode"] == "hold" and l1["note"] == "test_estop", l1)
    await E.release(st, note="test_estop")
    check("real Station: release reads back as not latched", await st.read_latch() is None)

    # server.read_last must return the NEWEST entries (tail offset 0 = newest)
    async def real_client(): return st.client
    saved = S.read_last, S.client
    S.read_last, S.client = REAL_READ_LAST, real_client
    try:
        q = f"test/read_last_{os.getpid()}"
        for i in range(4):
            await st.client.enqueue(q, f"e{i}".encode())
        one = [bytes(x).decode() for x in await S.read_last(q, 1, timeout=2)]
        three = [bytes(x).decode() for x in await S.read_last(q, 3, timeout=2)]
        check("real Station: read_last(n=1) is the newest entry", one == ["e3"], one)
        check("real Station: read_last(n=3) is the last 3, oldest first", three == ["e1", "e2", "e3"], three)
        await st.write_latch(E.encode_latch(True, "relax", note="t2"))
        l2 = await S.estop_latch()
        check("real Station: server sees a fresh press immediately", l2 and l2["mode"] == "relax", l2)
        await E.release(st, note="test_estop")
        check("real Station: server sees the release immediately", await S.estop_latch() is None)
    finally:
        S.read_last, S.client = saved


async def main():
    protocol_tests()
    await estop_tests()
    await server_tests()
    await autopilot_tests()
    await mirroring_tests()
    if len(sys.argv) > 3:
        await real_station_test(sys.argv[3])
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)

asyncio.run(main())
