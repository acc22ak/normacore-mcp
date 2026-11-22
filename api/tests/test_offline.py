"""Offline tests: no arm, no camera, no API calls.  Run: .venv/bin/python tests/test_offline.py"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arm import Arm, load_yaml  # noqa: E402
from safety import check_waypoints, parse_step  # noqa: E402
from skills import Skills  # noqa: E402
from station_io import SimIO  # noqa: E402
import usage as U  # noqa: E402
U.METER = U.Meter(Path("/tmp") / f"normacore-usage-{time.time_ns()}.jsonl")   # tests don't count toward real totals
import llm as L  # noqa: E402
L.STATE = Path("/tmp") / f"normacore-model-{time.time_ns()}.json"   # nor change the panel's model choice

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# live follower calibration read on 2026-11-22
LIVE = {"base_yaw": (1025, 3071), "shoulder": (947, 3149), "upper_arm_roll": (881, 3215), "elbow": (887, 3209),
        "forearm_roll": (53, 4043), "wrist_pitch": (1007, 3089), "wrist_roll": (101, 3995), "gripper": (1267, 2829)}


def make_arm(extra_limits=True):
    arm = Arm({**load_yaml("arm.yaml"), "use_extra_limits": extra_limits})
    arm.update_ranges({n: {"range_min": a, "range_max": b} for n, (a, b) in LIVE.items()})
    return arm


def test_arm_math():
    arm = make_arm()
    check("centre step is 0 degrees", abs(arm.norm_to_deg("base_yaw", arm.deg_to_norm("base_yaw", 0.0))) < 1e-6)
    check("base_yaw span ~180 deg", abs(arm.deg_per_norm("base_yaw") - 179.8) < 0.2)
    for n in LIVE:
        lo, hi = arm.limits(n)
        check(f"{n} limits inside 0..1 and ordered", 0.0 <= lo <= hi <= 1.0, (lo, hi))
    lo, hi = arm.limits("shoulder")
    check("shoulder soft limit caps at 0.60 (desk contact ~0.9)", hi <= 0.60, hi)
    lo, hi = arm.limits("base_yaw")
    check("base_yaw soft limit keeps clear of the laptop (>=0.30)", lo >= 0.30, lo)
    # URDF is narrower than calibration on wrist_pitch's top end: -78.9..101.1 deg
    check("wrist_pitch URDF upper limit respected",
          arm.norm_to_deg("wrist_pitch", arm.limits("wrist_pitch")[1]) <= 101.2)


def test_limits_switch():
    arm = make_arm(extra_limits=False)
    check("limits off: every joint may use its whole calibrated range",
          all(arm.limits(n) == (0.0, 1.0) for n in LIVE))
    wps, notes = check_waypoints(arm, [{"t": 0.0, "joints": {"shoulder": 0.95, "base_yaw": 0.1}}])
    check("limits off: a recorded pose passes unchanged", wps[0]["joints"] == {"shoulder": 0.95, "base_yaw": 0.1}
          and not notes, (wps, notes))


def test_safety():
    arm = make_arm()
    s, note = parse_step('{"do":"jog","joint":"base_yaw","degrees":90}', arm, set())
    check("jog clamped to +30", s and s["degrees"] == 30 and "clamped" in note, (s, note))
    s, _ = parse_step('{"do":"jog","joint":"base_yaw","degrees":-500}', arm, set())
    check("jog clamped to -30", s and s["degrees"] == -30)
    s, note = parse_step({"do": "jog", "joint": "upper_arm_roll", "degrees": 5}, arm, set())
    check("roll joint not jog-able", s is None and "can't be jogged" in note)
    s, note = parse_step({"do": "point", "target": "B"}, arm, set())
    check("point refused until slot taught", s is None and "no taught pose" in note)
    s, _ = parse_step({"do": "point", "target": "B"}, arm, {"B"})
    check("point allowed once taught", s == {"do": "point", "target": "B"})
    s, note = parse_step({"do": "jog", "joint": "base_yaw", "degrees": True}, arm, set())
    check("bool degrees rejected", s is None)
    s, note = parse_step("{not json", arm, set())
    check("garbage rejected", s is None and "unreadable" in note)
    s, note = parse_step({"do": "teleport"}, arm, set())
    check("unknown step rejected", s is None)
    wps, notes = check_waypoints(arm, [{"joints": {"shoulder": 0.95, "base_yaw": 0.0, "bogus": 1}}])
    j = wps[0]["joints"]
    check("waypoints clamped into limits", j["shoulder"] <= 0.60 and j["base_yaw"] >= 0.30 and "bogus" not in j, j)
    check("clamp notes reported", len(notes) == 3, notes)


def test_skill_plans():
    arm = make_arm()
    sk = Skills(None, arm)
    home = load_yaml("poses.yaml")["home"]
    nod = sk.plan({"do": "gesture", "name": "nod"}, home)
    check("nod returns to start pose", nod[-1]["joints"] == {"wrist_pitch": home["wrist_pitch"]}, nod[-1])
    check("nod moves wrist_pitch only", all(set(w["joints"]) == {"wrist_pitch"} for w in nod))
    jog = sk.plan({"do": "jog", "joint": "base_yaw", "degrees": 18}, home)
    delta_deg = (jog[0]["joints"]["base_yaw"] - home["base_yaw"]) * arm.deg_per_norm("base_yaw")
    check("jog 18 deg -> 18 deg of travel", abs(delta_deg - 18) < 1e-6, delta_deg)
    g = sk.plan({"do": "gripper", "state": "open"}, home)
    check("gripper open -> 0.65", g[0]["joints"] == {"gripper": 0.65})
    check("home -> taught pose", sk.plan({"do": "gesture", "name": "home"}, home)[0]["joints"] == home)


class FakeBrain:
    """Streams canned steps with a delay, like the real one, without calling the API."""

    def __init__(self, steps, delay=0.05):
        self.steps, self.delay = steps, delay

    async def decide(self, jpeg, instruction, context, on_step, log):
        log["t_sent"] = time.monotonic()
        for s in self.steps:
            await asyncio.sleep(self.delay)
            log.setdefault("t_first_step", time.monotonic())
            on_step(json.dumps(s))
        return {"steps": self.steps, "say": "ok"}


async def run_app(steps, after=None, sim_speed=4.0):
    import app as A
    A.LOG_FILE = Path("/tmp") / f"normacore-test-{time.time_ns()}.jsonl"
    io = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", load_yaml("poses.yaml")["home"], sim_speed=sim_speed)
    a = A.App(io, sim=True)
    await a.setup()
    a.brain = FakeBrain(steps)
    a.on_line("test request")
    if after:
        await after(a)
    await a.task
    a.executor.worker.cancel()
    return a, io, json.loads(A.LOG_FILE.read_text().splitlines()[-1])


def test_app_flow():
    steps = [{"do": "jog", "joint": "base_yaw", "degrees": 15}, {"do": "gesture", "name": "nod"},
             {"do": "jog", "joint": "shoulder", "degrees": -99}]   # home shoulder is ~0: past the limit
    a, io, rec = asyncio.run(run_app(steps))
    res = rec["results"]
    check("all 3 steps ran", len(res) == 3, res)
    check("each step reports reached", all(r.get("reached") for r in res), res)
    check("oversized shoulder jog was clamped by limits", any(r.get("clamps") for r in res), res)
    check("log has first-move time", "t_first_move" in rec["ms"], rec["ms"])
    check("history remembers the request", a.history[-1]["instruction"] == "test request")


def test_stop():
    steps = [{"do": "gesture", "name": "wave"}, {"do": "gesture", "name": "shake"}]

    async def press_stop(a):
        await asyncio.sleep(0.25)                  # first gesture is mid-motion
        t = time.monotonic()
        a.on_line("Stop!")
        a._stop_ms = (time.monotonic() - t) * 1000
    a, io, rec = asyncio.run(run_app(steps, press_stop, sim_speed=1.0))
    check("stop halts within 0.5 s (G4)", a._stop_ms < 500, a._stop_ms)
    check("motion aborted, later steps dropped", len(io.sent) == 1, len(io.sent))
    check("stop is logged", rec.get("error") == "stopped by user", rec.get("error"))


def test_reach_and_slow():
    arm = make_arm()
    s, note = parse_step({"do": "reach", "direction": "forward", "degrees": 50, "speed": "slow"}, arm, set())
    check("reach capped at 30", s and s["degrees"] == 30 and "clamped" in note, (s, note))
    s, _ = parse_step({"do": "reach", "direction": "sideways", "degrees": 10, "speed": "slow"}, arm, set())
    check("bad reach direction rejected", s is None)
    s, _ = parse_step({"do": "jog", "joint": "elbow", "degrees": 5, "speed": "warp"}, arm, set())
    check("bad speed rejected", s is None)
    home = load_yaml("poses.yaml")["home"]
    p = Skills(None, arm).plan({"do": "reach", "direction": "forward", "degrees": 20, "speed": "slow"}, home)[0]
    ds = (p["joints"]["shoulder"] - home["shoulder"]) * arm.deg_per_norm("shoulder")
    de = (p["joints"]["elbow"] - home["elbow"]) * arm.deg_per_norm("elbow")
    check("reach forward: shoulder +20, elbow -20, one keyframe (together)", abs(ds - 20) < 1e-6 and abs(de + 20) < 1e-6,
          (ds, de))
    a, io, rec = asyncio.run(run_app([{"do": "reach", "direction": "forward", "degrees": 15, "speed": "slow"},
                                      {"do": "jog", "joint": "base_yaw", "degrees": 5, "speed": "normal"}]))
    speeds = [x["speed_scale"] for x in io.sent]
    check("slow runs at half the normal speed", speeds == [0.25, 0.5], speeds)


def test_say_follows_outcome():
    # a step the safety gate skips (B has no taught pose) must not be announced as done
    a, io, rec = asyncio.run(run_app([{"do": "gesture", "name": "nod"}, {"do": "point", "target": "B"}]))
    check("Claude's sentence replaced when a step is skipped", rec["say"].startswith("I couldn't do"), rec["say"])
    check("replacement says what did run", "I did nod" in rec["say"], rec["say"])
    check("Claude's original sentence kept in the log", rec["say_claude"] == "ok")
    check("memory carries the outcome", "no taught pose" in a.history[-1]["outcome"], a.history[-1])
    # clamped but reached: keep the sentence, add the limit note
    a, io, rec = asyncio.run(run_app([{"do": "jog", "joint": "shoulder", "degrees": -30}]))
    check("clamp adds a limit note", rec["say"].endswith("(I stopped at my joint limit.)"), rec["say"])
    a, io, rec = asyncio.run(run_app([{"do": "gesture", "name": "wave"}]))
    check("clean run keeps Claude's sentence", rec["say"] == "ok", rec["say"])


def test_leader_mode_offers_no_taught_poses():
    from brain import GESTURES, build_schema, build_system
    arm = make_arm()
    g = [x for x in GESTURES if x not in ("home", "ready")]
    schema = json.dumps(build_schema(arm, [], True, g))
    system = build_system(arm, [], True, g)
    check("schema has no home/ready", '"home"' not in schema and '"ready"' not in schema)
    check("prompt doesn't offer home", "home (rest pose)" not in system)
    check("full list still offered on the follower", '"home"' in json.dumps(build_schema(arm, [], True)))


def test_record_and_repeat():
    import app as A
    import recorder as R
    R.REC_DIR = Path("/tmp") / f"normacore-rec-{time.time_ns()}"
    A.LOG_FILE = Path("/tmp") / f"normacore-test-{time.time_ns()}.jsonl"
    home = load_yaml("poses.yaml")["home"]
    # pure helpers: 1 s still, 2 s moving base_yaw 0.50 -> 0.60, 1 s still, at 100 Hz
    raw = []
    for i in range(400):
        t = i / 100
        x = 0.50 if t < 1 else 0.60 if t > 3 else 0.50 + 0.05 * (t - 1)
        raw.append((t, {**home, "base_yaw": x}))
    tr = R.trim_still(raw)
    check("still ends trimmed", 1.75 < tr[-1][0] < 2.0, tr[-1][0])
    th = R.thin(tr)
    check("thinned to ~20 Hz", 34 <= len(th) <= 42, len(th))
    check("travel measured", abs(R.travel(th)["base_yaw"] - 0.10) < 1e-6)
    check("all-still recording gives nothing", R.trim_still([(0, home), (1, dict(home))]) == [])

    async def flow():
        io = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", home, sim_speed=20.0)
        a = A.App(io, sim=True)
        await a.setup()
        out = []

        async def say(line):
            a.on_line(line)
            if a.task:
                await a.task
        import builtins
        real_print = builtins.print
        builtins.print = lambda *x, **k: out.append(" ".join(map(str, x)))
        try:
            await say("record")                         # sim joints start with torque on
            refused = any("torque is on" in o for o in out)
            await say("relax")
            await say("record")
            io.fake_history = raw
            await say("done")
            await say("repeat the same movement")
        finally:
            builtins.print = real_print
        a.executor.worker.cancel()
        return a, io, out, refused
    a, io, out, refused = asyncio.run(flow())
    check("record refused while torque is on", refused)
    check("recording saved", (R.REC_DIR / "latest.json").exists(), out)
    sent = io.sent[-1]
    check("replay sends timed samples", all("t" in w for w in sent["waypoints"]), sent["waypoints"][:2])
    check("replay default is the recorded speed", sent["speed_scale"] == 1.0, sent["speed_scale"])
    check("replay ends at the recorded end pose", abs(io.pose["base_yaw"] - 0.60) < 1e-3, io.pose["base_yaw"])
    check("replay reported done", any("✓ done" in o for o in out), out)
    wps, _ = check_waypoints(make_arm(), [{"t": 0.5, "joints": {"shoulder": 0.95}}])
    check("limits still apply to recorded samples", wps[0]["joints"]["shoulder"] <= 0.60 and wps[0]["t"] == 0.5)


def test_goal_loop():
    import app as A
    import goal as G
    A.LOG_FILE = Path("/tmp") / f"normacore-test-{time.time_ns()}.jsonl"
    G.LOG = Path("/tmp") / f"normacore-goal-{time.time_ns()}.jsonl"
    answers = [
        {"observation": "cup at the left edge of the gripper view", "steps": [
            {"do": "jog", "joint": "base_yaw", "degrees": 40, "speed": "normal"},
            {"do": "jog", "joint": "wrist_pitch", "degrees": -5, "speed": "slow"},
            {"do": "jog", "joint": "elbow", "degrees": 5, "speed": "normal"}], "status": "continue", "say": "turning"},
        {"observation": "cup centred", "steps": [], "status": "done", "say": "centred"},
    ]

    import llm

    class FakeModel:   # stands in for GoalRunner.ask (no API call)
        def __init__(self, delay=0.01):
            self.n, self.delay, self.seen = 0, delay, []

        async def __call__(self, system, parts, schema, m):
            self.seen.append(parts)
            await asyncio.sleep(self.delay)
            a = answers[min(self.n, len(answers) - 1)]
            self.n += 1
            return llm.Answer(json.dumps(a), {"input_tokens": 1, "output_tokens": 1}, m.id, True, "end_turn")

    async def run(delay=0.01, stop_after=None):
        io = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", load_yaml("poses.yaml")["home"], sim_speed=20.0)
        a = A.App(io, sim=True)
        await a.setup()
        a.goal.cams.use_rgbd = False
        a.goal.client = a.goal.ask = FakeModel(delay)
        lines = []
        a.goal.out = lambda t: lines.append(t)
        start = io.pose["base_yaw"]
        a.on_line("goal: centre the cup")
        if stop_after:
            await asyncio.sleep(stop_after)
            a.on_line("stop")
        await a.task
        a.executor.worker.cancel()
        return a, io, lines, start
    a, io, lines, start = asyncio.run(run())
    moved = (io.pose["base_yaw"] - start) * make_arm().deg_per_norm("base_yaw")
    check("goal: ends when Claude says done", lines[-1].startswith("★ Goal ended: done"), lines[-1])
    check("goal: jog capped at 15 deg per step", abs(moved - 15) < 1.0, moved)
    check("goal: at most 2 steps per turn", len(io.sent) == 2, len(io.sent))
    sent = a.goal.client.seen[0]
    imgs = [p for p in sent if isinstance(p, bytes)]
    check("goal: camera images sent", len(imgs) >= 1, len(imgs))
    turn2 = a.goal.client.seen[1][-1]
    check("goal: turn 2 knows what turn 1 did", "jog base_yaw +15" in turn2, turn2[-300:])
    a, io, lines, _ = asyncio.run(run(delay=1.0, stop_after=0.3))
    check("goal: stop during Claude's thinking ends the loop, no move", "stopped by you" in lines[-1] and not io.sent,
          (lines[-1], len(io.sent)))


def test_handoff_to_goal():
    import app as A
    import goal as G
    from brain import build_schema
    G.LOG = Path("/tmp") / f"normacore-goal-{time.time_ns()}.jsonl"
    s, _ = parse_step({"do": "goal", "goal": "grasp the motor, the black object"}, make_arm(), set())
    check("goal step accepted", s == {"do": "goal", "goal": "grasp the motor, the black object"}, s)
    check("goal step offered to Claude", '"goal"' in json.dumps(build_schema(make_arm(), [])))
    from brain import STEP_RE
    check("an answer with no moves yields no steps", STEP_RE.findall('{"steps": [], "say": "I see a pen"}') == [])

    async def flow():
        A.LOG_FILE = Path("/tmp") / f"normacore-test-{time.time_ns()}.jsonl"
        io = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", load_yaml("poses.yaml")["home"], sim_speed=20.0)
        a = A.App(io, sim=True)
        await a.setup()
        a.brain = FakeBrain([{"do": "goal", "goal": "grasp the motor"}])
        got = []

        async def fake_run(goal):
            got.append(goal)
            return "done after 1 turns"
        a.goal.run = fake_run
        a.on_line("grab that black motor")
        await a.task
        await asyncio.sleep(0.05)       # the loop starts right after the request ends
        await a.task
        a.executor.worker.cancel()
        return a, got
    a, got = asyncio.run(flow())
    check("command request hands over to the look-and-move loop", got == ["grasp the motor"], got)
    check("loop outcome remembered for follow-ups", "done" in a.history[-1]["outcome"], a.history[-1])


def test_grasp_report():
    class GraspIO(SimIO):
        async def play(self, waypoints, speed_scale):   # gripper stops at 0.51 on an object
            r = await super().play(waypoints, speed_scale)
            if "gripper" in waypoints[-1]["joints"]:
                self.pose["gripper"] = 0.51
                r["grasp_detected"] = False   # low current: only the stall shows it
            return r
    io = GraspIO(ROOT / "bench" / "frames" / "sample_desk.jpg", load_yaml("poses.yaml")["home"], sim_speed=20.0)
    sk = Skills(io, make_arm(extra_limits=False))
    r = asyncio.run(sk.run({"do": "gripper", "state": "close"}))
    check("gripper stopping short on an object = grasped, not a failure", r["reached"] and r["grasp"] is True, r)
    io2 = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", load_yaml("poses.yaml")["home"], sim_speed=20.0)
    r = asyncio.run(Skills(io2, make_arm(extra_limits=False)).run({"do": "gripper", "state": "close"}))
    check("gripper closing fully = nothing held", r["reached"] and r["grasp"] is False, r)


def test_touch():
    arm = make_arm(extra_limits=False)
    s, note = parse_step({"do": "touch", "joint": "elbow", "degrees": 50, "speed": "slow"}, arm, set())
    check("touch travel capped", s and s["degrees"] == 30 and "clamped" in note, (s, note))
    check("touch with 0 degrees rejected", parse_step({"do": "touch", "joint": "elbow", "degrees": 0}, arm, set())[0] is None)
    check("touch on a roll joint rejected", parse_step({"do": "touch", "joint": "forearm_roll", "degrees": 5}, arm, set())[0] is None)
    from brain import build_schema
    from goal import goal_schema
    check("touch offered in both modes", '"touch"' in json.dumps(build_schema(arm, [])) and
          '"touch"' in json.dumps(goal_schema(arm, True)))
    home = load_yaml("poses.yaml")["home"]
    dpn = arm.deg_per_norm("elbow")

    def sim():
        return SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", dict(home), sim_speed=200.0)
    # a surface 6 degrees further along elbow +
    io = sim(); io.surface = {"elbow": (home["elbow"] + 6 / dpn, +1)}
    r = asyncio.run(Skills(io, arm).run({"do": "touch", "joint": "elbow", "degrees": 20, "speed": "slow"}))
    over = (io.sent[-1]["waypoints"][0]["joints"]["elbow"] - io.surface["elbow"][0]) * dpn
    check("touch stops on the surface, not at full travel", r["contact"] is True and 5 <= r["moved_deg"] <= 7, r)
    check("then rests only ~0.5° past contact (light pressure)", 0 < over <= 0.6, over)
    check("touch never pushed far past the surface", max(w["waypoints"][0]["joints"]["elbow"] for w in io.sent)
          <= io.surface["elbow"][0] + 2.6 / dpn)
    # nothing there: full travel, no contact
    io = sim()
    r = asyncio.run(Skills(io, arm).run({"do": "touch", "joint": "elbow", "degrees": 5, "speed": "normal"}))
    check("no surface: travels the full 5° and reports no contact", r["contact"] is False and r["moved_deg"] == 5, r)
    # the other direction
    io = sim(); io.surface = {"elbow": (home["elbow"] - 3 / dpn, -1)}
    r = asyncio.run(Skills(io, arm).run({"do": "touch", "joint": "elbow", "degrees": -10, "speed": "normal"}))
    check("touch works with negative degrees too", r["contact"] is True and r["moved_deg"] <= 4, r)
    # current alone (servo reaches its target but strains): rises on a watched joint
    class Strain(SimIO):
        async def read_joints(self):
            m = await super().read_joints()
            if self.pose["elbow"] > home["elbow"] + 4 / dpn:
                m["shoulder"]["current"] = 20
            return m
    io = Strain(ROOT / "bench" / "frames" / "sample_desk.jpg", dict(home), sim_speed=200.0)
    r = asyncio.run(Skills(io, arm).run({"do": "touch", "joint": "elbow", "degrees": 15, "speed": "normal"}))
    check("a current rise on another joint also counts as contact", r["contact"] is True and "shoulder current" in r["signal"], r)


def test_draw_from_leader():
    import app as A
    import recorder as R
    R.REC_DIR = Path("/tmp") / f"normacore-rec-{time.time_ns()}"
    A.LOG_FILE = Path("/tmp") / f"normacore-test-{time.time_ns()}.jsonl"
    check("names cleaned", [R.safe_name(x) for x in ("a C", "the C shape", "smiley face!")] == ["c", "c", "smiley-face"])
    check("'draw a C slowly' parsed", A.split_named("draw a c slowly") == ("draw", "c"))
    home = load_yaml("poses.yaml")["home"]
    # a "C" drawn by the follower while it mirrors the leader: base_yaw and shoulder trace an arc,
    # gripper wiggles (the person squeezed the leader's trigger) - must NOT be replayed in pen mode
    import math
    raw = [(0.0, dict(home)), (0.5, dict(home))]
    for i in range(1, 61):
        a_ = math.pi * (0.25 + 1.5 * i / 60)
        raw.append((0.5 + i / 30, {**home, "base_yaw": home["base_yaw"] + 0.05 * math.cos(a_),
                                   "shoulder": home["shoulder"] + 0.05 + 0.05 * math.sin(a_),
                                   "gripper": 0.3 + 0.2 * (i % 2)}))
    raw.append((3.0, dict(raw[-1][1])))

    async def flow():
        io = SimIO(ROOT / "bench" / "frames" / "sample_desk.jpg", home, sim_speed=40.0)
        a = A.App(io, sim=True)
        await a.setup()
        out = []

        async def say(line):
            a.on_line(line)
            if a.task:
                await a.task
        import builtins
        real_print = builtins.print
        builtins.print = lambda *x, **k: out.append(" ".join(map(str, x)))
        try:
            await say("teleop on")                    # Station teleop on: follower torque is on, that's fine
            await say("record c")
            io.fake_history = raw
            io.fake_leader_history = [(t, {**p, "base_yaw": p["base_yaw"] + 0.001}) for t, p in raw]
            await say("done")
            await say("draw c")                         # teleop still on -> refused
            refused = any("teleop is still on" in o for o in out)
            await say("teleop off")
            n_before = len(io.sent)
            await say("draw a C")
            sent = io.sent[n_before:]
            await say("draw a smiley")                 # not saved: goes to Claude (sim brain), not a local replay
        finally:
            builtins.print = real_print
        a.executor.worker.cancel()
        return a, io, out, refused, sent
    a, io, out, refused, sent = asyncio.run(flow())
    check("recording while teleop is on is allowed", any("Move the LEADER arm" in o for o in out), out[:3])
    check("named recording saved", (R.REC_DIR / "named" / "c.json").exists(), out)
    saved = json.loads((R.REC_DIR / "named" / "c.json").read_text())
    check("teleop recording keeps the leader's path too", saved.get("leader", {}).get("samples", 0) > 10
          and saved["leader"]["bus"] == "5B91043428", list(saved))
    check("'draw c' refused while teleop still drives the follower", refused)
    check("'teleop on' / 'teleop off' handled locally", any("teleop ON" in o for o in out) and any("teleop OFF" in o for o in out), out)
    check("pen draw = approach, stroke, lift", len(sent) == 3, [len(s["waypoints"]) for s in sent])
    if len(sent) == 3:
        dpn = a.arm.deg_per_norm("elbow")
        first = sent[1]["waypoints"][0]["joints"]
        check("approach goes to the start with the elbow lifted ~6°, then lowers onto it",
              abs((first["elbow"] - sent[0]["waypoints"][0]["joints"]["elbow"]) * dpn - 6) < 0.5
              and sent[0]["waypoints"][1]["joints"] == first, (sent[0]["waypoints"], first))
        check("stroke is the recorded samples", all("t" in w for w in sent[1]["waypoints"]))
        check("pen mode never touches the gripper", all("gripper" not in w["joints"] for s in sent for w in s["waypoints"]))
        check("lifts again at the end", sent[2]["waypoints"][-1]["joints"]["elbow"] < sent[1]["waypoints"][-1]["joints"]["elbow"])
    check("Claude is told the saved names", "Saved recordings: c." in __import__("brain").describe_state(
        make_arm(), home, [], a.recorder.names()))


def test_usage():
    u = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 10000, "cache_creation_input_tokens": 0}
    c = U.cost("claude-opus-5-5", u)
    check("cost = 1k*$4 + 500*$20 + 10k*$0.20 per MTok", abs(c - (0.004 + 0.010 + 0.002)) < 1e-9, c)
    check("dated model ids priced as their family", U.price_for("claude-opus-5-5-20261001") == U.PRICES["claude-opus-5-5"])
    m = U.Meter(Path("/tmp") / f"normacore-usage-{time.time_ns()}.jsonl")
    m.add("command", "claude-opus-5-5", u); m.new_request(); m.add("goal", "claude-opus-5-5", u)
    s = m.summary()
    check("request resets, session and today accumulate",
          s["request"]["calls"] == 1 and s["session"]["calls"] == 2 and s["today"]["calls"] == 2, s)
    check("today survives a restart (re-read from the log)", U.Meter(m.log).today["calls"] == 2)
    check("short format", U.fmt(u, c) == "11.0k in (10.0k cached) · 500 out · $0.0160", U.fmt(u, c))


def test_model_switch():
    import os
    check("default model is Claude Opus 5.5", L.current().id == "claude-opus-5-5")
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        try:
            L.select("gpt-6.2-astra"); refused = False
        except ValueError:
            refused = True
        check("GPT refused without an OpenAI key", refused and L.current().id == "claude-opus-5-5")
        os.environ["OPENAI_API_KEY"] = "test-not-a-real-key"
        L.select("gpt-6.2-astra")
        check("switch to GPT-6.2 Astra is remembered", L.current().id == "gpt-6.2-astra")
        check("listing marks the current model", [m["id"] for m in L.listing() if m["current"]] == ["gpt-6.2-astra"])
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
        L.STATE.unlink(missing_ok=True)
    check("unknown model refused", not _raises_not(lambda: L.select("gpt-9")))
    check("GPT priced", U.price_for("gpt-6.2-astra") == (8.0, 40.0, 0.40, 10.0))

    # the GPT request: same schema (strict), images as data URLs, streamed text, usage normalised
    seen = {}

    class Ev:
        def __init__(self, **k):
            self.__dict__.update(k)

    class FakeResponses:
        async def create(self, **kw):
            seen.update(kw)

            async def gen():
                for piece in ('{"steps": [{"do": "gripper", "state": "open"}]', ', "say": "ok"}'):
                    yield Ev(type="response.output_text.delta", delta=piece)
                usage = Ev(input_tokens=1200, output_tokens=80,
                           input_tokens_details=Ev(cached_tokens=1000, cache_write_tokens=0))
                yield Ev(type="response.completed", response=Ev(status="completed", model="gpt-6.2-astra",
                                                                 usage=usage, incomplete_details=None))
            return gen()
    L._clients[("openai", 25.0)] = Ev(responses=FakeResponses())
    chunks = []
    ans = asyncio.run(L.ask("sys", [b"\xff\xd8jpeg", "open it"], {"type": "object"}, chunks.append,
                            model=L.MODELS["gpt-6.2-astra"]))
    del L._clients[("openai", 25.0)]
    check("GPT: structured output requested with strict JSON schema",
          seen["text"]["format"]["type"] == "json_schema" and seen["text"]["format"]["strict"] is True)
    check("GPT: image sent as a data URL, then the text",
          seen["input"][0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")
          and seen["input"][0]["content"][1] == {"type": "input_text", "text": "open it"})
    check("GPT: streamed text arrives in pieces", len(chunks) == 2 and json.loads(ans.text)["say"] == "ok")
    check("GPT: cached tokens split out of input (Anthropic-style usage)",
          ans.usage == {"input_tokens": 200, "output_tokens": 80, "cache_read_input_tokens": 1000,
                        "cache_creation_input_tokens": 0}, ans.usage)


def _raises_not(f):
    try:
        f()
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    test_arm_math()
    test_safety()
    test_limits_switch()
    test_skill_plans()
    test_app_flow()
    test_stop()
    test_reach_and_slow()
    test_say_follows_outcome()
    test_leader_mode_offers_no_taught_poses()
    test_record_and_repeat()
    test_goal_loop()
    test_handoff_to_goal()
    test_grasp_report()
    test_touch()
    test_draw_from_leader()
    test_usage()
    test_model_switch()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)
