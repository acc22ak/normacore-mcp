"""Goal mode: a look-act loop toward a goal typed in the panel ("goal: centre the cup in the gripper view").

Each turn: grab every live camera view -> one Claude call (images + joint angles + goal + what
happened in earlier turns) -> Claude returns what it sees, up to 2 small steps, and whether the goal
is done -> run the steps (same safety gate as command mode) -> look again.

Ends when Claude says done or stuck, after MAX_TURNS or MAX_S, when a step is refused or the arm
stops on its own (current / falling behind), or when the person types stop.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import llm
from arm import Arm
from safety import parse_step
from usage import METER, fmt
from vision import Cameras

MAX_TURNS = 20
MAX_S = 180.0
MAX_STEPS_PER_TURN = 2
GOAL_JOG_DEG = 15          # smaller moves than command mode: look again after each
CALL_TIMEOUT_S = 15.0
LOG = Path(__file__).resolve().parents[1] / "logs" / "goals.jsonl"


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def goal_schema(arm: Arm, has_rgbd: bool) -> dict:
    speed = {"type": "string", "enum": ["slow", "normal"]}
    steps = [
        _obj({"do": {"type": "string", "enum": ["jog"]}, "joint": {"type": "string", "enum": arm.jog_joints},
              "degrees": {"type": "integer"}, "speed": speed}),
        _obj({"do": {"type": "string", "enum": ["reach"]}, "direction": {"type": "string", "enum": ["forward", "back"]},
              "degrees": {"type": "integer"}, "speed": speed}),
        _obj({"do": {"type": "string", "enum": ["gripper"]}, "state": {"type": "string", "enum": ["open", "close"]}}),
        _obj({"do": {"type": "string", "enum": ["touch"]}, "joint": {"type": "string", "enum": arm.jog_joints},
              "degrees": {"type": "integer"}, "speed": speed}),
    ]
    if has_rgbd:
        steps.append(_obj({"do": {"type": "string", "enum": ["measure"]},
                           "x": {"type": "integer"}, "y": {"type": "integer"}}))
        steps.append(_obj({"do": {"type": "string", "enum": ["distance"]},
                           "x1": {"type": "integer"}, "y1": {"type": "integer"},
                           "x2": {"type": "integer"}, "y2": {"type": "integer"}}))
    return _obj({"observation": {"type": "string"},
                 "steps": {"type": "array", "items": {"anyOf": steps}},
                 "status": {"type": "string", "enum": ["continue", "done", "stuck"]},
                 "say": {"type": "string"}})


def goal_system(arm: Arm, has_rgbd: bool) -> str:
    rows = "\n".join(f"- {n}: +degrees {arm.joints[n].plus}" for n in arm.jog_joints)
    return (
        "You drive a small desktop robot arm (ElRobot: 7 joints plus a parallel gripper) toward a goal "
        "a person gave you, by looking through its cameras and making small moves. This is one turn "
        "of a loop: you see fresh images, choose at most 2 small steps, they run, and you look again.\n\n"
        "Steps:\n"
        f"- jog: turn one joint by up to {GOAL_JOG_DEG} degrees\n"
        "- reach: stretch forward (shoulder out + elbow open together) or pull back, up to "
        f"{GOAL_JOG_DEG} degrees\n"
        "- gripper: open or close\n"
        "- touch: move one joint in 1-degree steps until the tool or fingers touch something, sensed "
        "from the motors (a joint held back or its load current rising), then stop resting lightly on it. "
        f"degrees = the most it may travel (up to {GOAL_JOG_DEG}); its sign = the direction. The result says "
        "TOUCHED with how far it went, or no contact. This is the reliable way to put a pen on paper: the "
        "cameras can't tell touching from a few millimetres above. To lift off again, jog the same joint the "
        "other way by 2-5 degrees.\n"
        + ("- measure: physical HEIGHT at a pixel (x, y) of the rgbd colour image: how high the top of "
           "what is there sits above the table, and above the surface right around it. The answer "
           "arrives next turn. Does not move the arm.\n"
           "- distance: straight-line distance in cm between two pixels of the rgbd colour image "
           "(e.g. the two ends of a drawn line). Does not move the arm.\n" if has_rgbd else "")
        + "\nJoints and what positive degrees do:\n" + rows + "\n\n"
        "Rules:\n"
        "- First write 'observation': what you see that matters for the goal, in which image, and "
        "where (left/right/up/down, near/far). If the target isn't visible, say so.\n"
        "- Move in small increments (3-10 degrees) when close to the target, larger (up to "
        f"{GOAL_JOG_DEG}) when far. Never repeat a move that made things worse; reverse instead.\n"
        "- In the gripper camera, the gripper fingers are fixed in the image: to centre a target, "
        "turn base_yaw for left/right and wrist_pitch or reach for up/down/closer. Check the "
        "earlier turns to learn which way each joint moves the view.\n"
        "- After 'gripper close', the result says GRASPED (stopped on an object) or 'closed fully' (missed). "
        "For a pick-up goal, a grasp isn't done until the object is lifted: reach back or raise the arm a "
        "little and check the images that it came up with the gripper.\n"
        + ("- Judge height and contact with numbers, not by eye: the rgbd height image shows height above "
           "the table, and 'measure' on a tip (pen, finger, gripper) tells you how far it is above the "
           "surface under it. Above-surface under about 0.5 cm means touching; depth noise is about "
           "+-0.4 cm at this range, and thin things (a pen tip) may read as the surface behind them, so "
           "measure a pixel on the tip itself and confirm with the images. Measure before and after a "
           "move that should lower something onto a surface.\n" if has_rgbd else "") +
        "- status 'done' only when the images show the goal is achieved; 'stuck' if it can't be "
        "done, the target is lost for 3 turns, or something looks unsafe. Otherwise 'continue'.\n"
        "- 'say' is one short sentence for the person about this turn."
    )


class GoalRunner:
    def __init__(self, app, out):
        self.app, self.out = app, out
        self.abort = asyncio.Event()
        # with --photo the desk view is a saved picture, not live: never show it to Claude as a camera
        roles = ("gripper",) if getattr(app.io, "photo_desk", False) else ("gripper", "desk")
        self.cams = Cameras(app.io, roles=roles)

    def stop(self) -> None:
        self.abort.set()

    async def ask(self, system: str, parts: list, schema: dict, m) -> "llm.Answer":
        """One look-and-decide call to the selected model (tests replace this)."""
        return await llm.ask(system, parts, schema, model=m, timeout_s=m.timeout_s * 1.5)

    async def _until_abort(self, coro):
        """Run coro unless stop arrives first; returns None if stopped."""
        task = asyncio.ensure_future(coro)
        stopper = asyncio.ensure_future(self.abort.wait())
        done, _ = await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
        stopper.cancel()
        if task in done:
            return task.result()
        task.cancel()
        return None

    async def run(self, goal: str) -> None:
        self.abort.clear()
        arm, skills, io = self.app.arm, self.app.skills, self.app.io
        history: list[dict] = []
        pending_measures: list[str] = []
        t0 = time.monotonic()
        self.out(f"◎ Goal: {goal}  (max {MAX_TURNS} turns / {MAX_S:g} s; type stop to halt)")
        end = "turn limit reached"
        for turn in range(1, MAX_TURNS + 1):
            if self.abort.is_set():
                end = "stopped by you"
                break
            if time.monotonic() - t0 > MAX_S:
                end = "time limit reached"
                break
            rec = {"goal": goal, "turn": turn, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tt = time.monotonic()
            views, notes = await self.cams.grab()
            self.app.views = views
            self.app.views_version = getattr(self.app, "views_version", 0) + 1
            if not views:
                end = "no camera is available: " + "; ".join(notes)
                break
            has_rgbd = any(v.name == "rgbd" for v in views)
            motors = await io.read_joints()
            arm.update_ranges(motors)
            joints = {n: m["position_norm"] for n, m in motors.items()}
            content = []
            for v in views:
                content += [v.label + ":", v.jpeg]
            state = ", ".join(f"{n} {arm.norm_to_deg(n, joints[n]):.0f}°" for n in arm.jog_joints)
            hist = "\n".join(f"- turn {h['turn']}: saw: {h['observation']} | did: {h['did']} | result: {h['result']}"
                             for h in history[-6:]) or "- (first turn)"
            text = (f"GOAL: {goal}\n\nTurn {turn} of at most {MAX_TURNS}. Joint angles: {state}. "
                    f"Gripper opening {joints.get('gripper', 0):.2f} (about 0.01 = fingers touching, 0.65 = open).\n"
                    + (f"Cameras not available: {'; '.join(notes)}\n" if notes else "")
                    + (f"Measurements you asked for: {'; '.join(pending_measures)}\n" if pending_measures else "")
                    + f"Earlier turns:\n{hist}")
            content.append(text)
            pending_measures = []
            m = llm.current()
            try:
                msg = await self._until_abort(self.ask(goal_system(arm, has_rgbd), content,
                                                       goal_schema(arm, has_rgbd), m))
            except (llm.LLMError, asyncio.TimeoutError) as e:
                end = f"{m.label} call failed: {type(e).__name__}: {e}"
                break
            if msg is None:
                end = "stopped by you"
                break
            rec["ms_think"] = round((time.monotonic() - tt) * 1000)
            rec["usage"], rec["model"] = msg.usage, msg.model
            rec["usd"] = METER.add("goal", msg.model, msg.usage)
            if not msg.ok:
                end = f"{m.label} did not answer ({msg.stop})"
                break
            try:
                ans = json.loads(msg.text)
            except ValueError:
                end = f"{m.label} gave no readable answer"
                break
            rec.update(observation=ans["observation"], steps=ans["steps"], status=ans["status"], say=ans["say"],
                       cameras=[v.name for v in views])
            self.out(f"👁 turn {turn} ({', '.join(v.name for v in views)}, {rec['ms_think'] / 1000:.1f}s, "
                     f"{fmt(rec['usage'], rec['usd'])}): "
                     f"{ans['observation']}")
            did, results = [], []
            for raw in ans["steps"][:MAX_STEPS_PER_TURN]:
                if self.abort.is_set():
                    break
                if raw.get("do") == "distance":
                    m = await self.cams.distance(int(raw["x1"]), int(raw["y1"]), int(raw["x2"]), int(raw["y2"]))
                    pending_measures.append(m)
                    did.append(f"distance ({raw['x1']},{raw['y1']})-({raw['x2']},{raw['y2']})")
                    results.append(m)
                    self.out(f"  📏 {m}")
                    continue
                if raw.get("do") == "measure":
                    m = await self.cams.measure(int(raw["x"]), int(raw["y"]))
                    pending_measures.append(m)
                    did.append(f"measure ({raw['x']}, {raw['y']})")
                    results.append(m)
                    self.out(f"  📏 {m}")
                    continue
                raw = dict(raw)
                if "degrees" in raw and isinstance(raw["degrees"], (int, float)) and not isinstance(raw["degrees"], bool):
                    raw["degrees"] = max(-GOAL_JOG_DEG, min(GOAL_JOG_DEG, raw["degrees"]))
                step, note = parse_step(raw, arm, set())
                if step is None:
                    results.append(f"refused: {note}")
                    self.out(f"  ✗ refused: {note}")
                    continue
                label = self.app.fmt_step(step)
                self.out(f"  → {label}")
                r = await skills.run(step)
                ok = r["reached"]
                did.append(label)
                if r.get("contact") is True:
                    results.append(f"TOUCHED after {r['moved_deg']:g}° ({r['signal']}); now resting on it lightly")
                elif r.get("contact") is False:
                    results.append(f"no contact: {r['signal']}")
                elif r.get("grasp") is True:
                    results.append(f"GRASPED: the gripper stopped at {r['gripper_at']:.2f} on its way to fully closed "
                                   f"({skills.grip['close']}), so something is held between the fingers")
                elif r.get("grasp") is False:
                    results.append("closed fully: nothing between the fingers")
                else:
                    results.append("reached" if ok else f"not reached ({r.get('abort_reason') or r.get('errors')})")
                self.out(f"    {'✓ ' if ok else '✗ '}{results[-1]}")
                if r.get("abort_reason") and "stopped by" not in str(r.get("abort_reason")):
                    end = f"the arm stopped itself: {r['abort_reason']}"
                    self.abort.set()
            history.append({"turn": turn, "observation": ans["observation"], "did": ", ".join(did) or "nothing",
                            "result": "; ".join(results) or "-"})
            rec["results"] = results
            rec["ms_turn"] = round((time.monotonic() - tt) * 1000)
            self._log(rec)
            self.out(f"  “{ans['say']}”")
            if ans["status"] in ("done", "stuck"):
                end = ans["status"]
                break
            if self.abort.is_set():
                end = end if end.startswith("the arm") else "stopped by you"
                break
        took = time.monotonic() - t0
        mark = "★" if end == "done" else "■"
        self.out(f"{mark} Goal ended: {end} ({len(history)} turns, {took:.0f} s)")
        last = history[-1]["observation"] if history else ""
        return f"{end} after {len(history)} turns" + (f"; last saw: {last}" if last else "")

    def _log(self, rec: dict) -> None:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
