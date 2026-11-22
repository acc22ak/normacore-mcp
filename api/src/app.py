"""Talk to the arm in the terminal.

  .venv/bin/python src/app.py            # real arm + cameras through Station
  .venv/bin/python src/app.py --sim      # no hardware: saved desk photo + simulated arm

Type a request and press Enter. "stop" (or stop!, halt, freeze) halts the arm at once without
asking Claude. Ctrl-D quits. One JSON line per request goes to logs/runs.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


import perceive  # noqa: E402
from arm import Arm, load_yaml  # noqa: E402
from brain import GESTURES, Brain, describe_state  # noqa: E402
from safety import parse_step  # noqa: E402
from goal import GoalRunner  # noqa: E402
from recorder import Recorder, safe_name  # noqa: E402
from skills import Skills  # noqa: E402
from usage import METER, fmt as fmt_usage  # noqa: E402
import llm  # noqa: E402
from station_io import RealIO, SimIO, StationError  # noqa: E402

LISTENERS: list = []   # asyncio.Queues of the web panel's open pages; each gets every output line
HISTORY: deque = deque(maxlen=300)   # recent lines, replayed to a page when it (re)connects


def out(*parts, **_) -> None:
    """Show a line in the terminal and on every open web panel."""
    text = " ".join(str(p) for p in parts)
    print(text, flush=True)
    HISTORY.append(text)
    for q in list(LISTENERS):
        try:
            q.put_nowait(text)
        except Exception:
            pass


STOP_WORDS = {"stop", "halt", "freeze", "estop"}
# handled locally, instantly, without Claude
RECORD_WORDS = {"record", "start recording", "rec"}
DONE_WORDS = {"done", "stop recording", "finish", "finished", "end recording"}
RELAX_WORDS = {"relax", "torque off", "go limp"}
REPEAT_PREFIXES = ("repeat", "replay")
TELEOP_WORDS = {"teleop on": True, "teleop off": False, "start teleop": True, "stop teleop": False,
                "teleoperation on": True, "teleoperation off": False}
SPEED_WORDS = ("at full speed", "full speed", "original speed", "same speed", "slowly", "slow", "fast")


def split_named(low: str, prefixes=("draw", "repeat", "replay")) -> tuple[str, str] | None:
    """'draw a C slowly' -> ('draw', 'c'); 'repeat' -> ('repeat', ''); None if no prefix matches."""
    for p in prefixes:
        if low == p or low.startswith(p + " "):
            rest = low[len(p):]
            for w in SPEED_WORDS:
                rest = rest.replace(w, " ")
            return p, safe_name(rest)
    return None


def speed_of(low: str) -> float:
    return (1.0 if any(w in low for w in ("full speed", "original speed", "same speed", "fast"))
            else REPLAY_SPEED["slow"] if "slow" in low else REPLAY_SPEED["normal"])
REPLAY_SPEED = {"normal": 1.0, "slow": 0.5}   # normal = the speed it was recorded at
TIMEOUT_S = 8.0
LOG_FILE = ROOT / "logs" / "runs.jsonl"


class Executor:
    """Runs checked steps in order on one task; stop() drops everything pending."""

    def __init__(self, io, arm: Arm, skills: Skills, recorder: Recorder | None = None):
        self.io, self.arm, self.skills, self.recorder = io, arm, skills, recorder
        self.queue: asyncio.Queue = asyncio.Queue()
        self.cancelled = False
        self.log: dict = {}
        self.worker = asyncio.create_task(self._run())

    def begin(self, log: dict) -> None:
        self.cancelled, self.log, self.handoff = False, log, None
        log["results"] = []

    def submit(self, raw: str) -> None:
        if not self.cancelled:
            self.queue.put_nowait(raw)

    def stop(self) -> None:
        self.cancelled = True
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.io.stop()

    async def drain(self) -> None:
        await self.queue.join()

    async def _run(self) -> None:
        while True:
            raw = await self.queue.get()
            try:
                if self.cancelled:
                    continue
                step, note = parse_step(raw, self.arm, self.skills.taught_slots)
                if step is None:
                    self.log["results"].append({"raw": raw, "skipped": note})
                    out(f"  ✗ skipped: {note}")
                    continue
                self.log.setdefault("t_first_move", time.monotonic())
                out(f"  → {fmt_step(step)}" + (f"  ({note})" if note else ""), flush=True)
                try:
                    if step["do"] == "goal":   # started by the App once this request finishes
                        self.handoff = step["goal"]
                        r = {"reached": True, "abort_reason": None, "errors": {}, "clamps": []}
                    elif step["do"] == "replay":
                        r = await self.recorder.replay(REPLAY_SPEED[step["speed"]], step.get("name") or None,
                                                       step.get("pen", False))
                    else:
                        r = await self.skills.run(step)
                except Exception as e:  # Station error, refused move, ...: hold and drop the rest
                    r = {"reached": False, "abort_reason": f"{type(e).__name__}: {e}"}
                    self.stop()
                    await self.io.hold()
                r["step"], r["t"] = step, time.monotonic()
                self.log["results"].append({k: v for k, v in r.items() if k != "waypoints"})
                mark = (f"✓ touched after {r['moved_deg']:g}° ({r['signal']})" if r.get("contact") is True else
                        f"✓ no contact: {r['signal']}" if r.get("contact") is False else
                        "✓ grasped (something between the fingers)" if r.get("grasp") is True else
                        "✓ closed fully (nothing between the fingers)" if r.get("grasp") is False else
                        "✓ reached" if r["reached"] else f"✗ not reached ({r.get('abort_reason') or r.get('errors')})")
                out(f"    {mark}" + (f"; clamped: {r['clamps']}" if r.get("clamps") else ""), flush=True)
            finally:
                self.queue.task_done()


def final_say(claude_say: str, results: list[dict], error: str | None) -> tuple[str, str]:
    """The sentence shown (and later spoken) after the steps ran, plus a short outcome note for
    Claude's memory. Claude wrote its sentence before anything moved; when local code skipped,
    blocked or stopped a step, or the arm fell short, the sentence is rebuilt from what happened."""
    if error == "stopped by user":
        return "Stopped.", "stopped by the user before finishing"
    if error:
        return f"I'm holding still: {error}.", f"nothing done ({error})"
    failed = [r for r in results if r.get("skipped") or not r.get("reached", False)]
    clamped = [r for r in results if r.get("clamps")]
    if failed:
        r = failed[0]
        try:   # a skipped step only has Claude's raw JSON
            what = fmt_step(r.get("step") or json.loads(r["raw"]))
        except (KeyError, ValueError, TypeError):
            what = "that"
        why = r.get("skipped") or r.get("abort_reason") or f"ended off target {r.get('errors')}"
        done = [fmt_step(x["step"]) for x in results if x.get("reached")]
        say = f"I couldn't do {what}: {why}." + (f" I did {', '.join(done)}." if done else "")
        return say, say
    if clamped:
        return (f"{claude_say} (I stopped at my joint limit.)".strip(),
                "done, but stopped at a joint limit: " + "; ".join(c for r in clamped for c in r["clamps"]))
    return claude_say, "done" if results else ""


def fmt_step(s: dict) -> str:
    return {"gesture": lambda: s["name"], "point": lambda: f"point {s['target']}",
            "jog": lambda: f"jog {s['joint']} {s['degrees']:+g}°" + (" slowly" if s.get("speed") == "slow" else ""),
            "reach": lambda: f"reach {s['direction']} {s['degrees']:g}°" + (" slowly" if s.get("speed") == "slow" else ""),
            "gripper": lambda: f"gripper {s['state']}",
            "touch": lambda: f"move {s['joint']} {'+' if s['degrees'] > 0 else '-'} until touching (max {abs(s['degrees']):g}°)",
            "goal": lambda: f"look-and-move loop: {s['goal']}",
            "replay": lambda: (("draw " if s.get("pen") else "replay ") + (f"'{s['name']}'" if s.get("name") else "the recording")
                               + (" slowly" if s.get("speed") == "slow" else ""))}[s["do"]]()


class App:
    def __init__(self, io, sim: bool):
        self.io, self.sim = io, sim
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.task_is_local = False

    async def setup(self) -> None:
        await self.io.connect()
        self.arm = Arm()
        self.arm.update_ranges(await self.io.read_joints())
        self.skills = Skills(self.io, self.arm)
        self.scene = load_yaml("scene.yaml")
        # offer Claude only what this arm can do right now (taught poses belong to the follower)
        absolute = self.skills.absolute_ok
        gestures = GESTURES if absolute else [g for g in GESTURES if g not in ("home", "ready")]
        slots = sorted(self.skills.taught_slots) if absolute else []
        self.brain = Brain(self.arm, slots, timeout_s=TIMEOUT_S, gestures=gestures)
        replay_joints = [n for n in self.arm.joints if absolute or n != "gripper"]   # leader: trigger, skip it
        self.recorder = Recorder(self.io, self.arm, replay_joints)
        self.executor = Executor(self.io, self.arm, self.skills, self.recorder)
        self.goal = GoalRunner(self, out)
        self.views, self.views_version = [], 0   # camera views goal mode last sent to Claude

    def on_line(self, text: str) -> None:
        if not text.strip():
            return
        if text.strip().lower().rstrip("!. ") in STOP_WORDS:     # never waits for Claude
            t = time.monotonic()
            self.executor.log.setdefault("t_stop", t)
            self.executor.stop()
            self.goal.stop()
            if self.task and not self.task.done() and not self.task_is_local:
                self.task.cancel()   # a local replay ends by itself once the motion aborts
            out(f"■ Stopped ({(time.monotonic() - t) * 1000:.0f} ms to send).", flush=True)
            return
        if self.task and not self.task.done():
            out("  busy: wait for this request, or type stop", flush=True)
            return
        low = text.strip().lower().rstrip("!. ")
        METER.new_request()
        if low.startswith("goal:") and text.split(":", 1)[1].strip():
            # look-act loop; ends on its own or on stop (not cancelled: stop aborts the motion cleanly)
            self.task, self.task_is_local = asyncio.create_task(self.run_goal(text.split(":", 1)[1].strip())), True
            return
        named = split_named(low)
        if (low in RECORD_WORDS | DONE_WORDS | RELAX_WORDS
                or low.startswith(REPEAT_PREFIXES)
                or (named and named[0] == "draw" and named[1] in self.recorder.names())
                or low.startswith(("record ", "save as ", "name it "))
                or low in TELEOP_WORDS):
            self.task, self.task_is_local = asyncio.create_task(self.local(low)), True
            return
        self.task, self.task_is_local = asyncio.create_task(self.handle(text.strip())), False

    async def live_rgbd(self) -> tuple[bytes | None, tuple[int, int] | None]:
        """Live RealSense colour frame, or (None, None) when the depth camera isn't running."""
        if self.sim:
            return None, None
        try:
            views = await asyncio.wait_for(asyncio.to_thread(self.goal.cams._rgbd), 2.0)
            return views[0].jpeg, views[0].size
        except Exception:
            return None, None

    async def run_goal(self, goal: str) -> None:
        try:
            end = await self.goal.run(goal)
            self.history = (self.history + [{"instruction": f"(look-and-move loop) {goal}", "steps": [],
                                             "say": "", "outcome": end}])[-3:]
        except Exception as e:   # Station error etc.: hold still
            self.executor.stop()
            await self.io.hold()
            out(f"■ Goal ended, holding still: {type(e).__name__}: {e}", flush=True)

    fmt_step = staticmethod(lambda s: fmt_step(s))

    async def local(self, low: str) -> None:
        """record / done / relax / repeat: no Claude call."""
        try:
            if low in RECORD_WORDS:
                out("  " + await self.recorder.start(), flush=True)
            elif low.startswith("record "):
                out("  " + await self.recorder.start(low[len("record "):]), flush=True)
            elif low in TELEOP_WORDS:
                if self.recorder.recording:
                    out("  still recording: type 'done' first", flush=True)
                elif TELEOP_WORDS[low]:
                    out("  … moving the follower to the leader's pose, then starting teleop", flush=True)
                    out("  " + await self.io.teleop_on(self.arm), flush=True)
                else:
                    out("  " + await self.io.teleop_off(), flush=True)
            elif low.startswith(("save as ", "name it ")):
                out("  " + self.recorder.save_as(low.split(" ", 2)[2]), flush=True)
            elif low in DONE_WORDS:
                out("  " + await self.recorder.finish(), flush=True)
            elif low in RELAX_WORDS:
                self.executor.stop()
                out(f"  {await self.io.relax()} (arm is limp)", flush=True)
            else:
                verb, name = split_named(low)
                if name not in self.recorder.names():   # "repeat the same movement" = the last recording
                    name = ""
                await self.replay_now(low, speed_of(low), name or None, pen=(verb == "draw"))
        except asyncio.CancelledError:
            out("  stopped", flush=True)
        except Exception as e:   # Station error etc.: never leave a move half-sent
            self.executor.stop()
            await self.io.hold()
            out(f"■ Holding still: {type(e).__name__}: {e}", flush=True)

    async def replay_now(self, text: str, speed: float, name: str | None = None, pen: bool = False) -> None:
        log = {"t_instruction": time.monotonic(), "instruction": text, "io": self.io.name, "local": True}
        self.executor.begin(log)
        if self.recorder.recording:
            out("  still recording: type 'done' first", flush=True)
            return
        what = f"'{name}'" if name else "the last recording"
        out(f"  → {'draw' if pen else 'replay'} {what} at {speed:g}× the recorded speed"
            + (" (pen up to the start, lower, draw, lift)" if pen else ""), flush=True)
        log["t_first_move"] = time.monotonic()
        r = await self.recorder.replay(speed, name, pen)
        r["step"], r["t"] = {"do": "replay", "speed_scale": speed, "name": name, "pen": pen}, time.monotonic()
        log["results"].append(r)
        if r["reached"]:
            out(f"    ✓ done in {r.get('elapsed_s')} s" + (f"; {r['clamps'][0]}" if r["clamps"] else ""), flush=True)
        else:
            out(f"    ✗ not finished ({r.get('abort_reason') or 'off target ' + str(r.get('errors'))})", flush=True)
        if r.get("peak_current"):
            out("    peak current: " + ", ".join(f"{j} {v}" for j, v in r["peak_current"].items()), flush=True)
        self.write_log(log)

    async def handle(self, text: str) -> None:
        log = {"t_instruction": time.monotonic(), "instruction": text, "io": self.io.name}
        self.executor.begin(log)
        result = {"steps": [], "say": ""}
        try:
            jpeg, size = await self.live_rgbd()
            if jpeg:   # the fixed RealSense beside the robot: a live view of the whole table
                image, log["camera"] = jpeg, "rgbd"
            else:
                jpeg, age = await self.io.read_frame(self.scene.get("camera_role", "desk"))
                if age > float(self.scene.get("max_frame_age_s", 2.0)):
                    raise StationError(f"camera frame is {age:.1f} s old")
                image, size = perceive.prepare(jpeg, self.scene)
                log["camera"] = "desk photo" if getattr(self.io, "photo_desk", False) else "desk"
            joints = await self.io.read_joints()
            self.arm.update_ranges(joints)
            context = describe_state(self.arm, {n: m["position_norm"] for n, m in joints.items()}, self.history,
                                     self.recorder.names())
            log.update(t_frame=time.monotonic(), image_size=size, image_bytes=len(image))
            result = await asyncio.wait_for(
                self.brain.decide(image, text, context, self.executor.submit, log), llm.current().timeout_s + 2)
            await self.executor.drain()
        except asyncio.CancelledError:
            log["error"] = "stopped by user"
            try:   # let the interrupted step report how it ended, so the stop delay is logged
                await asyncio.wait_for(self.executor.drain(), 2.0)
            except asyncio.TimeoutError:
                pass
        except Exception as e:   # anything else (Station restarting, a camera gone, ...): say it and hold still
            reason = "that took too long" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}: {e}"
            self.executor.stop()
            try:
                await self.io.hold()
            except Exception:   # Station itself may be what failed; the motion is already stopped
                pass
            log["error"] = reason
            out(f"■ Holding still: {reason}", flush=True)
        finally:
            log["steps"], log["say_claude"] = result.get("steps", []), result.get("say", "")
            log["say"], log["outcome"] = final_say(log["say_claude"], log.get("results", []), log.get("error"))
            if log["say"]:
                out(f"  “{log['say']}”", flush=True)
            self.history = (self.history + [{"instruction": text, "steps": log["steps"], "say": log["say"],
                                             "outcome": log["outcome"]}])[-3:]
            self.write_log(log)
            goal = self.executor.handoff
            if goal and not log.get("error"):   # start the loop as its own task, after this one ends
                asyncio.get_running_loop().call_soon(self.start_goal, goal)

    def start_goal(self, goal: str) -> None:
        self.task, self.task_is_local = asyncio.create_task(self.run_goal(goal)), True

    def write_log(self, log: dict) -> None:
        t0 = log["t_instruction"]
        ms = {k: round((v - t0) * 1000) for k, v in log.items() if k.startswith("t_") and isinstance(v, float)}
        for r in log.get("results", []):
            if "t" in r:
                r["t"] = round((r["t"] - t0) * 1000)
        rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: v for k, v in log.items() if not k.startswith("t_")},
               "ms": ms}
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        parts = [f"{k[2:]} {v / 1000:.1f}s" for k, v in ms.items() if k != "t_instruction"]
        if log.get("usage") and "usd" in log:
            parts.append(fmt_usage(log["usage"], log["usd"]))
        out(f"  ⏱ {' · '.join(parts)}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="no hardware: saved photo + simulated arm")
    ap.add_argument("--frame", default=str(ROOT / "bench" / "frames" / "sample_desk.jpg"))
    ap.add_argument("--photo", action="store_true", help="real arm, but use --frame instead of the camera")
    ap.add_argument("--web", action="store_true", help="also serve the web panel (Station UI + command box)")
    ap.add_argument("--web-host", default="127.0.0.1", help="0.0.0.0 to open it from other machines")
    ap.add_argument("--web-port", type=int, default=8890)
    a = ap.parse_args()
    io = SimIO(Path(a.frame), load_yaml("poses.yaml")["home"]) if a.sim else RealIO()
    if a.photo and not a.sim:
        photo = Path(a.frame).read_bytes()

        live_frame = io.read_frame

        async def saved_frame(role: str = "desk") -> tuple[bytes, float]:
            if role == "desk":   # only the desk camera is replaced; the gripper camera stays live
                return photo, 0.0
            return await live_frame(role)
        io.read_frame = saved_frame
        io.photo_desk = True
    import logging
    logging.getLogger("normacore-mcp").setLevel(logging.WARNING)   # hide "Read stream N started" chatter
    for name in ("httpx", "httpx2", "httpcore", "openai", "anthropic"):   # no line per API request
        logging.getLogger(name).setLevel(logging.WARNING)
    app = App(io, a.sim)
    try:
        await app.setup()
    except StationError as e:
        sys.exit(f"Can't start: {e}")
    out(f"Ready ({io.name}). Type a request, 'stop' to halt, Ctrl-D to quit.", flush=True)

    loop = asyncio.get_running_loop()
    lines: asyncio.Queue = asyncio.Queue()

    def reader():  # stdin on its own thread, so 'stop' is read even while a request runs
        for line in sys.stdin:
            loop.call_soon_threadsafe(lines.put_nowait, line)
        loop.call_soon_threadsafe(lines.put_nowait, None)

    threading.Thread(target=reader, daemon=True).start()
    if a.web:
        sys.modules["app"] = sys.modules[__name__]   # web's "import app" must share this module's output list
        import web
        asyncio.create_task(web.serve(app, a.web_host, a.web_port))
        out(f"Web panel: http://{'localhost' if a.web_host in ('127.0.0.1', '0.0.0.0') else a.web_host}:{a.web_port}",
            flush=True)
    while (line := await lines.get()) is not None:
        app.on_line(line)
    if a.web:   # no terminal input (Ctrl-D or run in the background): keep serving the panel
        out("Terminal input closed; the web panel keeps running (Ctrl-C to quit).", flush=True)
        await asyncio.Event().wait()
    if app.task and not app.task.done():
        await app.task


if __name__ == "__main__":
    asyncio.run(main())
