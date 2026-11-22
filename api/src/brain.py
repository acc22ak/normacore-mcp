"""The only module that calls Claude. One streamed request per instruction:
[static system prompt] + [image (cache breakpoint)] + [joint state, recent turns, instruction].
Structured output keeps the answer to a short list of allowed steps plus one sentence; each
step is handed to the executor as soon as its JSON object closes in the stream.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

import llm
from arm import Arm
from usage import METER


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


GESTURES = ["nod", "shake", "shrug", "wave", "home", "ready"]
GESTURE_TEXT = {"nod": "nod (yes)", "shake": "shake (no)", "shrug": "shrug (can't / don't know)",
                "wave": "wave", "home": "home (rest pose)", "ready": "ready (look across the table)"}


def build_schema(arm: Arm, slots: list[str], has_gripper: bool = True, gestures: list[str] = GESTURES) -> dict:
    steps = [
        _obj({"do": {"type": "string", "enum": ["gesture"]},
              "name": {"type": "string", "enum": list(gestures)}}),
        _obj({"do": {"type": "string", "enum": ["jog"]},
              "joint": {"type": "string", "enum": arm.jog_joints},
              "degrees": {"type": "integer"},
              "speed": {"type": "string", "enum": ["slow", "normal"]}}),
        _obj({"do": {"type": "string", "enum": ["reach"]},
              "direction": {"type": "string", "enum": ["forward", "back"]},
              "degrees": {"type": "integer"},
              "speed": {"type": "string", "enum": ["slow", "normal"]}}),
    ]
    steps.append(_obj({"do": {"type": "string", "enum": ["touch"]},
                       "joint": {"type": "string", "enum": arm.jog_joints},
                       "degrees": {"type": "integer"},
                       "speed": {"type": "string", "enum": ["slow", "normal"]}}))
    steps.append(_obj({"do": {"type": "string", "enum": ["goal"]}, "goal": {"type": "string"}}))
    steps.append(_obj({"do": {"type": "string", "enum": ["replay"]},
                       "speed": {"type": "string", "enum": ["slow", "normal"]},
                       "name": {"type": "string"}, "pen": {"type": "boolean"}}))
    if slots:
        steps.append(_obj({"do": {"type": "string", "enum": ["point"]},
                           "target": {"type": "string", "enum": slots}}))
    if has_gripper:
        steps.append(_obj({"do": {"type": "string", "enum": ["gripper"]},
                           "state": {"type": "string", "enum": ["open", "close"]}}))
    # "steps" comes before "say" so the steps stream first
    return _obj({"steps": {"type": "array", "items": {"anyOf": steps}}, "say": {"type": "string"}})


def build_system(arm: Arm, slots: list[str], has_gripper: bool = True, gestures: list[str] = GESTURES) -> str:
    rows = []
    for name in arm.jog_joints:
        lo, hi = arm.limits(name)
        rows.append(f"- {name}: +degrees {arm.joints[name].plus}. Allowed range "
                    f"{arm.norm_to_deg(name, lo):.0f} to {arm.norm_to_deg(name, hi):.0f} degrees.")
    point = (f"- point: point at labelled region {', '.join(slots)} in the image\n" if slots else
             "- (pointing is not available yet: no regions are set up)\n")
    grip = "- gripper: open or close the hand\n" if has_gripper else ""
    return (
        "You control a small desktop robot arm (7 joints plus a gripper) for a live demo. "
        "A person types a request; you turn it into at most 3 steps from this list:\n"
        f"- gesture: {', '.join(GESTURE_TEXT[g] for g in gestures)}\n"
        "- jog: turn one joint by a whole number of degrees, at most 30 per step\n"
        "- reach: stretch the arm forward (shoulder leans out while the elbow opens, together) or pull "
        "it back, by up to 30 degrees per step\n"
        "- touch: move one joint in 1-degree steps until whatever the gripper holds (or the fingers) "
        "touches something, sensed from the motors, then stop pressing lightly. degrees = the most it may "
        "travel, its sign = the direction (elbow + lowers toward the desk). Use it for 'lower until it "
        "touches', 'put the pen on the paper', 'press down gently'.\n"
        "- goal: hand the request to the look-and-move loop, which keeps looking through the cameras "
        "and making small moves until it's done (find, approach, follow, point the gripper at, grasp, "
        "pick up, put down, push, any multi-step task). Write the goal as one clear sentence that keeps "
        "the person's own names for things and what they told you (e.g. 'grasp the motor, the black "
        "object on the table'). Use it alone, as the only step.\n"
        "- replay: repeat a movement the person demonstrated (by hand, or with the leader arm). name = one "
        "of the saved recordings listed with the request, or \"\" for the most recent one. pen = true when "
        "it is a drawing with the brush/pen the gripper holds ('draw a C'): the arm then lifts, moves to the "
        "start in the air, lowers, draws and lifts again, and never opens the gripper. If the person asks "
        "to draw something that isn't saved, say so and suggest recording it ('record <name>' then 'done').\n"
        f"{point}{grip}"
        "\nJoints you can jog, what positive degrees do, and where each may go:\n"
        + "\n".join(rows) +
        "\n\nThe image is from a fixed camera beside the table showing the arm and the table. The current joint angles and the last few exchanges are given "
        "with each request, so 'a bit more' or 'other way' refer to the previous step.\n"
        "Rules:\n"
        "- Answer yes/no questions about the table with nod or shake, based on the image.\n"
        "- Turn left/right uses base_yaw, raise/lower the arm uses shoulder (negative raises), "
        "look up/down uses wrist_pitch.\n"
        "- jog and reach take a speed: 'slow' when the person says slowly, gently or carefully, "
        "otherwise 'normal'.\n"
        "- Stretch / reach out / extend forward uses reach forward; pull in / retract uses reach back.\n"
        "- 'A little' is about 10 degrees, 'a lot' about 25. Never ask for more than 30 per step, and "
        "keep a joint inside its allowed range; if a request would go past it, go as far as allowed "
        "and say so.\n"
        "- Questions (what do you see, where is X, how are you) get an answer in 'say' and no steps, "
        "unless a gesture fits (nod/shake for yes/no about the table).\n"
        "- Anything that needs aiming at, reaching or grasping an object, or several looks and moves, "
        "uses goal. Simple moves that don't depend on where things are use the other steps.\n"
        "- When starting a goal, 'say' tells the person what you are about to try.\n"
        "- If the request can't be done with these steps, is unsafe, or is unclear, use a single "
        "shrug and say why.\n"
        "- 'say' is one short, friendly sentence spoken to the person."
    )


def describe_state(arm: Arm, joints: dict[str, float], history: list[dict], saved: list[str] = ()) -> str:
    angles = ", ".join(f"{n} {arm.norm_to_deg(n, joints[n]):.0f}°" for n in arm.jog_joints if n in joints)
    lines = [f"Current joint angles: {angles}."]
    if "gripper" in joints:
        lines.append(f"Gripper opening: {joints['gripper']:.2f} (about 0.01 = fingers touching, 0.65 = open).")
    lines.append("Saved recordings: " + (", ".join(saved) if saved else "none") + ".")
    if history:
        lines.append("Recent exchanges (oldest first):")
        for h in history[-3:]:
            lines.append(f"- User: {h['instruction']} -> steps {json.dumps(h['steps'])}, said: {h['say']!r}"
                         + (f" Outcome: {h['outcome']}" if h.get("outcome") else ""))
    return "\n".join(lines)


STEP_RE = re.compile(r'\{\s*"do"\s*:[^{}]*\}')   # each step is a flat {"do": ...}; nothing else matches


class Brain:
    def __init__(self, arm: Arm, slots: list[str], has_gripper: bool = True, timeout_s: float = 8.0,
                 gestures: list[str] = GESTURES):
        self.system = build_system(arm, slots, has_gripper, gestures)
        self.schema = build_schema(arm, slots, has_gripper, gestures)

    async def decide(self, jpeg: bytes, instruction: str, context: str,
                     on_step: Callable[[str], None], log: dict) -> dict:
        """Streams the answer from the selected model (Claude or GPT); calls on_step(raw_json) for
        each step as soon as it is complete. Returns {"steps": [...], "say": str}; on refusal or
        truncation, no steps and the reason."""
        buf, sent = "", 0
        log["t_sent"] = time.monotonic()

        def on_text(text: str) -> None:
            nonlocal buf, sent
            log.setdefault("t_first_token", time.monotonic())
            buf += text
            found = STEP_RE.findall(buf)
            for raw in found[sent:]:
                log.setdefault("t_first_step", time.monotonic())
                on_step(raw)
            sent = len(found)
        m = llm.current()
        ans = await llm.ask(self.system, [jpeg, f"{context}\n\nUser: {instruction}"], self.schema, on_text, m)
        log["t_done"] = time.monotonic()
        log["usage"], log["model"] = ans.usage, ans.model
        log["usd"] = METER.add("command", ans.model, ans.usage)
        if not ans.ok:
            return {"steps": [], "say": f"I'm holding still ({ans.stop})."}
        try:
            return json.loads(ans.text)
        except ValueError:
            return {"steps": [], "say": f"I'm holding still ({m.label} gave no readable answer)."}
