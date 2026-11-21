# CLAUDE.md: normacore-mcp

## What this is
Hackathon project for the **NormaCore challenge: "AI-powered robot control: integrate the
Station API into Codex or Claude so it can control the robot by looking through the cameras."**

`server.py` is an MCP server (Python, `mcp<2` FastMCP API). It connects to a running
NormaCore **Station** over TCP :8888 and exposes the tools `look`, `list_cameras`,
`get_robot_state`, `move_joints`, `nudge_joints`, `set_gripper`, `hold` and `relax`.
`README.md` has the setup steps for people. This file is context for agents.

**Your role:** you are the robot's operator *and* its developer. Plan with skills, verify on camera, learn new skills
the moment a human demonstrates one, and keep improving skills (data) and tools (code). Follow the
**Learning protocol** and **Roadmap** sections below, and record what you discover under **Learned facts**.

Upstream: https://github.com/norma-core/norma-core (MIT). We import its pure-python
client (`software/station/shared/station_py`) and the generated protobuf readers/writers
(`target/gen_python/protobuf/...`, which import `shared.gremlin_py.gremlin`). The checkout
path comes from `NORMA_CORE_REPO`, which must be the repo root because those imports are rooted there.

## Hardware we have (from the hackathon table)
- **ElRobot** 7+1 DoF 3D-printed arms (`hardware/elrobot` upstream), as a **leader/follower pair**:
  - **Leader**: handle + trigger instead of a gripper, ST3215 **7.4 V** low-ratio servos (back-drivable), **5 V PSU**.
  - **Follower**: parallel gripper + gripper camera (UVC, e.g. Innomaker U20CAM), ST3215 **12 V / 30 kg** servos, **12 V PSU**.
  - Never put 12 V on the leader.
- 2× serial bus servo driver boards (Waveshare: jumper must be on **B** = USB) in printed enclosures.
- Powered 7-port USB 3.0 hub (needed for cameras), USB-C data cables.
- Desk/table camera: any UVC webcam, fixed position looking at the workspace and the gripper.
- MacBook Air (Apple Silicon) runs Station. **Station has no Windows build** (macOS/Linux only).

## Motor IDs (ElRobot)
| id | joint (our naming) |
|---|---|
| 1 | base_yaw |
| 2 | shoulder |
| 3 | upper_arm_roll |
| 4 | elbow |
| 5 | forearm_roll |
| 6 | wrist_pitch |
| 7 | wrist_roll |
| 8 | gripper (leader: trigger) |

SO-101 arms use ids 1–6 with 6 = gripper (`GRIPPER_MOTOR_ID=6`). The joint names above are
our own labels, so check them on the real arm.

### Joint direction map (FILL THIS IN on the real robot)
Measure with `nudge_joints` ±0.05 plus `look`, then record it here so future runs know it:
| joint | +norm moves… | safe range notes |
|---|---|---|
| base_yaw | ? | |
| shoulder | ? | |
| upper_arm_roll | ? | |
| elbow | ? | |
| forearm_roll | ? | |
| wrist_pitch | ? | |
| wrist_roll | ? | |
| gripper | ? (open or close?) | |

## Station API facts (learned from the upstream source)
- Run: `station --web --tcp` → web UI on :8889, normfs TCP API on :8888. Config is `station.yaml`.
- Client: `station_py.new_station_client("host[:8888]", logger)` (must be created inside a running
  asyncio loop; it spawns background tasks). `client.read_from_tail(queue, offset=bytes([n]), limit=n, step=1, buf_size=…)`
  returns `QueueRead` whose `.data` asyncio.Queue yields `StreamEntry` objects and then `None`. `client.follow(queue, q)` streams live entries.
  `send_commands(client, [DriverCommand…])` enqueues a `StationCommandsPack` to the `commands` queue.
- **Queues**
  - `st3215/inference` → `st3215.InferenceState`: buses (by `bus.serial_number`) → motors with
    `id`, raw register dump `state` (bytes), `range_min/range_max` (from calibration; equal = uncalibrated), `error`.
  - `inference-states` → `inference.InferenceRx`: index of the latest entry pointer per queue, with
    `QueueDataType`. Camera queues are the entries with `QDT_USB_VIDEO_FRAMES`.
  - `usbvideo/<md5(camera unique_id)>` → `usbvideo.RxEnvelope`. `ET_FRAMES` envelopes hold a `FramesPack`
    whose `frames_data` are **JPEG**, resized so the shortest side = `usb-video.resize_target` (default 224).
    Other envelope types: device connected/disconnected/recording start/end/error.
  - `inference/normvla` → `normvla.Frame`: joints (sorted by motor id, with `position_norm`) + 224×224 JPEGs at ~10 Hz.
    This is what SmolVLA uses. We don't depend on it, because it can skip frames and is ambiguous with 2 buses.
- **ST3215 commands**: `st3215.Command(target_bus_serial=…, sync_write=ST3215SyncWriteCommand(address, motors=[MotorWrite(motor_id, value_bytes)]))`
  wrapped in `commands.DriverCommand(type=StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())`.
  Also available: `write`, `reg_write`, `action`, `reset`, calibration commands, `auto_calibrate`.
- **Registers** (little-endian): torque enable `0x28` (u8), acceleration `0x29` (u8), goal position `0x2A` (u16),
  goal speed `0x2E` (u16), present position `0x38` (u16, bit 15 = sign; normalize to 0–4095),
  status `0x40` (u8), present current `0x45` (u16). Send speed/accel **before** goal position.
- Normalized position = `(raw - range_min) / (range_max - range_min)`, which is the same convention as normvla `position_norm`.
- Web UI: motor-ID change page (set IDs one servo at a time) and **Auto Calibrate → Save** per bus
  (disabled below 7.0 V bus voltage).
- Upstream examples worth copying from: `software/station/examples/st3215-remote-teleop-py` (leader→follower
  mirroring, bus resolution, sync-write batching) and `software/ai/smolvla_py/scripts/run_policy.py` (normvla → policy → goal writes).

## Safety rules for agents driving the arm
1. Always `look` + `get_robot_state` before moving. After every move, `look` again.
2. Only ever command the **follower** bus (`FOLLOWER_BUS`). Never write to the leader.
3. Small steps: `nudge_joints` ≤ 0.05 when near objects/table; the server also clamps each call to `MAX_STEP_NORM` (0.15).
4. Don't move uncalibrated joints or joints reporting `error_status != 0`.
5. `relax` = torque off, and the arm falls. Use `hold` to freeze in place instead, unless something is jammed/overloaded.
6. If leader→follower teleop is running, stop it first: two writers fight over the follower.

## Dev notes
- Offline test without hardware: `python tests/offline_test.py . <norma-core path>` (monkeypatches queue reads/sends with simulated frames).
- `mcp` is pinned `<2`: mcp 2.x renamed FastMCP → `mcp.server.mcpserver.MCPServer`.
- stdout is the MCP channel, so log to stderr only.
- **Code changes to the MCP server only take effect after it restarts** (`/mcp` → reconnect `normacore`, or restart `claude`).
  Data files under `data/` are re-read on every tool call, so skill/pose edits apply immediately with no restart.
  That's why skills live in data, not code.

---

# Strategy (why we win)
Other teams will likely have Claude guess raw joint angles from images, which fails on stage. We keep
geometry out of the LLM: **humans teach skills by hand with the leader arm, Claude learns them instantly,
plans with them in plain English, verifies every step on camera, and keeps improving them.**
Pitch: *"Show it once by hand, then just tell it what to do. It watches, acts, checks itself and gets better."*

# Roadmap (update the Status column as work lands)
| P | Component | File(s) | What | Status |
|---|---|---|---|---|
| P0 | Station I/O | `station_io.py` | split connection/state/camera/command code out of `server.py` | todo |
| P0 | Motion core | `motion.py` | `execute_trajectory(targets, duration)`: ~20 Hz interpolation, per-joint velocity cap, abort+`hold` on servo error / current spike / soft-limit breach (`data/limits.json`); `gripper_status()` grasp detection (gripper stalled early + current high = holding) | todo |
| P0 | Poses | `poses.py`, `data/poses.json` | `save_pose`, `list_poses`, `go_to_pose` | todo |
| P0 | Demonstration capture | `learn.py` | `record_demo` / `skill_from_history` (see Learning protocol) | todo |
| P1 | Skills | `data/skills/*.json` | `list_skills`, `run_skill`, `edit_skill`; run log in `data/runs.jsonl` | todo |
| P1 | Vision | `vision.py` | `look(grid=True)` labelled A1–C3 grid on the desk cam; `find_objects(color)` HSV blobs → pixel + cell | todo |
| P2 | Pick anywhere | `poses.py` | taught `above_<cell>` poses for 9 cells, bilinear blend → `pick(cell)` / `place(cell)` | todo |
| P2 | E-stop | `estop.py` | separate process + own station connection: Space = hold, Esc = relax | todo |
| P3 | Demo agent/UI | `agent.py`, `ui/` | Claude API agent with the same tools + page with camera feeds, reasoning stream, STOP button | stretch |
| P3 | SmolVLA | upstream `software/ai/smolvla_py` | train a policy on recorded demos for one fast skill; Claude stays the planner | stretch |

Final tool surface: perception (`look`, `find_objects`, `gripper_status`, `get_robot_state`), motion (`go_to_pose`,
`nudge_joints`, `grasp`, `release`), skills (`list_skills`, `run_skill`, `edit_skill`, `pick`, `place`), learning
(`record_demo`, `skill_from_history`, `save_pose`), safety (`hold`, `relax`).
**If a tool you need for the protocol below doesn't exist yet, build it (roadmap order) rather than improvising with raw joint moves.**

---

# Learning protocol: how Claude learns and improves skills
"Skill" below means a **robot skill** (a JSON file in `data/skills/`), not a Claude Code skill.

## 1. When a human shows you something → make a skill right away
Teaching setup: upstream leader→follower teleop is running, so the follower copies the leader arm.
Triggers: "watch this", "learn this", "save what I just did as X", "remember this".
1. **Capture**: either `record_demo(start)` ... `record_demo(stop)`, or after the fact with
   `skill_from_history(seconds=N)`. Station keeps full history, so read the last N seconds of `st3215/inference`
   (`read_from_tail` with an entry count ≈ N × frame rate; measure the rate, it's ~10–20 Hz). **Use the follower bus
   positions** so replays are exact (no leader/follower calibration mismatch).
2. **Segment into keyframes**: endpoints, points where all joints are still for ≥0.3 s, and every gripper
   open/close change. Drop near-duplicates. Typical result: 3–8 keyframes.
3. **Understand it**: grab camera frames at the first and last keyframe (and at grasp moments). Camera queues
   have history too, so match frames by `monotonic_stamp_ns`. Describe in one sentence what the skill does and its
   preconditions (e.g. "cube must be in cell B2, gripper empty").
4. **Save** `data/skills/<name>.json` immediately (format below), status `"new"`.
5. **Confirm by replay**: tell the human you're about to replay. Run it at **50% speed** while they watch, look
   before and after, and report the result. If it works, set status `"ok"`. Teleop must be stopped before replay.
6. Commit the new skill to git (`learn: add skill <name>`).

## 2. Every time you run a skill → log and learn
- Before: `look`, check the preconditions. After: `look` + `gripper_status`, judge success from the images.
- Append one line to `data/runs.jsonl`: `{ts, skill, version, params, success, failure_reason, notes}`.
- Update `stats` in the skill file.

## 3. When a skill fails or is sloppy → adjust the skill (data first)
Diagnose from the camera frames, then make **one change at a time**:
- missed the object → shift the approach keyframe a little (≤0.03 norm per joint per iteration) toward what the camera shows;
- knocked the object → raise the approach keyframe / slow the descent segment;
- weak grasp → close the gripper further or add a dwell;
- too slow/jerky → adjust segment durations.
Bump `version`, add a `changelog` entry ("v3: raised approach by 0.02 on shoulder, missed cube by ~1 cm"), keep
`last_good` (the previous working keyframes) so you can roll back, re-test at 50% speed, then commit
(`learn: <skill> v3 - <why>`). After 3 failed fixes in a row, stop and ask the human to re-demonstrate.

## 4. When the *code* is the limit → adjust the code
If the same kind of failure repeats across skills (bad keyframe detection, missing tool, vision misses
objects), improve the code (`learn.py`, `vision.py`, `motion.py`, ...). Then: run `tests/offline_test.py`
(add a test for the change), commit, and tell the human the MCP server needs a reconnect (`/mcp`).
Record what you learned in the **Learned facts** section below.

## 5. Guardrails for self-modification (never broken without explicit human OK)
- Never loosen safety: `MAX_STEP_NORM`, velocity caps, `data/limits.json`, current-abort thresholds,
  the leader-bus ban, the `relax`/`hold` behaviour.
- Never replay a new or changed skill at full speed before one successful 50%-speed run with a human watching.
- Never run a skill while teleop is running.
- Every learning change is a separate git commit, so anything can be reverted.

## Skill file format (`data/skills/<name>.json`)
```json
{
  "name": "pick_cube_b2",
  "description": "Pick the cube from cell B2 with a top-down grasp",
  "preconditions": ["cube visible in B2", "gripper empty"],
  "effects": ["cube in gripper"],
  "bus": "<follower serial>",
  "keyframes": [
    {"joints": {"1": 0.50, "2": 0.31, "3": 0.50, "4": 0.72, "5": 0.50, "6": 0.40, "7": 0.50, "8": 0.90}, "duration_s": 1.5, "label": "above B2"},
    {"joints": {"2": 0.36, "4": 0.66}, "duration_s": 1.0, "label": "down"},
    {"gripper": "close", "label": "grasp"},
    {"joints": {"2": 0.31, "4": 0.72}, "duration_s": 1.0, "label": "lift"}
  ],
  "speed_scale": 1.0,
  "status": "new | ok | flaky | broken",
  "version": 1,
  "last_good": null,
  "stats": {"runs": 0, "success": 0},
  "changelog": [{"version": 1, "note": "learned from demo at <ts>"}],
  "source": {"type": "demo", "station_time_range_ns": [0, 0]}
}
```
Keyframes may list only the joints that change; the others keep their current value. Composite skills may
use `{"skill": "<other skill>"}` steps.

---

# Learned facts (Claude appends here as it learns; newest first)
<!-- format: - YYYY-MM-DD: fact (evidence) -->
- 2026-11-21: Station has no Windows build; run it on the MacBook (README).
