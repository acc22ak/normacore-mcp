# CLAUDE.md: normacore-mcp

## What this is
Hackathon project for the **NormaCore challenge: "AI-powered robot control: integrate the
Station API into Codex or Claude so it can control the robot by looking through the cameras."**

`server.py` is an MCP server (Python, `mcp<2` FastMCP API) that connects to a running NormaCore **Station** over
TCP :8888 and to the RealSense bridge (`realsense-bridge/`, ws :8890). It lets Claude see the table and move the
ElRobot follower arm. `README.md` is the overview; `docs/SETUP.md` has the setup steps for people. This file is context for agents.

**Your role:** you are the robot's operator *and* its developer. Priorities, in order:
**safety, then simplicity for the human, then speed.**

**One workflow.** Ordinary requests ("put the figure next to the pen") are done with the pick-and-place loop
below. Everything else in this file (joint-level tools, replayable skills, teach mode) is for calibration, recovery
or special cases, and each section says so. If two parts of this file ever disagree, this section wins: fix the other.

## The workflow: pick and place with the depth camera
How it works: objects are found in the depth camera's **color** image and placed with the calibrated table plane;
inverse kinematics moves the jaws; the camera checks the jaws before they close. Claude chooses *what* to do and
judges photos. Claude never guesses joint angles.

0. **Same task again?** `list_skills`: if a skill matches this task (by name/description) and its `scene` matches
   what `find_objects` shows now (every object within ~1 cm, the arm's measured accuracy), **run it instead of steps
   1-3**: `run_skill(name)` → `find_objects` to confirm → `report_skill_result(run_id, success)`. The replay is blind
   (no jaw check before closing), so only on a scene match; otherwise do the normal loop below.
   Otherwise, `start_attempt(task)` now, so the task can become a skill afterwards.
1. **Look:** `find_objects` with the arm at `home`. Each object has an id, position, height, `standing`, and
   `reach.how`: the call to use for it. `crowded_by` means another object is within a jaw's width: move that first.
   `reach.reachable: false` means ask the human to move it closer (the arm manages ~39 cm from the base).
2. **Pick:**
   - standing object in easy reach: `pick(object_id)`.
   - flat, small or far object, and after any miss: `pick(object_id, close=False)` → read the zoomed photo →
     `nudge_jaws(right_cm, toward_camera_cm)` until the object is between the jaw tips → `close_and_lift()`.
   - unsure about a plan: `pick(..., dry_run=True)` returns it without moving.
3. **Place:** `place_near(object_id)` (beside a pen's long side, or `end=0/1`) or `place(x, z)`.
4. **Finish:** `home`, then one `find_objects` to confirm the result, and say what happened in a line or two.
5. **Auto-skill (after a confirmed success only):** `save_attempt_as_skill(name, description, scene=...)` with the
   object positions from step 1 (`[{"color", "x", "z"}, ...]`). Name it `<object>_<from>_to_<to>` and say in one line
   that it's saved. Never save a failed or partly failed task. Next time the same task with the same scene starts at
   step 0, and the practice ladder speeds it up (2 successes in a row → one speed level up).
   Every `run_skill` reports `speedup_vs_original`: say it to the human ("done in 9 s, 4x faster than the first time").

**Rules (from the human, 2026-11-22; they override habits elsewhere in this file):**
- **After a miss, do not go home.** `lift_and_look` (raises the jaws, returns a fresh `find_objects`), then try again.
- **If a photo doesn't clearly show where the object is, `lift_and_look`.** With the jaws low the gripper hides the
  object and what's left of the picture misleads: guessing from it pushed a figure 8 cm across the table.
- **Always come at objects from above.** The tools do: `pick` approaches above the object's top, `nudge_jaws` goes
  up, across, down. Don't replace them with joint moves near an object.
- **`home` only at the end of a task** (~30 s each), or when the arm blocks the camera's view of the table.

**Reading a `pick` result:**
- `steps.plan.jaw_ends_clear_object_top_by_cm` must be positive; `jaw_angle_deg_below_horizontal` under 45 means a
  shallow, less reliable grasp (object far out).
- `steps.jaw_look[].source`: `camera` = the sideways jaw offset was measured live; `saved` = last saved value used;
  `none` = no correction (expect a miss).
- `steps.verify[]`: the check before closing. `seen: false` = the camera couldn't tell and the grasp went ahead
  unchecked. `failed_step: "verify"` = the jaws were off the object and it did not close.
- `model_error_cm` is encoder-vs-model only. It is **not** the distance to the object.
- Every `pick` / `place` / `close_and_lift` is appended to `data/runs.jsonl`; the frames used are kept in
  `data/frames/` (last 60). Debug misses from those, not from memory.

**What limits it today (measured 2026-11-22):** the arm lands within ~1 cm after corrections; that grabs a standing
figure but a figure lying flat needs the look-and-adjust steps. The depth camera sat at 25° above the table: from
there the gripper hides the object and positions along the view are ~2 cm uncertain per cm of height error.
**Mount it looking down at 50° or steeper**, then redo both calibrations below.

## Calibrations (redo when the depth camera or the arm base moves)
| What | How | Stored in |
|---|---|---|
| Joint ranges | Station web UI: Auto Calibrate → **Save** (human). Never recalibrate on your own. | Station |
| Camera pose + joint zeros | Move the arm through ~7 poses (incl. low/far), `arm_fit.py capture` each, `arm_fit.py fit <dir> --save`, then `arm_fit.py check <dir> <out>` and **look at the pictures** | `realsense-bridge/calibration.json` (`arm_fit`; the viewer adopts it) |
| Reach error | `calibrate_reach(side)` with that side of the table clear (~1.5 min) | `data/reach_map.json` |
| Sideways jaw offset | Measured by `pick` on its own whenever it sees the jaw tips | `data/jaw_offset.json` |

Frame: metres, base at origin, **+x forward (toward the depth camera), y up, +z = the robot's right = the left of
the depth camera's image**. The same frame as station-viewer's 3D view, which uses the same joint scales and zeros.

## Tool map
| Use | Tools |
|---|---|
| **Everyday** | `find_objects`, `pick`, `nudge_jaws`, `close_and_lift`, `lift_and_look`, `place`, `place_near`, `home` |
| Looking closer | `depth_point` (3D point under an image position), `tip_position`, `look` (Station's USB cameras: the gripper camera) |
| Calibration | `calibrate_reach`; `arm_fit.py` (CLI) |
| Safety | `hold` (freeze), `relax` (torque off: the arm **drops**; emergencies only), `stop_motion`, `estop.py` (separate program) |
| Low-level, for recovery and calibration only | `get_robot_state`, `move_joints`, `nudge_joints`, `set_gripper`, `go_to`, `play_motion` |
| Optional: replayable skills | `start_attempt`, `save_attempt_as_skill`, `list_skills`, `run_skill`, `report_skill_result`, `teach_mode` |
| Legacy, unused by the workflow | `find_markers`, `calibrate_gripper_marker`, `list_cameras`, `set_camera_role` (roles only matter for `look`) |

Upstream: https://github.com/norma-core/norma-core (MIT). We import its pure-python
client (`software/station/shared/station_py`) and the generated protobuf readers/writers
(`target/gen_python/protobuf/...`, which import `shared.gremlin_py.gremlin`). The checkout
path comes from `NORMA_CORE_REPO`, which must be the repo root because those imports are rooted there.

## Hardware we have (from the hackathon table)
- **ElRobot** 7+1 DoF 3D-printed arms (`hardware/elrobot` upstream), as a **leader/follower pair**:
  - **Leader**: handle + trigger instead of a gripper, ST3215 **7.4 V** low-ratio servos (back-drivable), **5 V PSU**.
  - **Follower**: parallel gripper + gripper camera (UVC, e.g. Innomaker U20CAM), ST3215 **12 V / 30 kg** servos, **12 V PSU**.
  - Never put 12 V on the leader.
  - The leader bus therefore reads ~5 V, and the web UI **disables Auto Calibrate below 7.0 V**. Calibrate the leader
    **manually**: Reset → move every joint by hand through its full range → Save (freeze). The follower (12 V) can use Auto Calibrate.
- 2× serial bus servo driver boards (Waveshare: jumper must be on **B** = USB) in printed enclosures.
- Powered 7-port USB 3.0 hub (needed for cameras), USB-C data cables.
- Table camera: an Intel RealSense D435 on the far side of the table, looking back at the arm (via `realsense-bridge/`,
  which needs `sudo` on macOS, so the human starts it). No separate desk webcam is in use.
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

### Joint direction map (measured 2026-11-21 on follower 5B8E115950; details in data/exploration.md)
Measure with `nudge_joints` ±0.05 plus `look`, then record it here so future runs know it:
| joint | +norm moves… | safe range notes |
|---|---|---|
| base_yaw | turns the arm to its left (~9° per 0.05) | 1024–3072 raw; candy is to the right of home |
| shoulder | leans the arm forward, gripper cam tilts down | home ≈ 0.01 (folded, near min) |
| upper_arm_roll | not measured yet | home ≈ 0.50 |
| elbow | **−** unfolds the forearm (gripper up), + folds it | home ≈ 0.94 (folded, near max); sags ~0.02 under gravity |
| forearm_roll | rolls the forearm, scene turns clockwise in the gripper cam | almost full-turn range, no real stops. **Keep within ±0.25 of 0.54**: gripper-cam cable loops round the wrist |
| wrist_pitch | points the gripper down | home ≈ 0.46 = gripper level |
| wrist_roll | rolls the gripper, scene turns clockwise in the gripper cam | almost full-turn range, no real stops. **Keep within ±0.25 of 0.54** (camera cable) |
| gripper | **+ opens**, − closes | |

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
- Normalized position: `server.py` `to_norm`/`from_norm` are ports of Station's mirroring math
  (`motors-mirroring/src/inference/normalize.rs`, `mirror.rs:96-98`). **Calibrated ranges can wrap past 4095→0**
  (`range_max < range_min`, range size `(4096 - min) + max`). A joint counts as calibrated when its range size is
  ≥ 100 steps (viewer's `MIN_CALIBRATED_RANGE`). Station's own mirroring adds a small `safety_margin` that we don't.

## Unverified on hardware (check these on the real robot, then move them to Learned facts)
- Engine limits are still untuned defaults: `MAX_VEL_NORM` 0.5, `TRACK_ERR_MAX` 0.10 × 5 ticks, `MOTION_ACCEL` 50
  (they ran on hardware all of 2026-11-22; raising them needs the human's OK, §5).
- The over-the-top `nudge_jaws`, `lift_and_look`, the check before closing in `pick`, and the color-based
  `find_objects` on **standing** objects have only run in simulation / on saved frames (written after the arm was
  unplugged on 2026-11-22). Watch the first runs.
- Whether pressing Esc in Claude Code sends an MCP cancel (the library would then cancel the call and the engine holds the arm;
  verified in mcp/shared/session.py, but the client side is unverified). Until checked, treat `relax`/power as the real e-stop.
- Web UI: motor-ID change page (set IDs one servo at a time) and **Auto Calibrate → Save** per bus
  (disabled below 7.0 V bus voltage, so the leader is calibrated manually: Reset → move joints by hand → Save).
- **Built-in teleop (motors-mirroring driver)**: Station mirrors leader → follower itself; no Python script needed.
  - Start: `DriverCommand(type=STC_MOTOR_MIRRORING_COMMAND (=1), body=motors_mirroring.Command(type=CT_START_MIRROR,
    source=MirroringBus(type=MBT_ST3215, unique_id=<leader serial>), targets=[MirroringBus(..., unique_id=<follower serial>)]))`.
  - Stop: `motors_mirroring.Command(type=CT_STOP_MIRROR, source=MirroringBus(unique_id=<follower serial>))`. The web UI
    sends the *follower* as `source` when stopping (`station-viewer/src/st3215/BusCard.tsx`).
  - State: queue `inference/mirroring` (`motors_mirroring.RxEnvelope` → `InferenceState.mirroring[]` = active source→targets,
    `modes[]` = per-bus LEADER/FOLLOWER); bus modes also in `motors_mirroring/modes`. `unique_id` = bus serial.
  - Web UI: the follower's bus card has the mirror-source selector.
- **No control arbitration yet**: if mirroring and our tools both write to the follower, last write wins (station-viewer
  ADR 0009 describes a backend authority lease, but it is *not implemented*). We must check mirroring state ourselves.
- **Current protection exists only in the mirroring path**: the current threshold (default 100) and per-motor overrides
  are read only by `motors-mirroring/src/config.rs:17-40` and used in `inference/mirror.rs` (blocks the overloading
  direction and holds). Verified: no other Rust code under `software/` reads it, so our direct ST3215 writes have
  **no** current protection. `motion.py` must do its own current/stall abort. Gripper squeezes can hit the threshold.
- **macOS app (.dmg)** runs its own station in the background on 8888/8889 and keeps `station.yaml` + `station_data/`
  in `~/Library/Application Support/`. Don't start a second station alongside it (port clash); if you switch to a CLI/source
  build, copy the calibrated config over.
- The WebSocket on `:8889/api` carries the same normfs protocol as TCP 8888 (8-byte LE length + protobuf), so a browser page
  can read queues directly. There is no REST API.
- Go client bug: `software/station/shared/station/streams.go` `SendCommands` sends commands without the
  `StationCommandsPack` wrapper, so they're silently dropped. Use the Python client.
- Source of several of these facts: `notes/norma-station.html` (Victor's architecture notes, in this repo).
- Upstream examples worth copying from: `software/station/examples/st3215-remote-teleop-py` (leader→follower
  mirroring, bus resolution, sync-write batching) and `software/ai/smolvla_py/scripts/run_policy.py` (normvla → policy → goal writes).

## Safety rules for agents driving the arm
1. Only ever command the **follower** bus (`FOLLOWER_BUS`). Never write to the leader.
2. `get_robot_state` at the start of a session: every joint calibrated, `range_frozen` true, no `error_status`.
   Don't move uncalibrated joints or joints reporting an error.
3. Keep objects and hands out of the arm's path; the human is the one near the power. `hold` freezes the arm in
   place. `relax` = torque off, and the arm falls: only when something is jammed or overloaded.
4. Near an object, use the jaw tools (`pick`, `nudge_jaws`, `close_and_lift`, `place`). Joint-level moves there are
   how things get pushed off the table.
5. If mirroring (teleop) targets the follower, don't move it: two writers fight over the arm and Station won't stop
   them. `server.py` enforces this (every motion tool refuses while the follower is a mirroring target; fails closed
   if `inference/mirroring` can't be read). Turn mirroring off with `teach_mode(on=False)` or in the web UI.
6. `teach_mode(on=True)` makes the follower jump to the leader's pose, so it refuses if any joint differs by more than
   `TEACH_MAX_POSE_DIFF` (0.2) unless `force=True`, and while the e-stop is latched.
7. Keep the Mac on AC power and close spare browser tabs showing the 3D view. On battery the machine throttles;
   motions then run slower (the engine's clock never jumps ahead), and planning takes many seconds.
8. A camera frame marked **STALE** by `look` is not the current scene: don't plan from it.

## Dev notes
- Offline test without hardware: `python tests/offline_test.py . <norma-core path>` (monkeypatches queue reads/sends with simulated frames).
- `mcp` is pinned `<2`: mcp 2.x renamed FastMCP → `mcp.server.mcpserver.MCPServer`.
- stdout is the MCP channel, so log to stderr only.
- **Code changes to the MCP server only take effect after it restarts**: with the default stdio transport, `/mcp` →
  reconnect `normacore` (or restart `claude`); with `MCP_TRANSPORT=streamable-http` (server runs as its own process on
  127.0.0.1:8790, logs visible), restart that process.
  Data files under `data/` are re-read on every tool call, so skill/pose edits apply immediately with no restart.
  That's why skills live in data, not code.

---

# Strategy
Geometry stays out of the LLM. Code does the measuring (color detection, table plane, inverse kinematics, jaw-tip
detection); Claude chooses what to do, reads photos, and decides when something looks wrong.
Pitch: *"Tell it what to move where. It finds the things, picks them from above, checks its own grip, and tries
again from where it is when it misses."*

# Status (2026-11-22)
| Part | State |
|---|---|
| Pick-and-place loop (the workflow above) | Works on hardware for standing objects within ~36 cm; flat or far objects need the look-and-adjust steps |
| Calibrations (camera pose, reach, jaw offset) | Done for the camera position of 2026-11-22; redo after moving the camera |
| Motion engine, e-stop, mirroring guard, Station reconnect | On hardware; engine limits untuned |
| Color-based `find_objects`, check before closing, `lift_and_look` | Written and tested offline only |
| Replayable skills + practice ladder, teach mode | Built and offline-tested; **never used on hardware**. Skills are now auto-saved after a successful task (workflow step 5) and replayed when the scene matches (step 0) |
| Next | Camera at ≥50° looking down; gripper camera in the check before closing; `pick` retrying by itself |

---

# Replayable skills (auto-saved after a success; replayed only when the scene matches)
Skills are created automatically after a successful pick-and-place task (workflow steps 0 and 5); this section
has the details. Use a replay only for a task that is repeated with objects on **fixed, marked spots**, where replaying a
recorded motion is faster than looking each time. It replays joint positions without looking, so it fails as soon as
anything is off its spot; the workflow at the top has no such requirement. Nothing below changes the rules at the top.
(Three sections further down apply to **all** work, not only skills: "Keep improving the code on your own",
"Evidence over assumptions" and "Guardrails for self-modification".)

## Learning protocol: how Claude learns and improves skills
"Skill" below means a **robot skill** (a JSON file in `data/skills/`), not a Claude Code skill.

### 1. Learning a task by yourself
No human demonstration needed: you solve the task once, slowly, and your own successful attempt becomes the skill.
1. `start_attempt(task)` marks the start (station time), so nothing before it is recorded.
2. **Solve it carefully**: look with both cameras, move with `go_to` / `move_joints` / `nudge_joints` / `set_gripper`.
   Think between moves: every pose where the arm **rests** becomes a candidate keyframe, so make deliberate moves
   (approach above the object, descend, close the gripper, lift, move, release, retreat) rather than wiggling.
3. **Verify** with `look` that the task really succeeded. If not, `start_attempt` again and retry; never save a failure.
4. `save_attempt_as_skill(name, description)` keeps only the rest poses, merges steps that lie on a straight line
   (gripper open/close moments always stay), and times the moves from `SKILL_BASE_VEL` (0.25 norm/s at speed 1.0).
   Tested offline: a 27 s step-by-step attempt became 4 poses, ~6 s at speed 1.0, ~3 s at 2.0.
5. Name skills `<what>_<from>_to_<to>` (e.g. `cube_a_to_b`); write preconditions into the description ("cube on A,
   gripper empty"). Commit the new skill to git (`learn: add skill <name>`).
6. First replay is at speed 0.5 (practice ladder). Reset the scene first: put the object back, or run the reverse skill.

**Optional: the human shows it with the leader arm** (when they ask, or when you've failed to learn it twice).
There is no recording step; teach mode *is* the attempt:
1. Ask the human to put the leader arm roughly in the follower's pose, then `teach_mode(on=True)`. Once the
   follower has settled, the attempt starts automatically.
2. The human shows the task once with the leader arm (the follower copies it).
3. `teach_mode(on=False)` marks the end (before the arm's brief sag), then `save_attempt_as_skill(name, description)`.
   For a demonstration it keeps the *shape of the path* (sampled every 0.1 s, near-straight stretches merged,
   `DEMO_PATH_TOL` 0.01), because humans don't pause between moves like you do. Skill `source.type` is `"demo"`.
4. From here it's the same as a self-learned skill: replay at 0.5, practice ladder, improve it with your own attempts.

## 2. Every time you run a skill → practice ladder (this is how it gets fast)
- Before: `look` once, check the preconditions (object on its marked spot?). Then `run_skill(name)`: the whole
  motion runs in one call, no thinking in between.
- After: `look` once, then **always** `report_skill_result(run_id, success, note)`. That updates the ladder and
  appends the run to `data/runs.jsonl`:
  **2 successes in a row → one speed level up** (0.5 → 0.75 → 1.0 → 1.25 → … → 2.0), **a failure → one level down**.
  An engine abort (collision, servo error, ...) already counts as a failure.
- The engine's `MAX_VEL_NORM` cap still applies: if a run reports `stretched_segments > 0`, the ladder is past what
  the cap allows; tell the human (raising the cap needs their OK, §5).

## Marked spots (the stage)
Fast skills replay recorded motion without looking in between, so they only work if things are where they were during
the learning attempt (calibration drift and gear play add up, see §6). The team marks: the arm base outline, the desk
camera position, corner guides for each object spot (A, B, ...) and a home landmark. Refer to spots by their letters,
and before every `run_skill` check on camera that the object sits on its spot; if not, ask the human to put it back
(or adjust the skill per §3) instead of running blind.

## 3. When a skill fails or is sloppy → adjust the skill (data first)
Diagnose from the camera frames, then make **one change at a time**:
- missed the object → shift the approach keyframe a little (≤0.03 norm per joint per iteration) toward what the camera shows;
- knocked the object → raise the approach keyframe / slow the descent segment;
- weak grasp → close the gripper further or add a dwell;
- too slow/jerky → adjust segment durations.
Bump `version`, add a `changelog` entry ("v3: raised approach by 0.02 on shoulder, missed cube by ~1 cm"), keep
`last_good` (the previous working keyframes) so you can roll back, re-test at 50% speed, then commit
(`learn: <skill> v3 - <why>`). After 3 failed fixes in a row, re-learn it from scratch (`start_attempt`, solve with the cameras, `save_attempt_as_skill(..., overwrite=True)`) and tell the human.

## 4. When the *code* is the limit → adjust the code
If the same kind of failure repeats across skills (bad keyframe detection, missing tool, vision misses
objects), improve the code (`learn.py`, `vision.py`, `motion.py`, ...). Then: run `tests/offline_test.py`
(add a test for the change), commit, and tell the human the MCP server needs a reconnect (`/mcp`).
Record what you learned in the **Learned facts** section below.

## 4b. Keep improving the code on your own (not only after failures)
When you notice something slow, clumsy, fragile or wrong in this repo's code or docs (a tool that needs too many
calls, a check that wastes time, a confusing message, an outdated statement, a missing test), improve it:
1. **Measure first.** Get evidence: timings from tool reports (`elapsed_s`, `late_ticks`, `ticks`), test output, run logs,
   or the norma-core source (cite file:line). No "probably faster": show the before number.
2. **Small change, with a test.** One improvement per commit; add or extend a test in `tests/` that proves it.
   Run `tests/test_motion.py` and `tests/offline_test.py`; both must pass.
3. **Show the after number** in the commit message, and add a line to **Learned facts**.
4. **Tell the human** what changed and that the MCP server needs a restart.
5. Speed-ups never come from loosening safety (§5): faster playback is allowed only through `speed_scale` and the
   practice rules, not by raising current/velocity/tracking limits without the human's OK.
6. Before editing, `git pull` and read teammates' new commits; don't overwrite their work, and flag problems you find in it.

## Evidence over assumptions (applies to everything above)
Back claims about Station, the servos or the robot with proof: norma-core source (file:line), a test, or a measurement on
the arm. Anything unproven is labelled **unverified** and goes in the "Unverified on hardware" list until checked.

## 5. Guardrails for self-modification (never broken without explicit human OK)
- Never loosen safety: `MAX_STEP_NORM`, `MAX_VEL_NORM`, `TRACK_ERR_MAX`, `CURRENT_LIMIT(S)`, `data/limits.json`,
  the leader-bus ban, the `relax`/`hold` behaviour.
- Never replay a new or changed skill at full speed before one successful 50%-speed run with a human watching.
- Never run a skill while teleop is running.
- Every learning change is a separate git commit, so anything can be reverted.

## 6. Calibration is never perfect: adapt to drift, don't trust the numbers blindly
Joint numbers are estimates. **The camera is the ground truth.** Expect every replay to land a little off,
and design for correcting it.

**Verified sources of error** (norma-core source):
- Calibration ranges are just the min/max positions seen while the joint was swept or moved by hand
  (`st3215/src/calibrate.rs` `calculate_arc`). A hand-moved leader calibration is only as good as how far each joint was pushed.
- **Until calibration is frozen (web UI Save), Station widens a joint's range every time it reaches a new position**
  (`st3215/src/port_meta.rs:407-423`, skipped once frozen at `:349`). The same physical pose then gets a different
  normalized value. `get_robot_state` reports `range_frozen` and a warning per bus: get unfrozen buses saved before learning skills.
- Re-calibrating changes `range_min/max`, so **every normalized value in saved skills shifts**. Raw encoder positions are
  the stabler reference: auto-calibration writes the servo Position Offset (EEPROM 0x1F) to 0
  (`auto_calibrate/calibrator.rs:423-424, 500`), so raw positions keep the same zero unless a servo horn is remounted or
  the offset is changed.

**Expected but unverified** (measure on the robot, then move to Learned facts): gear backlash, sag under gravity/load
(shoulder/elbow), servo warm-up, the desk camera or table being bumped, objects not exactly where they were during the demo.

**Rules:**
1. **Closed loop at the moments that matter.** Approach, pre-grasp, grasp and place are verified with `look` (gripper cam
   for fine alignment). If off, correct with small `nudge_joints` (≤0.02 per step), look again, and only then continue.
   Never assume a replayed keyframe landed where the demo was.
2. **Store calibration with every skill and pose.** Keep both `norm` and `raw` per joint, plus the bus's `range_min/max`
   and `range_frozen` at save time (`calibration` block in the skill format). Before running, compare with the current
   ranges. If any joint's range moved by more than 20 steps, replay from `raw` (if it's inside the current range),
   set status `"needs_check"`, re-verify at 50% speed, then update `norm` and the stored calibration.
3. **Measure tracking error.** After each segment, compare goal vs present position per joint. A consistent offset
   (e.g. shoulder always 0.01 low under load) becomes a per-joint compensation in `data/compensation.json` (each
   ≤0.03 norm, with evidence: run ids). It is learned data, not a calibration change.
4. **Fold repeated corrections into the skill.** If the same visual correction was needed in ≥2 of the last 3 runs,
   bake it into the keyframes as a new skill version (protocol §3), with the run ids as evidence in the changelog.
5. **Drift check.** At session start and after any failure that looks systematic: `home`, `find_objects`, and
   compare against the stored reference image/description of home and a fixed table landmark (e.g. a tape mark).
   If it moved, decide which drifted: the camera (landmark moved in image, arm didn't), the calibration (arm pose
   differs, landmark same) or the table. Tell the human what you found.
6. **Prefer forgiving skills**: approach from above, open the gripper wide, centre on the object with the gripper cam
   before closing, slow final 2 cm.
7. **Don't fix drift by loosening safety, and don't re-calibrate on your own.** Re-calibration sweeps the arm through
   its full range and invalidates normalized skill data: propose it to the human, with evidence, and let them run it.

## Skill file format (`data/skills/<name>.json`)
```json
{
  "name": "pick_cube_b2",
  "description": "Pick the cube from cell B2 with a top-down grasp",
  "preconditions": ["cube visible in B2", "gripper empty"],
  "effects": ["cube in gripper"],
  "bus": "<follower serial>",
  "calibration": {"captured_at": "<ts>", "frozen": true,
                  "ranges": {"1": [1012, 3050], "2": [3900, 200], "...": []}},
  "keyframes": [
    {"joints": {"1": 0.50, "2": 0.31, "3": 0.50, "4": 0.72, "5": 0.50, "6": 0.40, "7": 0.50, "8": 0.90},
     "raw":    {"1": 2031, "2": 23, "3": 2048, "4": 2400, "5": 2050, "6": 1700, "7": 2048, "8": 2900},
     "duration_s": 1.5, "label": "above B2", "verify": "gripper cam: cube centred below jaws"},
    {"joints": {"2": 0.36, "4": 0.66}, "duration_s": 1.0, "label": "down"},
    {"gripper": "close", "label": "grasp"},
    {"joints": {"2": 0.31, "4": 0.72}, "duration_s": 1.0, "label": "lift"}
  ],
  "speed_scale": 1.0,
  "status": "new | ok | flaky | broken | needs_check",
  "version": 1,
  "last_good": null,
  "stats": {"runs": 0, "success": 0},
  "changelog": [{"version": 1, "note": "learned from demo at <ts>"}],
  "source": {"type": "demo", "station_time_range_ns": [0, 0]}
}
```
Self-learned skills (from `save_attempt_as_skill`) store `keyframes` with `duration_s` + `raw`, a `start_pose`/`start_raw`
that `run_skill` glides to first, and `"practice": {"level", "streak", "speed_scale"}`. `run_skill` also still plays
older skill files with a dense `"motion": [{"t", "joints", "raw"}, ...]`.
Keyframes may list only the joints that change; the others keep their current value. `{"gripper": "open"|"close"}`
needs `GRIPPER_OPEN_NORM` (which end is open, unverified on our arm); `{"gripper": 0.3}` always works.
Composite `{"skill": "<other skill>"}` steps are planned but `run_skill` refuses them for now. `raw` and `calibration` are there so a skill survives re-calibration (§6 rule 2): `run_skill` replays from
`raw` (refusing positions outside today's range rather than clamping them), and after a reported success it stores the
re-based positions + calibration (old ones in `last_good`). A `speed_scale` override above the practice speed is
capped until the skill's status is `ok`;
`verify` says what the camera should show after that keyframe (§6 rule 1).

---

# Learned facts (Claude appends here as it learns; newest first)
- 2026-11-21: Station `read_from_tail` offsets count from 0 = newest entry (real Station: 3 entries, offset 0 → [e2], offset 1 → [e1, e2]). `read_last(n)` used offset n, so n=1 returned the second-newest entry (one write behind: harmless for st3215/inference at ~400 Hz, wrong for rarely-written queues like estop/state or inference/mirroring). Fixed to offset n-1; verified by tests/test_estop.py against a real Station.
- 2026-11-21: `st3215/inference` updates ~400 Hz per bus on hardware (~530 entries/s with 2 buses), not 10–20 Hz. Reading it 1:1 is ~1 ms/entry, so long history reads use the read `step` (server-side stride; verified: 75 s in 1.6 s with step 25).
- 2026-11-21: On Linux, Station's camera unique_id is `vid:pid:bus:devnum` (decimal), and devnum changes on every replug, so each replug makes a new `/<station>/usbvideo/<md5>` queue; old queues stay, ending with `ET_DEVICE_RECORDING_END` (not `ET_DEVICE_DISCONNECTED`). Linux also needs a udev rule giving the user rw on UVC USB devices, or Station logs `Failed to open device ... -3` (libusb access denied).
<!-- format: - YYYY-MM-DD: fact (evidence) -->
- 2026-11-22: Real picking session, 4 figures: 3 picked and put by the pen, 1 pushed out of reach. First-try success only on a standing figure in the centre with the live jaw look working. What made the others work: reach correction, an approach that clears the object's top, and adjusting by eye from a zoomed photo before closing (`data/runs.jsonl`).
- 2026-11-22: The real jaws land short along the reach direction: -2.5 cm at 27 cm, -3.0 cm at 32 cm (`calibrate_reach`, measured from the side where reach is across the camera's view; the 37 cm probe read -6.5 cm and was rejected as too large, the 22 cm probe misread the tips). `pick`/`place` aim further out by `data/reach_map.json`. The sideways jaw offset showed up again in those probes (-1.5 to -2.8 cm), so it is a property of the arm, not of the camera calibration.
- 2026-11-22: With the depth camera 25 deg above the table, a jaw tip's position along the camera's view moves ~2 cm per cm of unknown jaw height, and the gripper hides the object it is over. Photos taken with the jaws low were misread three times. Only the sideways error is trusted at that angle (`_jaw_error_at_object`); `lift_and_look` exists for the rest.
- 2026-11-22: A sideways jaw move with a 1.5 cm hop pushed a 4 cm figure and a box ~8 cm, to the edge of reach. Sideways moves now go up over the object's top, across, and down (`nudge_jaws`).
- 2026-11-22: On battery with a busy browser the motion loop's ticks arrived seconds late; it followed the wall clock, so the goal jumped ahead and the arm lunged into the object ("shoulder is 0.749 away from its target"). The motion clock now advances at most two periods per tick: a slow machine gives a slow motion. Grasp planning also froze the event loop for up to 6 s, which coincided with the Station connection going dead (reads empty until a new connection); planning now runs in a thread and `read_buses` reconnects once.
- 2026-11-22: A motor whose read fails arrives with `error` set and an empty register dump (`st3215/src/state.rs`: `state = envelope.data`); parsed naively that is position 0. The viewer drew the joint at the end of its range for a frame (the "glitch"), and the server would have fed it to the motion checks. Both now hold the last real reading (`arm-pose.ts`, `read_buses`).
- 2026-11-22: Color-image object detection (`depth.find_objects_rgb`: wide-median table model, shadow rejection, table-plane positions, depth only for height) finds the lying minifigure the depth-only detector missed and repeats positions within ~1 mm across arm poses on the saved frames (tests/test_depth.py). Not yet tried on standing objects or on the live camera.
- 2026-11-22: `place` asked the gripper to "release" to 0.45 while holding a wide figure at 0.56, i.e. narrower than the object: it didn't let go. Release now opens at least 0.2 wider than the grip.
- 2026-11-22: The arm model puts the point between the jaws ~2 cm to the side of where it really is: on the three saved calibration frames with the jaws pointing down (p2, p3_low, p6) the real jaw midpoint is at -22.6 / -18.5 / -20.7 mm along the jaw closing direction, i.e. the model's "centre" sits on one real jaw tip (`arm_fit.py check`; tests/test_depth.py). Cause not known (roll-joint zeros set by eye, or the gripper differs from the URDF): **unverified**. `pick` now measures it live (`depth.find_jaw_tips`) and `data/jaw_offset.json` holds the fallback. Only the sideways part (across the camera's view) is trusted: along the view two methods disagreed by 1-2 cm.
- 2026-11-22: The old aim point came from the gripper's world-axis bounding box and moved up to 14 mm in the gripper's own frame with tilt, roll and opening. It is now a fixed point on the wrist roll axis (`arm_model.TIP_LOCAL`, tests/test_arm_model.py).
- 2026-11-22: IK is exact inside reach (0.0 mm, tilt held) and joint-space carries don't dip toward the table (lowest gripper point stayed >= 8 cm on 11 cm-high carries), but from an awkward start pose it stalled far from the target in 3 of 30 random starts: `_ik` retries from `IK_SEEDS`. Forearm and wrist roll now have hard limits 0.29-0.79 (cable). A joint-space descent drifted ~6 mm sideways over 8 cm: descents now go through 3 straight-line points.
- 2026-11-22: Refitting the calibration with roll-joint zeros included changed the wrist roll zero by 19 deg without improving the fit error and only halved the visible jaw offset, so it was **not** saved: the saved frames don't pin the roll zeros down (all 7 have the rolls at the same values). Capture frames with different roll values before trying again.
- 2026-11-22: Depth noise on the bare table is 2.2 mm (1 sigma) per frame at 320x240; `find_objects` and `depth_point` now use the median of 5 frames (`DEPTH_FRAMES`). The bridge's nearest-value pooling is left as it is: a median there would invent points between an object and the table at edges.
- 2026-11-21: Unfrozen calibration ranges keep widening during use (st3215 port_meta.rs:407-423); auto-calibration zeroes the servo Position Offset (calibrator.rs:423-424); mirroring deadband is 20 steps ≈1.8° (mirror.rs:190-195).
- 2026-11-21: Station has built-in leader→follower mirroring, controllable via `STC_MOTOR_MIRRORING_COMMAND`; nothing arbitrates between it and our writes (motors-mirroring source, ADR 0009).
- 2026-11-21: Leader bus is ~5 V, so web-UI Auto Calibrate is disabled (<7.0 V); calibrate the leader manually (St3215BusCalibrationPage.tsx).
- 2026-11-21: The .dmg app keeps station.yaml in ~/Library/Application Support/ and owns ports 8888/8889 (notes/norma-station.html).
- 2026-11-21: Station has no Windows build; run it on the MacBook (README).
