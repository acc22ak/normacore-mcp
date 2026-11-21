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
  - The leader bus therefore reads ~5 V, and the web UI **disables Auto Calibrate below 7.0 V**. Calibrate the leader
    **manually**: Reset → move every joint by hand through its full range → Save (freeze). The follower (12 V) can use Auto Calibrate.
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
- Normalized position: `server.py` `to_norm`/`from_norm` are ports of Station's mirroring math
  (`motors-mirroring/src/inference/normalize.rs`, `mirror.rs:96-98`). **Calibrated ranges can wrap past 4095→0**
  (`range_max < range_min`, range size `(4096 - min) + max`). A joint counts as calibrated when its range size is
  ≥ 100 steps (viewer's `MIN_CALIBRATED_RANGE`). Station's own mirroring adds a small `safety_margin` that we don't.

## Unverified on hardware (check these on the real robot, then move them to Learned facts)
- Whether leader and follower normalized poses really line up after both are calibrated (the code maps % of range to % of range; physical alignment depends on how each arm was calibrated).
- How long Station takes to turn follower torque off after STOP, and how far the arm sags before `teach_mode(off)` re-holds it.
- How fast the follower jumps when mirroring starts (mirroring sets speed by distance, `mirror.rs` `get_speed_and_accel_by_distance`).
- Read rate of `st3215/inference` (needed for `skill_from_history`).
- Which gripper direction is open.
- Engine defaults are untuned guesses: `MAX_VEL_NORM` 0.5, `TRACK_ERR_MAX` 0.10 × 5 ticks, `MOTION_ACCEL` 50. Measure `max_tracking_error`,
  `late_ticks` and `elapsed_s` vs `planned_s` from real `play_motion` reports, then tune (raising limits needs the human's OK, §5).
- Whether 20 Hz is reachable: each tick does one command write + one state read over TCP; `late_ticks` shows if it isn't.
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
1. Always `look` + `get_robot_state` before moving. After every move, `look` again. Each `look` label says how old
   its frame is; a frame marked **STALE** (older than `FRAME_MAX_AGE_S`, or the camera reported a disconnect) is not
   the current scene. Never plan or verify a move with it: stop and tell the human the camera needs checking.
   Use camera **roles**: `look("desk")` for the workspace, `look("gripper")` for fine alignment. Station doesn't know
   which camera is which (queues are `/<station>/usbvideo/<md5 of device id>`), so at session start call `list_cameras`;
   if roles are missing or `role_problems` is non-empty, `look` at all cameras, identify them (gripper cam shows the jaws
   and its view moves with the arm; desk cam is fixed and shows the whole table), call `set_camera_role`, and tell the human.
   Roles are saved by `camera_id` (USB `vendor:product:serial`), not queue id: on Linux the device id ends in the USB
   device number, so every replug creates a new queue. Queues of unplugged cameras (no frame for `CAMERA_GONE_S`, or
   a disconnect/capture-end event) are hidden; `list_cameras(include_gone=True)` shows them.
2. Only ever command the **follower** bus (`FOLLOWER_BUS`). Never write to the leader.
3. Small steps: `nudge_joints` ≤ 0.05 when near objects/table; the server also clamps each call to `MAX_STEP_NORM` (0.15).
4. Don't move uncalibrated joints or joints reporting `error_status != 0`.
5. `relax` = torque off, and the arm falls. Use `hold` to freeze in place instead, unless something is jammed/overloaded.
6. If mirroring (teleop) targets the follower, don't move it: two writers fight over the arm and Station won't stop
   them. `server.py` enforces this: `move_joints`/`nudge_joints`/`set_gripper`/`hold` refuse while the follower is a
   mirroring target or when the bus is a leader (fails closed if `inference/mirroring` can't be read; allowed if the
   queue doesn't exist, i.e. no mirroring driver). `relax` is always allowed. Stop mirroring with `teach_mode(on=False)`.
7. `teach_mode(on=True)` makes the follower jump to the leader's pose, so it refuses if any joint differs by more than
   `TEACH_MAX_POSE_DIFF` (0.2) unless `force=True`. `teach_mode(on=False)` waits for Station's torque-off, then holds.

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

# Strategy (why we win)
Other teams will likely have Claude guess raw joint angles from images, which fails on stage. We keep
geometry out of the LLM: **humans teach skills by hand with the leader arm, Claude learns them instantly,
plans with them in plain English, verifies every step on camera, and keeps improving them.**
Pitch: *"Show it once by hand, then just tell it what to do. It watches, acts, checks itself and gets better."*

# Roadmap (update the Status column as work lands)
| P | Component | File(s) | What | Status |
|---|---|---|---|---|
| P0 | Station I/O | `station_io.py` | split connection/state/camera/command code out of `server.py` | todo |
| P0 | Fast playback engine | `motion.py` (pure logic), `server.py` (`go_to`, `play_motion`, `stop_motion`) | ~20 Hz streaming of eased keyframes or dense recordings, `speed_scale`, per-joint velocity cap (`MAX_VEL_NORM`), soft limits (`data/limits.json`); aborts + holds on servo error bits (voltage ignored like Station), over-current (=collision), tracking error, stale state, stop/hold/relax, cancellation (shielded hold); gripper over-current = grasp (stops squeezing) | done, offline-tested (simulated servos); **not yet tried on hardware** |
| P0 | Teach mode + mirroring guard | `server.py` | `teach_mode(on/off)` starts/stops Station's built-in leader→follower mirroring; every motion tool refuses while the follower is a mirroring target | done, offline-tested; **not yet tried on hardware** |
| P0 | Poses | `poses.py`, `data/poses.json` | `save_pose`, `list_poses`, `go_to_pose` | todo |
| P0 | Demonstration capture | `learn.py` (pure), `server.py` | `skill_from_history(name, seconds=60)`: reads the last N s of `st3215/inference` (rate measured from bus stamps; offsets >255 encoded little-endian), trims idle, ~20 Hz dense motion + raw + keyframe summary + calibration snapshot | done, offline-tested; **not yet tried on hardware** |
| P1 | Skills + practice ladder | `server.py`, `data/skills/*.json`, `data/runs.jsonl` | `list_skills`, `run_skill` (fast, one call), `report_skill_result`; ladder 0.5 → 0.75 → 1.0 → … after **2 successes in a row**, one level down on failure; engine abort = failure; calibration change → replay from raw at ≤0.5 + `needs_check` | done, offline-tested; **not yet tried on hardware** |
| P1 | Drift handling | `motion.py`, `poses.py`, `data/compensation.json` | calibration snapshot + raw positions in skills/poses, replay-from-raw when ranges changed, goal-vs-present tracking error, per-joint compensation, `check_drift` (home pose + landmark). See Learning protocol §6. `range_frozen` + warning in `get_robot_state` already done | partly done |
| P1 | Vision | `vision.py` | `look(grid=True)` labelled A1–C3 grid on the desk cam; `find_objects(color)` HSV blobs → pixel + cell | todo |
| P2 | Pick anywhere | `poses.py` | taught `above_<cell>` poses for 9 cells, bilinear blend → `pick(cell)` / `place(cell)` | todo |
| P2 | E-stop | `estop.py` | separate process + own station connection: Space = hold, Esc = relax | todo |
| P3 | Demo agent/UI | `agent.py`, `ui/` | Claude API agent with the same tools + page with camera feeds, reasoning stream, STOP button | stretch |
| P3 | SmolVLA | upstream `software/ai/smolvla_py` | train a policy on recorded demos for one fast skill; Claude stays the planner | stretch |

Final tool surface: perception (`look`, `list_cameras`, `set_camera_role`, `find_objects`, `gripper_status`, `get_robot_state`), motion (`go_to`,
`play_motion`, `stop_motion`, `go_to_pose`, `nudge_joints`, `grasp`, `release`), planned (`edit_skill`, `pick`, `place`), learning
(`teach_mode`, `skill_from_history`, `save_pose`), skills (`list_skills`, `run_skill`, `report_skill_result`), safety (`hold`, `relax`).
**If a tool you need for the protocol below doesn't exist yet, build it (roadmap order) rather than improvising with raw joint moves.**

---

# Learning protocol: how Claude learns and improves skills
"Skill" below means a **robot skill** (a JSON file in `data/skills/`), not a Claude Code skill.

## 1. When a human shows you something → make a skill right away
Teaching setup: call `teach_mode(on)` (Station's built-in leader→follower mirroring), so the follower copies the
leader arm while the human demonstrates. Mirroring has its own current protection, so demos are safe to record.
Triggers: "watch this", "learn this", "save what I just did as X", "remember this". If the human starts demonstrating
and mirroring is off, offer to turn teach mode on.
1. **Capture**: right after the demo, `skill_from_history(name, seconds=60, description=...)` (default: the last
   minute; use a smaller N if other movement happened before the demo). It uses the **follower** positions, trims
   the still time before/after, and saves `data/skills/<name>.json` with status `"new"` at practice speed 0.5.
   Check the returned `history.covered_s` and `duration_s`: if they don't match what the human did, say so.
2. **Understand it**: `look` now (the end state) and write a one-sentence description with preconditions
   (e.g. "cube on spot A, gripper empty") into the skill's `description`/`preconditions`.
   (Camera frames at start/end are not stored automatically yet.)
3. **Confirm by replay**: `teach_mode(on=False)`, put the object back on its marked spot, tell the human, then
   `run_skill(name)` (runs at 0.5) and judge it with one `look`.
4. Commit the new skill to git (`learn: add skill <name>`).

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
the demo (teach-mode deadband alone is ~1.3 cm at full reach, see §6). The team marks: the arm base outline, the desk
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
(`learn: <skill> v3 - <why>`). After 3 failed fixes in a row, stop and ask the human to re-demonstrate.

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
  normalized value. `get_robot_state` reports `range_frozen` and a warning per bus: get unfrozen buses saved before teaching.
- Re-calibrating changes `range_min/max`, so **every normalized value in saved skills shifts**. Raw encoder positions are
  the stabler reference: auto-calibration writes the servo Position Offset (EEPROM 0x1F) to 0
  (`auto_calibrate/calibrator.rs:423-424, 500`), so raw positions keep the same zero unless a servo horn is remounted or
  the offset is changed.
- In teach mode the follower ignores leader differences of ≤20 steps (`motors-mirroring/src/inference/mirror.rs:190-195`,
  `deadband: 20` in `config.rs:48`). That's ≈1.8° per joint (360/4096×20), so **demos carry up to that error per joint**
  before anything else.

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
5. **Drift check.** At session start and after any failure that looks systematic: `go_to_pose("home")`, `look`, and
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
Recorded skills (from `skill_from_history`) store a dense `"motion": [{"t", "joints", "raw"}, ...]` that `run_skill`
plays, plus `"practice": {"level", "streak", "speed_scale"}`; their `keyframes` are a readable summary
(`start`/`pause`/`gripper turns`/`end`). Hand-written skills may instead give only `keyframes` with `duration_s`.
Keyframes may list only the joints that change; the others keep their current value. Composite skills may
use `{"skill": "<other skill>"}` steps. `raw` and `calibration` are there so a skill survives re-calibration (§6 rule 2);
`verify` says what the camera should show after that keyframe (§6 rule 1).

---

# Learned facts (Claude appends here as it learns; newest first)
- 2026-11-21: On Linux, Station's camera unique_id is `vid:pid:bus:devnum` (decimal), and devnum changes on every replug, so each replug makes a new `/<station>/usbvideo/<md5>` queue; old queues stay, ending with `ET_DEVICE_RECORDING_END` (not `ET_DEVICE_DISCONNECTED`). Linux also needs a udev rule giving the user rw on UVC USB devices, or Station logs `Failed to open device ... -3` (libusb access denied).
<!-- format: - YYYY-MM-DD: fact (evidence) -->
- 2026-11-21: Unfrozen calibration ranges keep widening during use (st3215 port_meta.rs:407-423); auto-calibration zeroes the servo Position Offset (calibrator.rs:423-424); mirroring deadband is 20 steps ≈1.8° (mirror.rs:190-195).
- 2026-11-21: Station has built-in leader→follower mirroring, controllable via `STC_MOTOR_MIRRORING_COMMAND`; nothing arbitrates between it and our writes (motors-mirroring source, ADR 0009).
- 2026-11-21: Leader bus is ~5 V, so web-UI Auto Calibrate is disabled (<7.0 V); calibrate the leader manually (St3215BusCalibrationPage.tsx).
- 2026-11-21: The .dmg app keeps station.yaml in ~/Library/Application Support/ and owns ports 8888/8889 (notes/norma-station.html).
- 2026-11-21: Station has no Windows build; run it on the MacBook (README).
