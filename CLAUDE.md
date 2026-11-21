# CLAUDE.md: normacore-mcp

## What this is
Hackathon project for the **NormaCore challenge: "AI-powered robot control: integrate the
Station API into Codex or Claude so it can control the robot by looking through the cameras."**

`server.py` is an MCP server (Python, `mcp<2` FastMCP API). It connects to a running
NormaCore **Station** over TCP :8888 and exposes the tools `look`, `list_cameras`,
`get_robot_state`, `move_joints`, `nudge_joints`, `set_gripper`, `hold` and `relax`.
`README.md` has the setup steps for people. This file is context for agents.

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
- Ideas / next steps: a scripted `home` pose; record a joint direction map automatically; a higher `resize_target` (480) for better vision;
  hand-off to a trained SmolVLA policy for fast low-level skills with Claude as the high-level planner.
