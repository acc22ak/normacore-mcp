# NormaCore Station → Claude / Codex (MCP)

An MCP server that lets Claude Code (or Codex) **see through the station cameras**
and **move the ElRobot follower arm** through the NormaCore Station API.

```
 Claude Code / Codex ──MCP stdio──► server.py ──TCP :8888──► station (MacBook) ──USB──► arms + cameras
```

**Normal use: say what to move where** ("put the Lego figure next to the green pen"). Claude runs the
pick-and-place loop: `find_objects` (objects found in the depth camera's color image, with positions and how to
grasp each) → `pick` (from above; it checks the jaws before closing) → `place` / `place_near` → `home`.
For small or flat objects it stops with the jaws open, looks at a zoomed photo, adjusts with `nudge_jaws` and then
`close_and_lift`. After a miss it uses `lift_and_look` and tries again from where it is.

This needs the RealSense depth camera (`realsense-bridge/`, run with `sudo`), mounted looking **down at the table
at 50° or steeper**, and three calibrations: camera pose (`arm_fit.py`), reach (`calibrate_reach`) and the sideways
jaw offset (measured by `pick` itself). `CLAUDE.md` has the workflow, the rules and the tool map.

Also there, for special cases: joint-level tools (`move_joints`, `go_to`, `play_motion`, ...), replayable skills
(`start_attempt` / `save_attempt_as_skill` / `run_skill`: record a motion once and replay it fast, for objects on
fixed marked spots), `teach_mode` (the leader arm drives the follower), and `hold` / `relax` / `estop.py`.

---

## 0. Run everything on the MacBook

Station ships for **macOS (Apple Silicon) and Linux only**. Windows isn't supported
yet, so run the station on the MacBook Air. Run Claude Code on that Mac too:
it's the simplest setup. (You *can* run this MCP server on another laptop and point
`STATION_HOST` at the Mac's IP, but hackathon Wi-Fi often blocks laptop-to-laptop traffic.)

---

## 1. Hardware checklist

### Power: don't mix up the bricks
| Arm | Servos | PSU |
|---|---|---|
| **Leader** (handle + trigger) | ST3215 **7.4 V** (C044/C046) | **5 V** brick |
| **Follower** (gripper + camera) | ST3215 **12 V** 30 kg | **12 V** brick |

Read the label on each brick before plugging it in. **12 V on the leader burns its 7.4 V servos.**

### Servo IDs (manual step 4)
Every servo needs a unique ID **1–8**, base to gripper (ElRobot: 1 base, 2 shoulder,
3 upper-arm roll, 4 elbow, 5 forearm roll, 6 wrist pitch, 7 wrist roll, 8 gripper/trigger).
New servos all ship as ID 1. If they were assembled without setting IDs:
connect **one servo at a time** to the driver board and set its ID in the Station
web UI (ST3215 → motor config page). Set IDs on both arms.

### Leader arm (`manual-leader.pdf`, 7 pages)
The leader uses the follower manual up to the wrist, then swaps the gripper for:
1. **Handle (A) + Wrist Roll (C)**, 1× M2x6 self-tapping screw.
2. **Servo 8** pressed fully into the handle footprint, 4× M3x8, plug in the servo wire (any socket).
3. **Idler horn (E)** + 4× M2x6 + 1× M3x8.
4. **Trigger (B)**, 4× M3x8.
- Fasten the servo horn on the wrist servo firmly before step 1.
- From your photo the leader looks done. Check the trigger moves freely and the last servo wire reaches the board.

### Wiring (both arms)
- Servos are **daisy-chained**: servo → servo → … → driver board. Each joint must still turn fully after you route the wires.
- **Waveshare bus servo board: set the jumper to `B` (USB)**. On `A` the PC can't see the servos.
- Driver board: DC jack to that arm's PSU, USB-C (**data** cable, not charge-only) to the hub.
- **Powered USB hub**: switch the hub on and plug it into its own power. Cameras draw too much for the Mac's ports.

### Cameras
- **Gripper cam**: already on the follower's camera mount. Route the cable along the arm (manual steps 68–75) and leave slack at every joint.
- **Desk cam** (any UVC webcam): fix it so it sees the **whole workspace plus the gripper**,
  about 45° from the front-side and roughly 40–60 cm away. Put a clamp or tape on it; it must not move between runs. Even light, and avoid the window glare in your photo.
- Both cameras go into the powered hub.

---

## 2. Station (on the Mac)

1. Download `station-macos-arm64.dmg` (or `.zip`) from
   https://github.com/norma-core/norma-core/releases (README points to v0.1.0-beta.10 or newer).
2. Run: `./station --web --tcp` and open **http://localhost:8889**.
   The `.dmg` app starts its own station in the background on 8888/8889, so **use either the app or the CLI, never both**
   (they'd clash on the ports). The app keeps `station.yaml` and `station_data/` in `~/Library/Application Support/`.
3. You should see **2 ST3215 buses** (leader + follower) and **2 cameras**.
   **Write down the follower bus serial** (shown on the bus card).
4. **Calibrate both arms** (bus card → calibration page). Calibration sets each joint's `range_min/range_max`,
   and the MCP refuses to move uncalibrated joints.
   - **Follower (12 V)**: **Auto Calibrate** → **Save**.
   - **Leader (5 V)**: Auto Calibrate is disabled below 7.0 V, so do it by hand: **Reset**, move **every joint through its
     full range** by hand, then **Save**.
   - **Always press Save.** It freezes the ranges. Until then Station keeps widening them as joints move, which shifts
     every normalized position (`get_robot_state` warns about unsaved buses).
5. Let Claude see more detail: in `station.yaml` set `usb-video: resize_target: 480`
   (default 224 is tiny for an LLM; this also changes stored dataset frames). With the `.dmg` app the file is in
   `~/Library/Application Support/`. Restart station.

Sanity test: leader → follower teleop is **built into Station**. On the follower's bus card, pick the leader bus as
the mirror source; the follower now copies the leader. Set it back to none to stop.
**Stop mirroring before Claude drives.** Station doesn't stop two writers from fighting over the follower arm.

---

## 3. Install the MCP server

```bash
git clone --depth 1 https://github.com/norma-core/norma-core.git ~/norma-core
cd normacore-mcp
# needs Python >= 3.10 (the mcp package requires it); macOS system python3 is 3.9, so use uv:
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

### Claude Code
```bash
claude mcp add normacore \
  -e NORMA_CORE_REPO=$HOME/norma-core \
  -e STATION_HOST=localhost \
  -e FOLLOWER_BUS=<FOLLOWER_SERIAL> \
  -- $PWD/.venv/bin/python $PWD/server.py
```
Then in `claude`: `/mcp` should list `normacore` with 31 tools.
Then name the cameras once: ask Claude to `look`, tell which picture is which, and `set_camera_role` them
(or set `CAMERAS`). Station itself has no idea which camera is the gripper cam. Roles are saved by the
camera's USB vendor:product:serial, so they survive replugging it or moving it to another port.

### Codex (`~/.codex/config.toml`)
```toml
[mcp_servers.normacore]
command = "/ABS/PATH/normacore-mcp/.venv/bin/python"
args = ["/ABS/PATH/normacore-mcp/server.py"]
env = { NORMA_CORE_REPO = "/Users/you/norma-core", STATION_HOST = "localhost", FOLLOWER_BUS = "<FOLLOWER_SERIAL>" }
```

### Env vars
| var | default | meaning |
|---|---|---|
| `NORMA_CORE_REPO` | — (required) | norma-core checkout (Python client + generated protobufs) |
| `STATION_HOST` | `localhost` | station host (port 8888) |
| `FOLLOWER_BUS` | auto if only 1 bus | serial of the arm Claude may move. **Never the leader.** |
| `MAX_STEP_NORM` | `0.15` | max joint change per call (0..1 of range); bigger requests get clamped |
| `MOVE_SPEED` | `600` | servo speed in steps/s |
| `GRIPPER_MOTOR_ID` | `8` | `6` for an SO-101 |
| `LEADER_BUS` | auto if exactly 2 buses | leader arm serial: never commanded; used by `teach_mode` |
| `TEACH_MAX_POSE_DIFF` | `0.2` | `teach_mode(on)` refuses if leader/follower poses differ more than this (normalized) |
| `DEMO_PATH_TOL` | `0.01` | how closely a leader-arm demonstration's path is kept when saved as a skill |
| `CAMERAS` | — | default camera roles, e.g. `desk=32e6:9228:202604081837,gripper=3f2a` (`camera_id` from `list_cameras`, queue id or queue hash prefix); `data/cameras.json` wins |
| `CAMERAS_FILE` | `data/cameras.json` | where `set_camera_role` saves roles |
| `MAX_VEL_NORM` | `0.5` | fast-mode speed cap per joint (normalized range per second); untuned, start low |
| `MOTION_RATE_HZ` | `20` | fast-mode control rate (goals per second). 30 overloaded the station with a camera + 2 arms (late ticks, tracking aborts) |
| `SERVO_SPEED_HEADROOM` | `1.2` | servo speed cap = planned peak x this, so the servo never flattens the smooth curve |
| `CURRENT_LIMIT` / `CURRENT_LIMITS` | `100` / — | fast-mode collision limit (same units as Station's current threshold); per motor e.g. `8:40,5:60` |
| `TRACK_ERR_MAX` / `TRACK_ERR_TICKS` | `0.10` / `5` | abort when a joint lags its target this much for this many ticks |
| `SKILLS_DIR` / `RUNS_FILE` | `data/skills` / `data/runs.jsonl` | where skills and the run log are saved |
| `HISTORY_MAX_ENTRIES` | `20000` | most entries one history read (`save_attempt_as_skill`) transfers; longer attempts use a bigger server-side stride, never a shorter one |
| `GRIPPER_OPEN_NORM` | — | `1` or `0`: which end of the gripper range is open, for `{"gripper": "open"/"close"}` skill steps. Check with `set_gripper` + `look` first |
| `LIMITS_FILE` | `data/limits.json` | optional soft limits per joint, e.g. `{"2": [0.1, 0.9]}` |
| `FRAME_MAX_AGE_S` | `2.0` | `look` marks a camera frame older than this as STALE (e.g. camera unplugged or frozen). Assumes station and MCP share a clock |
| `CAMERA_GONE_S` | `60` | a camera with no frame for this long, or that reported a disconnect, counts as unplugged and is hidden from `look("all")` / `list_cameras` (Station keeps a queue for every camera it ever saw; `list_cameras(include_gone=True)` shows them) |
| `MCP_TRANSPORT` | `stdio` | `streamable-http` to run the server yourself (e.g. in a tmux pane to watch logs) |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8790` | listen address for `streamable-http` |

Running as HTTP instead of stdio:
```bash
MCP_TRANSPORT=streamable-http NORMA_CORE_REPO=$HOME/norma-core .venv/bin/python server.py
claude mcp add --transport http normacore http://127.0.0.1:8790/mcp
```
Code changes then need a restart of that process (not `/mcp`).

---

## E-stop (run this before Claude drives)

A separate program with its own Station connection. Run it **on a laptop next to the arm**, in its own terminal, and keep that window focused:

```bash
NORMA_CORE_REPO=$HOME/norma-core STATION_HOST=localhost FOLLOWER_BUS=<FOLLOWER_SERIAL> .venv/bin/python estop.py
```

| key | action |
|---|---|
| **Space** | **HOLD**: freeze the arm where it is (torque stays on) |
| **X** / **Esc** | **RELAX**: torque off on every arm. It goes limp and **drops**, so support it |
| **R R** | release the latch (R twice within 3 s) |
| **Q** | quit (a latch stays set) |

One-shot form: `estop.py hold | relax | release | status`.

It stops the servos directly **and** sets a latch in Station (`estop/state` queue). The MCP server and
autopilot check that latch before every move and on every motion tick, so a running `go_to` / `play_motion` /
`run_skill` / `move_until` stops within one tick and every new move is refused until a human releases it.
After a RELAX the server does **not** re-enable torque. The latch lives in Station, so it works when the e-stop
and the MCP server run on different laptops. Limits: it needs Station to be reachable (the last resort is the
12 V brick), and Station's own leader→follower mirroring doesn't read the latch (RELAX still works; HOLD gets
overwritten by mirrored goals).

---

## 4. First run: keep a hand near the power

1. Start Station, the RealSense bridge (`sudo .venv/bin/python bridge.py` in `realsense-bridge/`) and the e-stop.
   Keep the Mac on AC power.
2. Calibrate once per camera position (ask Claude: "calibrate the depth camera", then "calibrate the reach"), with
   the table clear on one side.
3. Put an object 20-35 cm in front of the arm and say, for example:
   > Put the Lego figure next to the green pen.

Tips:
- Objects further than ~36 cm from the base can only be reached at a shallow jaw angle; beyond ~39 cm not at all.
- Leave a jaw's width (about 3.5 cm) between objects, or Claude has to move one out of the way first.
- `relax` = torque off, and **the arm drops**. `hold` freezes it in place.
- If a pick misses, Claude lifts the jaws, looks again and retries from there. Results are logged in
  `data/runs.jsonl`, and the camera frames it used in `data/frames/`.

## Offline self-test (no robot needed)
```bash
.venv/bin/python tests/offline_test.py . ~/norma-core
.venv/bin/python tests/test_motion.py .
.venv/bin/python tests/test_learn.py .
```
Runs every tool against simulated station data.
