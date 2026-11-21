# NormaCore Station → Claude / Codex (MCP)

An MCP server that lets Claude Code (or Codex) **see through the station cameras**
and **move the ElRobot follower arm** through the NormaCore Station API.

```
 Claude Code / Codex ──MCP stdio──► server.py ──TCP :8888──► station (MacBook) ──USB──► arms + cameras
```

Tools: `look`, `list_cameras`, `get_robot_state`, `move_joints`, `nudge_joints`,
`set_gripper`, `hold`, `relax` (safe stop: torque off), `teach_mode` (Station's built-in leader→follower
mirroring on/off; motion tools refuse while it's on), `set_camera_role` (name cameras "desk" / "gripper" so
`look("desk")` works; saved in `data/cameras.json`).

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
Then in `claude`: `/mcp` should list `normacore` with 10 tools.
Then name the cameras once: ask Claude to `look`, tell which picture is which, and `set_camera_role` them
(or set `CAMERAS`). Station itself has no idea which camera is the gripper cam.

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
| `LEADER_BUS` | auto if exactly 2 buses | leader arm serial, used by `teach_mode`; never commanded |
| `TEACH_MAX_POSE_DIFF` | `0.2` | `teach_mode(on)` refuses if leader/follower poses differ more than this (normalized) |
| `CAMERAS` | — | default camera roles, e.g. `desk=usbvideo/9c1e,gripper=3f2a` (queue id or unique hash prefix); `data/cameras.json` wins |
| `CAMERAS_FILE` | `data/cameras.json` | where `set_camera_role` saves roles |
| `FRAME_MAX_AGE_S` | `2.0` | `look` marks a camera frame older than this as STALE (e.g. camera unplugged or frozen). Assumes station and MCP share a clock |
| `MCP_TRANSPORT` | `stdio` | `streamable-http` to run the server yourself (e.g. in a tmux pane to watch logs) |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8790` | listen address for `streamable-http` |

Running as HTTP instead of stdio:
```bash
MCP_TRANSPORT=streamable-http NORMA_CORE_REPO=$HOME/norma-core .venv/bin/python server.py
claude mcp add --transport http normacore http://127.0.0.1:8790/mcp
```
Code changes then need a restart of that process (not `/mcp`).

---

## 4. First run: keep a hand near the arm

Prompt Claude with something like:
> Use the normacore tools. Look through both cameras and describe the scene. Then get the robot state,
> open the gripper, and move it slowly (nudges ≤ 0.05) toward the red cube. Look after every move.
> If anything looks wrong, call relax.

Tips:
- Find out which direction each joint moves with small `nudge_joints` calls (±0.05) and `look`.
  Write the mapping into a `CLAUDE.md` (e.g. "shoulder +: arm leans forward"). Claude gets much better with it.
- `relax` = torque off, and **the arm drops**. Hold it, or call `hold` first.
- Camera frames are ~10 Hz and moves take ~0.4 s, so this is a slow look → think → move loop, not real-time control.
  For fast skills, record leader→follower demos and train SmolVLA (`software/ai/smolvla_py`).
  Claude can then plan the steps and the policy executes them.

## Offline self-test (no robot needed)
```bash
.venv/bin/python tests/offline_test.py . ~/norma-core
```
Runs every tool against simulated station data.
