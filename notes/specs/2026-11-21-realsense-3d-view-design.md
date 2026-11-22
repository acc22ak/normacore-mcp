# RealSense D435 in station-viewer: camera feed + point cloud in the 3D view

Date: 2026-11-21
Status: implemented (user said "go, don't ask me again"; calls below were made by Claude)

## Calls made after the builder check

- **Calibration point** = the 3D view's existing gripper tip (`measureTipLocal`, bounding-box centre of the end link),
  shown as an orange marker while calibrating. The clicked surface point sits ~1–2 cm closer to the camera than that
  centre; since the camera views the gripper from roughly one direction, most of that offset is absorbed into the pose.
  Accept ~1–2 cm accuracy for now.
- **Proxy exposure**: the Vite dev server already exposes Station's `/api` (including robot commands) on all
  interfaces, so `/realsense` through the same proxy adds no new class of exposure. The bridge itself binds 127.0.0.1.
- **Which arm gets the cloud**: a "depth points" toggle on each bus card (only while the bridge is connected).
  Default on for a bus that is currently a mirroring target (the follower); explicit choice saved per bus serial.
- **Arm accuracy**: the calibration panel tells the user to run Align 3D first. No staleness tracking.
- **Align panel placement**: overlay in the arm's 3D view (bottom-left), containing a small clickable feed, the
  sliders and calibration. The live page also gets a larger RealSense card under the device views.
- **Frame**: pose is in the three.js world frame of the 3D view (robot base at origin, Y up, arm front = +Z).
  Euler order YXZ; all-zero angles look from +Z back at the robot.

## Intent

An Intel RealSense D435 is plugged into the Mac that runs Station (verified: librealsense 2.59.2 via Homebrew,
`sudo rs-enumerate-devices` lists `RealSense D435`, serial `939622074856`, firmware `5.17.3.10`, USB 3 link).
It sits on the table **about 55 cm in front of the ElRobot follower arm**, looking at it.

The user wants to see it in station-viewer, the web app they already use to watch the robot:

1. the **camera feed** (color, plus a depth view), and
2. the **depth as a 3D point cloud inside the robot's 3D view**, so real objects on the table appear next to the
   virtual arm,
3. with an **alignment tool** so the point cloud lines up with the virtual arm.

Why: to see the real scene and the robot in one spatial view (the robot's 3D view today shows only the arm).
Accurate alignment also opens the door to later using depth for reaching, but that is not part of this work.

## Goals

- Live color feed and colorized depth feed from the D435 shown in station-viewer, with a hover readout of distance.
- Live point cloud (colored by the RGB image) rendered in the **follower** arm's 3D view, in the robot's frame.
- Alignment tool with two parts:
  - **Manual sliders**: camera pose relative to the robot base (x, y, z in cm; yaw, pitch, roll in degrees), live update.
  - **Click calibration**: user moves the arm to 4–6 spots; at each spot clicks the gripper tip in the color image;
    the tool pairs that 3D point (from depth) with the tip position the 3D view computes from joint angles, and
    solves the camera pose. Shows per-pair error and RMS. The solved pose is written into the sliders for fine-tuning.
- Pose persists across reloads (browser `localStorage`, keyed by camera serial).
- When the bridge isn't running, station-viewer behaves exactly as it does today (no errors, no empty panels).

## Non-goals

- Recording depth/color into Station queues, History playback, timeline integration.
- Giving the MCP server (`normacore-mcp`, Claude's `look`) access to depth. (Possible later; see Approaches B.)
- Changes to Station (Rust/Go) or to the macOS app's bundled viewer.
- Multiple RealSense cameras; IMU (the D435 has none); depth-camera settings UI (presets, laser power, etc.).
- Showing the point cloud in the leader arm's 3D view or on the History page.

## Anti-goals (failure even if it "works")

- The 3D view or the robot UI gets noticeably slower or janky because of the point cloud / camera bytes.
  Camera payloads must stay out of React state (same rule as station-viewer ADR 0007).
- The viewer breaks, logs error spam, or shows a broken panel when the bridge isn't running.
- The bridge is reachable from other machines on the network (bind to localhost only).

## Constraints

- **librealsense needs root on macOS** (the system UVC driver claims the camera; without sudo it fails with
  "cannot access depth sensor"). The bridge is started by the user with `sudo` in a normal Terminal; the `!`
  prompt in Claude Code can't take a password.
- Python bindings: `pyrealsense2` has no macOS wheels; **`pyrealsense2-macosx==2.56.5`** has a cp312 arm64 wheel
  (checked with `uv pip compile`). Use Python 3.12 via `uv`.
- station-viewer changes are seen via `npm run dev` (Vite on `:5173`, proxies `/api` to Station `:8889`).
  The macOS Station app serves its own built viewer, which won't include this.
- station-viewer rules (`AGENTS.md`): React 19 function components, strict TS, Tailwind v4, three.js;
  `HomePage` and shared code must not import concrete device modules; tests only for plausible regressions.
- The norma-core checkout is on branch `devin/station-viewer-restyle` with uncommitted user changes
  (`App.tsx`, `package.json`, `package-lock.json`, a deleted `AsciiRobot.tsx`). Don't revert, stash or commit them.
- The user runs the code themselves; don't run the bridge or dev server as part of building.

## Approaches considered

**Data path (how depth gets from the camera to the browser)**

- **A. Bridge streams directly to the viewer over its own WebSocket — chosen by the user.**
  Small Python program reads the D435 and serves color + depth on `ws://127.0.0.1:8890`. Simple, full rate,
  no storage cost. No recording.
- B. Bridge writes into a Station queue (normfs supports writes to new queues). Gives History + MCP access, but
  depth is several MB/s into Station storage and needs a custom format and viewer plumbing. Rejected for now;
  A's wire format should be simple enough that a B-style writer could be added later.

**Alignment**

- 1. Manual sliders only: quick, accuracy depends on patience.
- 2. Click calibration only: ~1 cm accuracy, no way to fine-tune.
- **Both (Claude's call, user didn't object):** sliders as the base, click calibration to snap, sliders to refine.
- Fiducial markers (ArUco on the gripper) for automatic calibration: rejected, needs printed markers and OpenCV.

## Design

### 1. Bridge: `robotic-hackathon/realsense-bridge/`

Sibling of `normacore-mcp/` (outside the norma-core repo, like the MCP server).

- `bridge.py`, `requirements.txt` (`pyrealsense2-macosx`, `numpy`, `websockets`), `README.md` with the run command:
  `sudo .venv/bin/python bridge.py` (venv made with `uv venv --python 3.12`).
- Streams color 640×480 RGB8 and depth 640×480 Z16 at 30 fps; **aligns depth to color** (`rs.align`) so every
  depth pixel has a color. Sends to clients at a configurable rate (default 15 fps; `--fps`).
- Depth downsampled 2× to **320×240 uint16 millimetres** (nearest or min-pool, builder's call) to keep messages
  ~150 KB. Color sent as JPEG (q≈80) at 640×480; the intrinsics in each message refer to the depth resolution.
- Light filtering: zero out depth beyond `--max-depth` (default 1.5 m). Spatial/temporal filters optional, builder's call.
- WebSocket server on `127.0.0.1:8890`, broadcast latest frame to all clients, drop frames for slow clients
  (never queue unboundedly).
- Survives camera unplug/replug: keeps the server up, reports status, retries the pipeline every ~2 s.
- **Wire format** (binary message per frame):
  `uint32 LE headerLen` + UTF-8 JSON header + JPEG bytes + depth bytes. Header:
  ```json
  {"type":"frame","seq":123,"ts":1732222222.123,
   "color":{"w":640,"h":480,"bytes":51234},
   "depth":{"w":320,"h":240,"bytes":153600,"unit_m":0.001},
   "intrinsics":{"fx":..., "fy":..., "ppx":..., "ppy":...},
   "serial":"939622074856"}
  ```
  Plus text JSON status messages: `{"type":"status","state":"streaming"|"no_device"|"error","message":"...","serial":...,"firmware":...}`
  sent on connect and on changes.
  Depth is in the camera optical frame convention (x right, y down, z forward).

### 2. Viewer data layer: `station-viewer/src/realsense/`

A shared feature folder (like `src/usbvideo/`), not a Station device module, since it doesn't come from Station.

- `realsense-client.ts`: one WebSocket to the bridge, auto-reconnect with backoff (quiet: no console spam while
  the bridge is down), parses the wire format.
- `realsense-store.ts`: external store (pattern of `usbvideo/live-camera-store.ts`): latest frame + status,
  per-listener subscription. Frame bytes never go into React state; components subscribe and draw imperatively.
  Exposes a small React hook for status only (`connected`, `serial`, `state`).
- URL: connect via a Vite dev proxy path `/realsense` → `ws://127.0.0.1:8890` (added to `vite.config.ts`) so the page
  works same-origin; builder may fall back to `ws://<host>:8890` if the proxy is impractical.
- `camera-pose.ts`: pose type `{x,y,z (m), yaw,pitch,roll (rad)}` in the **robot base frame**, load/save in
  `localStorage` keyed by serial, simple subscribe for live slider updates.

### 3. Camera feed panel

- Shown on `HomePage` alongside the existing camera surfaces **only while the bridge is connected**.
- Displays the color image; toggle to colorized depth (turbo-like colormap, near = warm); hover shows depth in cm
  at the cursor. Click events are exposed for the calibration mode (below).
- Look and controls follow the existing camera surfaces (`CameraSurface`, `CameraHudControls`) and the current
  dark technical restyle.

### 4. Point cloud in the 3D view

- Only in the **follower** bus's 3D view (`isLeader` false) on the live HomePage.
- `BaseRobotRenderer` gets a small, generic extension point (e.g. an optional prop or child that receives the
  three.js scene + the robot base transform and returns a dispose function). `BaseRobotRenderer` must not import
  `src/realsense/` directly; the follower `BusCard` wires them together.
- A `THREE.Points` object parented under a "camera" `Object3D` whose transform = robot base transform × camera pose
  × optical-to-three.js axis fix. Points deprojected from depth with the intrinsics
  (`X=(u-ppx)·Z/fx`, `Y=(v-ppy)·Z/fy`), colored from the JPEG (sample at the aligned pixel).
- Performance: preallocate buffers for 320×240 points and update in place, or upload depth+color as textures and
  deproject in a vertex shader (preferred if simple). Builder's call; must not drop the 3D view below smooth.
- Small camera frustum gizmo at the camera pose so the user sees where the camera is. Toggle "Show depth points"
  in the 3D view's existing controls; point size control optional.
- **Default pose:** 0.55 m in front of the robot base along the arm's forward direction, ~0.30 m above the table,
  facing the robot, pitched down ~30°. "Forward" is determined from the ElRobot URDF and its
  `elrobotBaseRpy` (`[-π/2, 0, -π/2]`); the user will fine-tune.

### 5. Alignment tool

A collapsible panel attached to the follower's 3D view, visible when the bridge is connected.

- **Sliders:** x, y, z (cm, ±100, 0.5 cm steps), yaw, pitch, roll (deg, ±180, 0.5° steps), with number inputs,
  "Reset to default", live update, auto-save.
- **Click calibration:**
  1. "Start calibration". The user moves the arm (teleop/teach mode, by hand). The 3D view's existing
     end-effector tip (`measureTipLocal` / end-effector sphere in `BaseRobotRenderer`) is highlighted so the user
     knows which physical point to click.
  2. User clicks that point on the gripper in the color feed. The tool takes the median depth in a 5×5 window around
     the click (rejecting zeros), deprojects to a camera-frame 3D point, and records it with the current tip position
     in the robot base frame. If depth is missing at the click, say so and don't record.
  3. After ≥4 pairs, "Solve" runs a rigid fit (Kabsch/Umeyama, rotation + translation, no scale), shows per-pair
     error and RMS in cm, flags pairs > 3× median error, lets the user delete a pair and re-solve.
  4. "Apply" writes the pose to the sliders/storage. "Cancel" discards.
  - Guidance text: spread the spots out (different heights, left/right, near/far), keep the arm still when clicking.

### 6. Tests (per station-viewer `AGENTS.md`)

- Rigid-fit solver: recovers a known pose from synthetic noisy pairs; rejects degenerate input (<3 non-collinear points).
- Wire-format parser: header/JPEG/depth slicing with a hand-built message; truncated message is rejected.
- No UI snapshot tests.

## Left to the builder

- Exact component names/file split, styling details, where the toggle and panel sit in the 3D view.
- CPU vs shader point deprojection; depth downsampling method; bridge-side filters.
- Reconnect/backoff timings; JPEG quality.
- How the optical-frame → three.js/robot-frame axis conversion is expressed in code (must be correct and tested via
  the default pose looking at the robot).

## Added after first hardware test (2026-11-21)

Click calibration alone was off by ~10 cm and ~11° of roll (gripper is small and dark in the image at ~90 cm).
Two automatic steps were added to the align panel; click calibration stays as a fallback.

1. **Level to table** (`table-plane.ts`): RANSAC + least-squares plane fit on the lower part of the depth image →
   pitch, roll and camera height (table assumed at the robot base height, y = 0). Measured: plane flat to 2 mm.
2. **Fit to arm shape** (`arm-icp.ts`): ICP between the real arm's depth points and the 3D arm's mesh vertices, solving
   only yaw + side/distance (level kept). **Record pose** at 4–6 different arm poses first: a single pose can slide
   along the long, thin arm (two single-pose fits differed by 9 cm). Measured with 5 poses: 5.9 mm RMS on fitted poses,
   11.4 mm RMS on a held-out pose; result within ~1 cm of the best single-pose fit. **Check** reports the error at the
   current pose without fitting.
