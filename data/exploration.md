# Arm exploration log (follower bus 5B8E115950, ElRobot)

Working area given by the human: **84 × 45 cm** (where the arm base sits in it: not yet known).
Cameras: only the **gripper camera** is connected (no desk camera), so positions are judged from
where the gripper camera points, not measured in cm.

## Session 2026-11-21

### Home pose (saved in data/poses.json)
- Placed by hand, torque off, confirmed by the human.
- Gripper camera at home: looks **forward and level across the desk**.
- shoulder ≈ 0.02 (near its min), elbow ≈ 0.98 (near its max) → home is the folded pose.

### Direction hints (not yet confirmed one joint at a time)
- With shoulder ≈ 0.84 / elbow ≈ 0.29 the gripper camera looked at the **ceiling** → raising the
  shoulder norm probably lifts the arm. Confirm with a single ±0.05 nudge while watching.
- shoulder +0.05 from 0.84: the camera view tilted toward the clutter on the left (cardboard box
  grew); not enough to call up/down from the gripper camera alone.

### ⚠ forearm_roll (motor 5) is unreliable — do not command it until recalibrated
- Calibration from the 19:13 save: range 4–4092 (almost the full circle), which is not a real limit.
  That save also failed to write motors 6–8 (EEPROM timeouts), after several resets during auto-calibration.
- `hold` + `go_to(home)` asked forearm_roll for raw 3516, i.e. where it already was. It turned
  **~230° to raw 876** instead, and the gripper camera image rotated ~100°.
- Every go_to/play_motion sends a goal to every joint, so any move can spin it again.
  Check the gripper camera cable for twist after this.
- Fix: recalibrate the arm (Live → Calibrate → Recalibrate → Auto calibrate → check ranges → Save),
  then re-save home (its raw values were recorded under the bad calibration).

## Session 2026-11-21, after recalibration (arm aligned in 3D)
Workspace: **84 × 45 cm, marked by gray tape** on the desk. Objects placed by the human: a **green pen**
and a **small white candy**.
Home re-saved under the new calibration (data/poses.json). Old forearm warning no longer applies:
`hold` after recalibration moved nothing (forearm stayed at raw 2222).

### Home view (gripper cam, level)
Gray boundary tape crosses the image left→right at mid-height (far edge of the area). Green pen lies
close in front, slightly left of centre, pointing away from the arm. Candy not visible from home.

### Joint directions (one-joint nudges from home, judged on the gripper cam)
| joint | +norm moves… | step size seen | notes |
|---|---|---|---|
| base_yaw | arm turns to **its left** (scene shifts right in image) | ~9° per 0.05 | candy appears at the right edge at 0.461 (right of home) |
| shoulder | arm **leans forward**; gripper cam tilts **down** toward the desk | at 0.15 (elbow 0.94) the cam looks steeply down at the desk just past the pen | |
| elbow | **− unfolds** the forearm: gripper rises, cam tilts **up**; + folds it back | 0.94 → 0.84 brought the tape back into view | sags ~0.017 below target under gravity |
| wrist_pitch | gripper points **down** (cam tilts down) | 0.46 → 0.51: tape moved to the top edge | |
| wrist_roll | rolls the gripper; scene turns **clockwise** in the image | 0.05 tilted the tape ~20° | range 98–3998 (almost full turn): no real stops |
| forearm_roll | rolls the forearm; scene turns slightly clockwise (same sense as wrist roll) | +0.03 moved exactly +0.03 | fixed by the recalibration; still almost a full-turn range |
| upper_arm_roll | swings the forearm sideways in the folded pose (pen slid out top-left at +0.05) | low confidence | |
| gripper | **+ opens**, − closes | 0.56 → 0.70 visibly wider | range 1268–2828 |

### Reach straight ahead (base_yaw 0.51)
Each step = shoulder +0.1, elbow −0.15 (arm extends forward, cam rises and tilts up; add wrist_pitch + to look down).
| step | shoulder | elbow | wrist_pitch | gripper cam sees |
|---|---|---|---|---|
| home | 0.01 | 0.94 | 0.46 | tape at mid-frame, pen close front-left |
| 1 | 0.25 | 0.71 | 0.51 | tape closer/larger, pen by the left jaw |
| 2 | 0.36 | 0.58 | 0.51 | cam higher, more level; tape just ahead; pen under the left jaw |
| 3 | 0.46 | 0.46 | 0.61 | gripper just short of the **far tape** (saved as pose `reach_far_center`) |
So the far edge of the area straight ahead is reachable at about shoulder 0.46 / elbow 0.46.

### Sweep at full forward reach (shoulder 0.46, elbow 0.46, wrist_pitch 0.61)
| base_yaw | ≈ angle from centre | gripper cam sees |
|---|---|---|
| 0.51 | 0° (straight ahead) | far tape straight ahead, pen at bottom-left |
| 0.66 | ~27° left | far tape runs diagonally; desk edge + radiator top-left; pen under the right jaw |
| 0.81 | ~54° left | **far-left corner of the tape** (strip along the desk edge + strip down the right of the image) |
| 0.95 | ~79° left (near base max 1.0) | radiator + back desk edge; tape along that edge and a **left boundary strip** |
| 0.36 | ~27° right | **candy** just ahead, slightly right of the right jaw (closer than the far tape) |
| 0.20 | ~56° right | candy at the left edge of the image → candy ≈ base 0.28–0.30 at this reach; tape curving top-right |
| 0.06 | ~81° right (near base min 0.0) | **far-right corner**: strip along the top + **right boundary strip** close to the right jaw |
(base_yaw spans 1024–3072 raw = 180° total, so 0.05 ≈ 9°.)

### Layout (inferred)
The arm base sits near the **middle of one long side**, facing across the area. At full forward reach
(shoulder ≈ elbow ≈ 0.46) the gripper is near the far tape straight ahead; the far-left corner is at
base ≈ 0.81, the far-right corner at base ≈ 0.06–0.2, and the side strips are at the ends of the base
range (≈ 0.95 left, ≈ 0.06 right). So the whole marked area is within base_yaw 0.05–0.95.

### Objects
| object | where | how to find it again |
|---|---|---|
| green pen | close, in front, slightly left of centre, pointing away from the arm | visible from home; under the left jaw at reach step 2 (shoulder 0.36 / elbow 0.58) |
| white candy | right of centre, mid-depth (closer than the far tape) | at full reach, base ≈ 0.28–0.36 |

### Gotchas
- `nudge_joints` returns before the joint has finished moving, and frames lag a little: **wait ~1 s
  before `look`**, or the picture shows the previous pose (it fooled me once: same image twice).

- **`speed_scale` doesn't slow a `go_to` without `duration_s`** (it's sized by the speed cap). Pass
  `duration_s` (e.g. 2–3 s for big base swings) to move slowly.
- **Unlisted joints drift** a few hundredths over many `go_to`s (they're held at their measured value).
  Give every joint an absolute target for precise poses.
- **20 Hz control rate, not 30**: at 30 Hz the station (camera + 2 arms) fell behind and a base move
  aborted on tracking error. Some late ticks still happen at 20 Hz; harmless so far.

### Going home safely from full reach (learned the hard way)
- One all-joints `go_to(home)` from full reach **aborted halfway**: the shoulder couldn't lift the
  extended arm back fast enough (0.11 behind, safety stop). The straight joint-space path also swings
  the gripper low over the desk.
- What worked, in two steps, every joint given an absolute target:
  1. **shoulder → 0.01 alone, over 4 s** (others held): leans the arm back, gripper rises away from the desk.
  2. **everything else → home, over 3 s.**
- Rule of thumb: moves that make the shoulder lift the extended arm need ≥ 4 s; move it on its own.

### Gripper camera mount (from the human's photos, 2026-11-21)
- On a small mast **on top of the gripper body, behind the jaws**, centred between them left↔right.
- Roughly **4–6 cm above the gripper centre line** and **~8–10 cm behind the jaw tips** (estimated
  from photos, not measured). Points **forward, about parallel to the jaws** (maybe slightly down).
- Jaws = two long flat plates on a rack, closing **left↔right in the image**.
- Consequences:
  - **Blind zone between the jaws**: the space between the plates near their base is below the bottom
    edge of the image. An object touching the bottom edge is at about the **jaw tips**, not inside.
  - **Point the gripper steeply down (wrist_pitch ≈ 0.68–0.72) for grasps**: the camera then looks down
    past the jaws and keeps the object in view almost until it's between the plates.
- ⚠ **Cable**: the camera's USB cable runs along the forearm with a loose loop near the wrist.
  Keep **forearm_roll and wrist_roll within about ±90° of straight** (≈ ±0.25 norm from 0.54) so the
  cable doesn't wind up or tear the connector.

### Pick-and-place: candy → next to the pen (done, 2026-11-21, 4 grasp attempts)
What worked (final attempt):
1. Travel pose toward the object: base first at home height, then extend (shoulder ~0.42, elbow ~0.50,
   wrist_pitch ~0.61–0.70) over ≥ 3–4 s.
2. Approach in small steps of **shoulder +0.03…0.05 / elbow −0.03…0.07** (forward and down along the
   jaws), looking after each: the object moves **down the image toward the jaw tips** as you close in.
   For mostly-straight-down use **shoulder + and elbow + together**.
3. **Open wide (0.85–0.9) before the last steps.** A lying candy fits across its length in the open gap.
4. Close with `go_to({"gripper": 0.3}, duration_s≈3)` and check it worked: the gripper **stops short of
   target** (0.32 vs 0.30, `settled: false`) and **current rises** (8 vs 2–3 on an empty close).
5. **Verify the hold before carrying**: small lift (shoulder −0.05) + **base wiggle ±0.04**. If the object
   stays between the jaws while the desk moves, it's held. If it slides in the image, it's on the desk.
6. Carry at travel height (shoulder 0.46 / elbow 0.46), go to the target, lower one or two small
   steps, open, lift, then home in 2 stages.
Placement: candy at base ≈ 0.53, shoulder ≈ 0.48, elbow ≈ 0.62, wrist_pitch ≈ 0.68 → lands just right
of the pen tip.

What went wrong (avoid):
- **Closing on the jaw tips with low force** (gripper 0.385, current 1): the candy slipped out during a
  big lift. Grip deeper and tighter.
- **Re-tightening by guessing**: after a stopped close the gripper had kept moving to 0.41; asking for
  0.48 *opened* it. Read the gripper position first.
- **Overshooting forward**: once the object is below the bottom edge of the gripper cam you're blind;
  two closes happened in front of the candy. If it leaves the view, back off until it's visible
  between the jaws again, then go straight down.
- The `go_to` tracking check often stops a closing gripper on contact before `grasp_detected` fires,
  so don't rely on `grasp_detected` alone.

### Still to explore
- ~~One-joint nudges~~ done (upper_arm_roll still low confidence).
- Near ring (the area close to the base, left and right of the pen) at short reach.
- Height above the desk per pose: the gripper cam can't measure it. Test lowering toward the pen slowly
  and stop on the first contact current.
- Real cm positions: a desk camera looking down would turn this into a measured map.

### Pick-and-place with the RealSense depth camera: minifigure → next to the pen tip (done, 2026-11-21, first try)
New tool chain (station-viewer dev page, `window.__realsense` in the browser): calibrated RealSense D435
(Level to table + Fit to arm shape, ~1 cm) gives object positions in the 3D view's world frame (m, robot base at
origin, Y up) via `pixelToWorld(fx, fy)`; `solveTipIk(x, y, z)` gives normalized joint targets for `go_to`;
`tip()` gives the live gripper position from the joint readings. Closing the loop on tip() vs object worked.
- Minifigure (standing, ~4 cm tall) at world (0.357, 0, 0.075). Grasp: IK pose above (y 0.12 commanded) →
  open 0.88 → IK y 0.065 → IK y 0.05 (real tip ≈ 0.047; arm sags ~1.5–3 cm below the IK height when extended)
  → close in steps 0.42 → 0.30 → 0.24: stopped at 0.295 with **current 21** = firm hold.
  Grasp pose: base 0.427, shoulder 0.742, uar 0.514, elbow 0.397, forearm 0.530, wrist_pitch 0.439, wrist_roll 0.538.
- Place 4 cm beside the pen tip, toward the base: world (0.351, 0, -0.172). Pose: base 0.667, shoulder 0.842,
  uar 0.462, elbow 0.332, wrist_pitch 0.41; open only to 0.45 (the right jaw is close to the pen).
- **Shoulder > 0.6 with the arm extended is heavy**: lifting back needs ≥ 7 s for 0.45 → 0.2. Elbow fold back to
  home needs ≥ 6 s. A gripper close over 3 s trips the tracking check (gripper is slow): close with set_gripper steps.
- Laptop on low battery throttled the CPU: 40/45 late ticks and stale-state aborts. Keep the Mac on AC power.
- Not saved as a skill: the attempt window also contained the camera recalibration moves. Re-record cleanly
  (start_attempt right before the approach) to make `minifig_to_pen_tip`.

### One-call tools: find_objects / pick / place / place_near / home (2026-11-21, first hardware runs)
- `home` works from anywhere, including fully stretched out over the table (shoulder 0.99): shoulder back in
  ≤0.3 chunks (~9-10 s each), rolls straight, fold. ~32 s.
- `pick` motion is reliable: approach (1 correction) → descend (2 corrections) lands within 1-2 mm of the
  aimed point by the arm model. The aimed point is 3 cm past the "tip" along the jaws (`GRASP_REACH_M`):
  the jaws reach 5.5 cm past the tip, so aiming the tip put the jaw ends past small objects and pushed them.
- Lying minifigure (1.3 x 2 x 4 cm): 6 attempts closed on air. The gripper camera last showed it centred
  between the jaws but *ahead of the jaw tips*, consistently ~2 cm short along the pointing direction even
  after aiming 2 cm further: the remaining error is between the depth-camera position and the arm model at
  low, far poses. Next: close the loop with the gripper camera before closing (centre the object between
  the jaws, step forward until it disappears into the blind zone), or recalibrate with poses near the table.
- `find_objects` false detections: arm parts near the table when the arm is low (call it with the arm at
  home); a lying minifigure (< 9 mm above the fitted table) and a thin pen are missed → `depth_point`.
