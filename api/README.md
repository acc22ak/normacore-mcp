# normacore-api

Talk to the ElRobot follower arm in plain language: one Claude API call per request, local
code checks every step, Station moves the arm, joint feedback confirms it. Plan: `plan.html`.

## Setup
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    # .env holds ANTHROPIC_API_KEY (chmod 600, git-ignored)
Station must run (`cd ~/station && ./station --web --tcp`). Station access reuses
the repo root's `server.py` as a library, so its motion engine and safety checks apply.

## Run
    .venv/bin/python src/app.py --sim      # no hardware: saved desk photo + simulated arm
    .venv/bin/python tools/smoke.py        # M0: follower live? joints, limits, both cameras
    .venv/bin/python tools/smoke.py --home # ...and glide to HOME at half speed (asks first)
    .venv/bin/python tools/teach.py ready  # save the current pose (home, ready, A, B, C)
    .venv/bin/python src/app.py            # the real thing
    .venv/bin/python tests/test_offline.py # no arm, no API

In the app, type a request; `stop` halts the arm immediately without asking Claude.
Every request appends a timing line to `logs/runs.jsonl`.

## Layout
- `config/arm.yaml` joints, directions, limits (URDF + demo soft limits), hazards
- `config/poses.yaml` home/ready/A/B/C poses, gestures, speed, reach tolerance
- `config/scene.yaml` crop, image size, slot boxes for the A/B/C labels
- `src/station_io.py` only module that talks to Station (RealIO, SimIO)
- `src/brain.py` only module that calls Claude
- `src/safety.py` every step passes here; nothing outside the limits is sent
- `src/skills.py` steps -> keyframes -> motion -> verified from joint readings
- `src/perceive.py` crop, resize, labels; `src/app.py` terminal loop, stop path, logging
