# realsense-bridge

Streams an Intel RealSense D435 (color + depth aligned to color) to station-viewer over
`ws://127.0.0.1:8890`. station-viewer shows the feed and draws the point cloud in the
follower arm's 3D view.

## Setup (once)

```sh
cd realsense-bridge
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

librealsense needs root on macOS (the system camera driver grabs the device otherwise):

```sh
sudo .venv/bin/python bridge.py
```

Options: `--fps 15`, `--max-depth 1.5` (m), `--decimate 2` (depth 320×240), `--jpeg-quality 80`,
`--port 8890`, `--host 127.0.0.1`.

Then run station-viewer in dev mode (`npm run dev` in
`norma-core/software/station/clients/station-viewer`, open http://localhost:5173). Vite proxies
`/realsense` to the bridge. The RealSense panel appears when the bridge is connected.

## Wire format

Binary message per frame: `uint32 LE headerLen` + JSON header + JPEG + depth (`uint16 LE`, mm,
row-major, `depth.w × depth.h`, 0 = no data). Intrinsics in the header are for the depth grid,
and depth is aligned to the color image. Text messages are status updates:
`{"type":"status","state":"streaming|no_device|error","message":...,"serial":...,"firmware":...}`.
