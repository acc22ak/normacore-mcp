"""Intel RealSense (D435 etc.) depth camera: a small daemon that owns the camera + an MCP server.

Station can't use this camera: its Linux camera driver (libuvc) fails to open the D435
("Failed to open device for camera 32902:2823:...: -12", a multi-function UVC device), and
Station only handles colour formats, not 16-bit depth. So the D435 is read here with
librealsense (pyrealsense2) instead, independent of Station.

Only one process can stream from the camera, and every Claude session spawns its own MCP
server, so the camera is owned by one daemon that serves frames over HTTP on localhost:

    .venv/bin/python depthcam.py serve      # daemon: http://127.0.0.1:8791 (live page at /)
    .venv/bin/python depthcam.py mcp        # MCP stdio server: look_depth, depth_at, depthcam_status

Register the MCP side with:
    claude mcp add depthcam -- $PWD/.venv/bin/python $PWD/depthcam.py mcp

Like autopilot.py this is a separate, removable server: server.py doesn't import it.
The daemon needs `pip install -r requirements-depth.txt`; the MCP side only needs `mcp`.

HTTP API (daemon):
  GET /                 live page: colour + depth side by side, click a pixel to measure
  GET /color.jpg        latest colour frame (JPEG)
  GET /depth.jpg        latest depth frame, colourised and aligned to the colour frame (JPEG)
  GET /stream/color     MJPEG stream (also /stream/depth)
  GET /depth_at?x=&y=&r=2   median depth (m) around colour pixel (x, y) + 3D point in the
                        colour camera frame (m; x right, y down, z forward)
  GET /status           serial, resolution, fps, frame age, intrinsics

Env vars:
  DEPTHCAM_HOST / DEPTHCAM_PORT   daemon listen address (default 127.0.0.1 / 8791)
  DEPTHCAM_URL                    where the MCP side finds the daemon (default http://127.0.0.1:8791)
  DEPTHCAM_WIDTH / DEPTHCAM_HEIGHT / DEPTHCAM_FPS   stream mode (default 640 x 480 @ 30)
  DEPTHCAM_SERIAL                 pick one camera when several are connected
  FRAME_MAX_AGE_S                 frames older than this are reported STALE (default 2.0, as in server.py)
"""

from __future__ import annotations

import errno
import io
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, stream=sys.stderr,  # stdout is the MCP channel in mcp mode
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("depthcam")

HOST = os.environ.get("DEPTHCAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEPTHCAM_PORT", "8791"))
URL = os.environ.get("DEPTHCAM_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
WIDTH = int(os.environ.get("DEPTHCAM_WIDTH", "640"))
HEIGHT = int(os.environ.get("DEPTHCAM_HEIGHT", "480"))
FPS = int(os.environ.get("DEPTHCAM_FPS", "30"))
SERIAL = os.environ.get("DEPTHCAM_SERIAL", "").strip()
FRAME_MAX_AGE_S = float(os.environ.get("FRAME_MAX_AGE_S", "2.0"))


# ----------------------------------------------------------------------------
# pure helpers (unit-tested without a camera)

def median_depth(depth, x: int, y: int, r: int, scale: float) -> tuple[float | None, int]:
    """Median of the valid (non-zero) depth values in the (2r+1)^2 window around (x, y),
    in metres, and how many valid pixels it used. `depth` is a 2D uint16 array (rows = y)."""
    h, w = depth.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"pixel ({x}, {y}) is outside the {w}x{h} frame")
    win = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    vals = sorted(int(v) for v in win.ravel() if v)
    if not vals:
        return None, 0
    return vals[len(vals) // 2] * scale, len(vals)


def deproject(intr: dict, x: float, y: float, z: float) -> list[float]:
    """Pixel + depth (m) -> 3D point (m) in the camera frame, pinhole model (no distortion:
    the D435 colour stream reports ~zero distortion coefficients)."""
    return [round((x - intr["ppx"]) / intr["fx"] * z, 4), round((y - intr["ppy"]) / intr["fy"] * z, 4), round(z, 4)]


# ----------------------------------------------------------------------------
# daemon: owns the camera

class FrameStore:
    """Latest frames, shared between the capture thread and HTTP handlers."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self.color = None          # HxWx3 uint8 (RGB)
        self.depth = None          # HxW uint16, aligned to colour
        self.depth_rgb = None      # HxWx3 uint8, colourised depth
        self.stamp = 0.0           # time.time() of the latest frame
        self.info: dict = {}       # serial, intrinsics, depth_scale, ...
        self.error: str | None = None
        self._jpeg: dict[str, tuple[int, bytes]] = {}

    def put(self, color, depth, depth_rgb):
        with self.cond:
            self.color, self.depth, self.depth_rgb = color, depth, depth_rgb
            self.stamp = time.time()
            self.seq += 1
            self.error = None
            self.cond.notify_all()

    def jpeg(self, which: str) -> tuple[int, bytes] | None:
        """(seq, JPEG bytes) of 'color' or 'depth', encoded once per frame."""
        from PIL import Image
        with self.cond:
            arr, seq = (self.color if which == "color" else self.depth_rgb), self.seq
            cached = self._jpeg.get(which)
        if arr is None:
            return None
        if cached and cached[0] == seq:
            return cached
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=85)
        out = (seq, buf.getvalue())
        with self.cond:
            self._jpeg[which] = out
        return out

    def status(self) -> dict:
        with self.cond:
            age = time.time() - self.stamp if self.stamp else None
            return {**self.info, "frames": self.seq, "age_s": round(age, 3) if age is not None else None,
                    "error": self.error}


def capture_loop(store: FrameStore, stop: threading.Event) -> None:
    """Stream colour + depth (aligned to colour) from the RealSense; reconnects on errors."""
    import numpy as np
    import pyrealsense2 as rs

    while not stop.is_set():
        pipe = rs.pipeline()
        try:
            cfg = rs.config()
            if SERIAL:
                cfg.enable_device(SERIAL)
            cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
            cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
            profile = pipe.start(cfg)
            dev = profile.get_device()
            ci = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            store.info = {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
                "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
                "width": ci.width, "height": ci.height, "fps": FPS,
                "depth_scale_m": dev.first_depth_sensor().get_depth_scale(),
                "intrinsics": {"fx": ci.fx, "fy": ci.fy, "ppx": ci.ppx, "ppy": ci.ppy,
                               "model": str(ci.model), "coeffs": list(ci.coeffs)},
                "aligned_to": "color",
            }
            log.info("streaming %s %s (USB %s) %dx%d@%d", store.info["name"], store.info["serial"],
                     store.info["usb"], ci.width, ci.height, FPS)
            align, colorizer = rs.align(rs.stream.color), rs.colorizer()
            while not stop.is_set():
                frames = align.process(pipe.wait_for_frames(5000))
                c, d = frames.get_color_frame(), frames.get_depth_frame()
                if not c or not d:
                    continue
                store.put(np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy(),
                          np.asanyarray(colorizer.colorize(d).get_data()).copy())
        except Exception as e:  # camera unplugged, busy, USB hiccup: report and retry
            store.error = f"{type(e).__name__}: {e}"
            log.warning("camera error, retrying in 2 s: %s", store.error)
            stop.wait(2.0)
        finally:
            try:
                pipe.stop()
            except Exception:
                pass


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Depth camera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;padding:12px;font:14px system-ui,sans-serif;background:#111;color:#ddd}
 .row{display:flex;gap:12px;flex-wrap:wrap} figure{margin:0;flex:1 1 320px} .wrap{position:relative}
 img{width:100%;display:block;border-radius:6px;cursor:crosshair;background:#222}
 figcaption{margin:4px 0;color:#aaa} #out{margin-top:8px;font-family:ui-monospace,monospace}
 .mark{position:absolute;width:14px;height:14px;border:2px solid #0f0;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;display:none}
</style></head><body>
<div class="row">
 <figure><figcaption>Colour</figcaption><div class="wrap"><img id="c" src="/stream/color"><div class="mark" id="mc"></div></div></figure>
 <figure><figcaption>Depth (aligned to colour)</figcaption><div class="wrap"><img id="d" src="/stream/depth"><div class="mark" id="md"></div></div></figure>
</div>
<div id="out">Click a pixel to measure its distance.</div>
<div id="st" style="color:#888;margin-top:6px"></div>
<script>
function measure(ev){
  const img=ev.target, r=img.getBoundingClientRect();
  const x=Math.round((ev.clientX-r.left)/r.width*img.naturalWidth), y=Math.round((ev.clientY-r.top)/r.height*img.naturalHeight);
  for (const id of ["mc","md"]){const m=document.getElementById(id);
    m.style.left=(x/img.naturalWidth*100)+"%"; m.style.top=(y/img.naturalHeight*100)+"%"; m.style.display="block";}
  fetch(`/depth_at?x=${x}&y=${y}&r=2`).then(r=>r.json()).then(j=>{
    document.getElementById("out").textContent = j.depth_m==null ? `(${x}, ${y}): no depth here`
      : `(${x}, ${y}): ${j.depth_m.toFixed(3)} m   point [x, y, z] = [${j.point_m.join(", ")}] m`;});
}
document.getElementById("c").onclick=measure; document.getElementById("d").onclick=measure;
setInterval(()=>fetch("/status").then(r=>r.json()).then(s=>{document.getElementById("st").textContent=
  `${s.name||"?"} ${s.serial||""}  ${s.width||"?"}x${s.height||"?"}@${s.fps||"?"}  frame age ${s.age_s}s`+(s.error?`  ERROR: ${s.error}`:"");}),1000);
</script></body></html>"""


def make_handler(store: FrameStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/status":
                return self._json(200, store.status())
            if u.path in ("/color.jpg", "/depth.jpg"):
                got = store.jpeg(u.path[1:-4])
                if got is None:
                    return self._json(503, {"error": store.error or "no frame yet"})
                return self._send(200, got[1], "image/jpeg")
            if u.path in ("/stream/color", "/stream/depth"):
                return self._stream(u.path.rsplit("/", 1)[1])
            if u.path == "/depth_at":
                try:
                    x, y = int(q["x"][0]), int(q["y"][0])
                    r = max(0, min(10, int(q.get("r", ["2"])[0])))
                except (KeyError, ValueError):
                    return self._json(400, {"error": "need integer x and y (colour-frame pixels), optional r"})
                with store.cond:
                    depth, info, stamp = store.depth, dict(store.info), store.stamp
                if depth is None:
                    return self._json(503, {"error": store.error or "no frame yet"})
                try:
                    z, n = median_depth(depth, x, y, r, info["depth_scale_m"])
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                return self._json(200, {
                    "x": x, "y": y, "r": r, "valid_pixels": n,
                    "depth_m": round(z, 4) if z is not None else None,
                    "point_m": deproject(info["intrinsics"], x, y, z) if z is not None else None,
                    "frame_age_s": round(time.time() - stamp, 3)})
            return self._json(404, {"error": "not found"})

        def _stream(self, which: str):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with store.cond:
                        store.cond.wait_for(lambda: store.seq != last, timeout=2.0)
                    got = store.jpeg(which)
                    if got is None or got[0] == last:
                        continue
                    last = got[0]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(got[1])).encode() + b"\r\n\r\n" + got[1] + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve() -> None:
    store, stop = FrameStore(), threading.Event()
    try:  # bind before touching the camera, so a second copy doesn't grab it
        httpd = ThreadingHTTPServer((HOST, PORT), make_handler(store))
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            sys.exit(f"depthcam: port {PORT} is already in use, so the daemon is probably already running "
                     f"(see http://{HOST}:{PORT}/). Stop that one first, or set DEPTHCAM_PORT.")
        raise
    httpd.daemon_threads = True
    threading.Thread(target=capture_loop, args=(store, stop), daemon=True).start()
    log.info("depth camera page: http://%s:%d/", HOST, PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


# ----------------------------------------------------------------------------
# MCP side: talks to the daemon over HTTP (no pyrealsense2 needed here)

def _get(path: str, timeout: float = 3.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(URL + path, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"depth camera daemon not reachable at {URL} ({e}). Start it with "
                           "`.venv/bin/python depthcam.py serve` in the normacore-mcp folder.") from None


def _get_json(path: str) -> dict:
    code, body = _get(path)
    data = json.loads(body or b"{}")
    if code != 200:
        raise RuntimeError(f"depth camera: {data.get('error', code)}")
    return data


def build_mcp():
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("depthcam", instructions=(
        "RealSense RGB-D camera next to the robot. look_depth shows colour + depth (aligned); "
        "depth_at(x, y) gives the distance in metres at a colour-image pixel. A frame marked STALE "
        "is not the current scene: don't plan moves with it."))

    @mcp.tool()
    async def depthcam_status() -> dict:
        """Depth camera model, serial, resolution, intrinsics and how old the latest frame is."""
        return _get_json("/status")

    @mcp.tool()
    async def look_depth() -> list:
        """Latest colour image and depth image (colourised, aligned pixel-for-pixel to the colour
        image) from the RealSense, plus the distance at the image centre. Pixel coordinates for
        depth_at refer to the colour image."""
        st = _get_json("/status")
        code_c, color = _get("/color.jpg")
        code_d, depth = _get("/depth.jpg")
        if code_c != 200 or code_d != 200:
            return [f"depth camera has no frame yet: {st.get('error') or 'starting'}"]
        age = st.get("age_s")
        stale = age is None or age > FRAME_MAX_AGE_S
        w, h = st.get("width", 0), st.get("height", 0)
        centre = _get_json(f"/depth_at?x={w // 2}&y={h // 2}&r=3") if w and h else {}
        label = (f"{st.get('name')} {st.get('serial')}: colour + aligned depth, {w}x{h}, "
                 + (f"STALE: frame {age} s old. This is NOT the current scene; check the camera and "
                    "the depthcam daemon." if stale else f"frame {age:.2f} s old")
                 + (f". Centre ({w // 2}, {h // 2}): {centre['depth_m']} m" if centre.get("depth_m") else ""))
        return [label, Image(data=color, format="jpeg"), Image(data=depth, format="jpeg")]

    @mcp.tool()
    async def depth_at(x: int, y: int, radius: int = 2) -> dict:
        """Distance at colour-image pixel (x, y): median of valid depth in a (2*radius+1)^2 window,
        in metres, plus the 3D point [x right, y down, z forward] in metres in the colour camera's
        frame. depth_m is None where the camera has no depth (too close <~0.2 m, shiny or dark)."""
        return _get_json(f"/depth_at?x={int(x)}&y={int(y)}&r={int(radius)}")

    return mcp


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "serve":
        serve()
    elif mode == "mcp":
        build_mcp().run(transport="stdio")
    else:
        sys.exit("usage: depthcam.py serve | mcp")
