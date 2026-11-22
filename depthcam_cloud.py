"""3D point-cloud view for depthcam.py.

Drop this file next to depthcam.py. It adds two routes to the live server:
  /cloud      a 3D page you can rotate (drag), pan (right-drag) and zoom (scroll)
  /cloud.bin  the latest frame as a point cloud (binary, see cloud_bin())

Hook-up (three small edits in depthcam.py):
  1. import:         from depthcam_cloud import store_cloud_frames, handle_cloud
  2. capture_loop(): after aligning, call store_cloud_frames(depth_frame, color_frame)
                     with the *aligned* pyrealsense depth and colour frames.
  3. do_GET():       first line:  if handle_cloud(self): return
"""
import struct
import threading
import time
from urllib.parse import urlparse, parse_qs

import numpy as np

_lock = threading.Lock()
_src = {}


def store_cloud_frames(depth_frame, color_frame=None):
    """Keep a copy of the latest aligned depth (+ colour) frame and its intrinsics.

    depth_frame / color_frame are pyrealsense2 frames after rs.align(rs.stream.color),
    so depth pixels line up with colour pixels and share the colour intrinsics.
    """
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    depth = np.asanyarray(depth_frame.get_data()).copy()
    color = None
    if color_frame is not None:
        c = np.asanyarray(color_frame.get_data())
        if c.ndim == 3 and c.shape[:2] == depth.shape:
            c = c[..., :3]
            if "bgr" in str(color_frame.profile.format()).lower():
                c = c[..., ::-1]
            color = np.ascontiguousarray(c)
    with _lock:
        _src.update(depth=depth, color=color, scale=float(depth_frame.get_units()),
                    fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, t=time.time())


def set_cloud_source(depth, scale, fx, fy, cx, cy, color_rgb=None):
    """Same as store_cloud_frames but from numpy arrays (depth uint16, colour HxWx3 RGB)."""
    with _lock:
        _src.update(depth=depth, color=color_rgb, scale=float(scale),
                    fx=fx, fy=fy, cx=cx, cy=cy, t=time.time())


# ---------------------------------------------------------------------------
# Table plane: physical "how high above the table" for the arm's look-and-move loop
_plane = {"t": 0.0, "fit": None}
_marks = []          # last measurements: [{"xyz": [..], "label": str, "t": time}]


def _points(s, step=4, max_m=2.0):
    d = s["depth"][::step, ::step].astype(np.float32) * s["scale"]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    u *= step
    v *= step
    m = (d > 0.1) & (d < max_m)
    z = d[m]
    return np.stack([(u[m] - s["cx"]) * z / s["fx"], (v[m] - s["cy"]) * z / s["fy"], z], axis=1)


def fit_table(s, iters=150, tol=0.006, rng=np.random.default_rng(0)):
    """Largest plane in view (RANSAC, then least squares on its inliers): the table top.
    Returns (n, d, inlier_fraction) with n . p + d = height above the table (m), n pointing
    towards the camera side, or None."""
    P = _points(s)
    if len(P) < 500:
        return None
    best, best_in = None, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        d = -n @ a
        inl = np.abs(P @ n + d) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (n, d), inl
    Q = P[best_in]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]   # refine: least-squares normal of the inliers
    d = -n @ c
    if d < 0:                                # camera (origin) on the positive side
        n, d = -n, -d
    return n, float(d), float(best_in.mean())


def table_plane(max_age=1.0):
    with _lock:
        s = dict(_src)
    if not s:
        return None
    if time.time() - _plane["t"] > max_age or _plane["fit"] is None:
        _plane.update(t=time.time(), fit=fit_table(s))
    return _plane["fit"]


def _xyz(s, x, y):
    z = s["depth"][y, x] * s["scale"]
    return np.array([(x - s["cx"]) * z / s["fx"], (y - s["cy"]) * z / s["fy"], z], np.float32)


def height_at(x, y, r=4, ring=(14, 26)):
    """Physical height at colour pixel (x, y):
      height_m    top of whatever is there (90th percentile of a (2r+1)^2 window, so a thin pen
                  tip isn't averaged away and single noisy pixels don't win), above the TABLE plane
      surface_m   the surface around it (median of a ring r=14..26 px out), above the table:
                  e.g. a paper pad ~0.01
      above_surface_m  height_m - surface_m: how far the thing at (x, y) sticks up from what's
                  around it; a pen tip touching the paper reads ~0 (noise about +-4 mm at 0.8 m)."""
    with _lock:
        s = dict(_src)
    fit = table_plane()
    if not s or fit is None:
        return {"error": "no depth frame or no table in view"}
    n, d, frac = fit
    dep = s["depth"]
    h, w = dep.shape

    def heights(ys, xs):
        z = dep[ys, xs].astype(np.float32) * s["scale"]
        ok = z > 0
        xs, ys, z = xs[ok], ys[ok], z[ok]
        H = n[0] * (xs - s["cx"]) * z / s["fx"] + n[1] * (ys - s["cy"]) * z / s["fy"] + n[2] * z + d
        return H, xs, ys, z

    yy, xx = np.mgrid[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    H, xs, ys, z = heights(yy.ravel(), xx.ravel())
    if not len(H):
        return {"x": x, "y": y, "height_m": None, "note": "no depth here (too close, shiny or dark)"}
    top = float(np.percentile(H, 90))
    i = int(np.argmin(np.abs(H - top)))
    p = np.array([(xs[i] - s["cx"]) * z[i] / s["fx"], (ys[i] - s["cy"]) * z[i] / s["fy"], z[i]], np.float32)
    yy, xx = np.mgrid[max(0, y - ring[1]):min(h, y + ring[1] + 1), max(0, x - ring[1]):min(w, x + ring[1] + 1)]
    rr = np.hypot(yy - y, xx - x)
    m = (rr >= ring[0]) & (rr <= ring[1])
    Hs = heights(yy[m], xx[m])[0]
    surf = float(np.median(Hs)) if len(Hs) > 10 else None
    below = p - n * top                         # the table point straight below it
    with _lock:
        _marks[:] = (_marks + [{"xyz": p.tolist(), "below": below.tolist(), "h": top,
                                "above": None if surf is None else top - surf, "t": time.time()}])[-6:]
    out = {"x": x, "y": y, "height_m": round(top, 4), "at_pixel": [int(xs[i]), int(ys[i])],
           "point_m": [round(float(v), 4) for v in p], "camera_dist_m": round(float(np.linalg.norm(p)), 4),
           "table_fit": round(frac, 2)}
    if surf is not None:
        out.update(surface_m=round(surf, 4), above_surface_m=round(top - surf, 4))
    return out


def distance_between(x1, y1, x2, y2, r=2):
    """Straight-line distance (m) between the surfaces at two colour pixels."""
    with _lock:
        s = dict(_src)
    if not s:
        return {"error": "no depth frame"}
    pts = []
    for x, y in ((x1, y1), (x2, y2)):
        win = s["depth"][max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        win = win[win > 0]
        if not len(win):
            return {"error": f"no depth at ({x}, {y})"}
        z = float(np.median(win)) * s["scale"]
        pts.append(np.array([(x - s["cx"]) * z / s["fx"], (y - s["cy"]) * z / s["fy"], z]))
    return {"distance_m": round(float(np.linalg.norm(pts[0] - pts[1])), 4)}


def height_image(max_h=0.10):
    """Height above the table as an RGB image aligned to the colour image: table and below =
    black, then blue (0.5 cm) -> green -> yellow -> red (max_h and up); no reading = dark grey."""
    with _lock:
        s = dict(_src)
    fit = table_plane()
    if not s or fit is None:
        return None
    n, d, _ = fit
    z = s["depth"].astype(np.float32) * s["scale"]
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    H = n[0] * (u - s["cx"]) * z / s["fx"] + n[1] * (v - s["cy"]) * z / s["fy"] + n[2] * z + d
    t = np.clip((H - 0.005) / (max_h - 0.005), 0, 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    hue = (0.66 * (1 - t))                      # 0.66 blue -> 0 red
    k = lambda m: (m + hue * 6) % 6
    for i, m in enumerate((5, 3, 1)):
        rgb[..., i] = (255 * (1 - np.clip(np.minimum(np.minimum(k(m), 4 - k(m)), 1), 0, 1))).astype(np.uint8)
    rgb[H < 0.005] = 0
    rgb[z == 0] = 40
    return rgb


def cloud_bin(step=4, min_m=0.1, max_m=4.0):
    """Deproject the latest frame.

    Returns bytes:  uint32 N | N*3 float32 xyz (metres, camera frame: x right,
    y down, z forward) | N*3 uint8 rgb.   None if no frame has arrived yet.
    """
    with _lock:
        s = dict(_src)
    if not s:
        return None
    step = max(1, int(step))
    d = s["depth"][::step, ::step].astype(np.float32) * s["scale"]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    u *= step
    v *= step
    m = (d > min_m) & (d < max_m)
    z = d[m]
    x = (u[m] - s["cx"]) * z / s["fx"]
    y = (v[m] - s["cy"]) * z / s["fy"]
    xyz = np.stack([x, y, z], axis=1).astype("<f4")
    if s["color"] is not None:
        rgb = s["color"][::step, ::step][m]
    else:
        rgb = np.full((len(z), 3), 200, np.uint8)
    return struct.pack("<I", len(z)) + xyz.tobytes() + rgb.astype(np.uint8).tobytes()


def handle_cloud(handler):
    """Serve /cloud and /cloud.bin from a BaseHTTPRequestHandler. Returns True if handled."""
    url = urlparse(handler.path)
    if url.path == "/cloud":
        body, ctype = CLOUD_PAGE.encode(), "text/html; charset=utf-8"
    elif url.path in ("/table", "/height_at", "/distance", "/marks"):
        import json
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/table":
                fit = table_plane()
                out = ({"error": "no table found"} if fit is None else
                       {"normal": [round(float(v), 5) for v in fit[0]], "d": round(fit[1], 5),
                        "camera_height_m": round(fit[1], 4), "inlier_fraction": round(fit[2], 3)})
            elif url.path == "/height_at":
                out = height_at(int(q["x"]), int(q["y"]), int(q.get("r", 4)))
            elif url.path == "/distance":
                out = distance_between(int(q["x1"]), int(q["y1"]), int(q["x2"]), int(q["y2"]))
            else:
                with _lock:
                    out = {"marks": [m for m in _marks if time.time() - m["t"] < 120]}
        except (KeyError, ValueError, IndexError) as e:
            out = {"error": f"bad request: {e}"}
        body, ctype = json.dumps(out).encode(), "application/json"
    elif url.path == "/height.jpg":
        import io as _io
        from PIL import Image
        img = height_image()
        if img is None:
            handler.send_response(503); handler.send_header("Content-Length", "0"); handler.end_headers()
            return True
        buf = _io.BytesIO(); Image.fromarray(img).save(buf, "JPEG", quality=85)
        body, ctype = buf.getvalue(), "image/jpeg"
    elif url.path == "/cloud.bin":
        q = parse_qs(url.query)
        try:
            step = int(q.get("step", ["4"])[0])
            max_m = float(q.get("max", ["4"])[0])
        except ValueError:
            step, max_m = 4, 4.0
        body = cloud_bin(step=step, max_m=max_m)
        if body is None:
            handler.send_response(503)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return True
        ctype = "application/octet-stream"
    else:
        return False
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)
    return True


CLOUD_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth point cloud</title>
<style>
 html,body{margin:0;height:100%;overflow:hidden;background:#111;color:#ddd;font:13px system-ui,sans-serif}
 canvas{display:block;width:100vw;height:100vh;touch-action:none;cursor:grab}
 #ui{position:absolute;top:10px;left:10px;background:#000b;padding:10px 12px;border-radius:8px;min-width:210px}
 #ui label{display:flex;justify-content:space-between;gap:8px;margin:5px 0;align-items:center}
 #ui input[type=range]{width:110px}
 #ui a{color:#8cf}
 #stat{margin-top:6px;color:#9a9;font-variant-numeric:tabular-nums}
 #help{margin-top:4px;color:#888;font-size:11px}
 /* ?embed=1: compact, for a side pane - settings folded behind a small button */
 body.embed #ui{display:none;top:34px;left:6px;min-width:190px;font-size:12px}
 body.embed.open #ui{display:block}
 #gear{display:none;position:absolute;top:6px;left:6px;z-index:2}
 body.embed #gear{display:block}
 #mini{display:none;position:absolute;bottom:6px;left:8px;color:#9a9;font-size:11px;font-variant-numeric:tabular-nums;pointer-events:none}
 body.embed #mini{display:block}
 #labels{position:absolute;inset:0;pointer-events:none;overflow:hidden}
 .tag{position:absolute;transform:translate(-50%,-120%);background:rgba(0,0,0,.75);color:#fd4;border:1px solid #fd4;
      border-radius:4px;padding:1px 5px;font:12px/1.3 ui-monospace,monospace;white-space:nowrap}
 #hud{position:absolute;top:6px;right:8px;color:#bcb;font:11px/1.4 ui-monospace,monospace;text-align:right;pointer-events:none}
 #hud .bar{display:inline-block;width:90px;height:8px;vertical-align:middle;border-radius:2px;
           background:linear-gradient(90deg,#000 0 6%,#00f 6%,#0ff 30%,#0f0 50%,#ff0 72%,#f00)}
 button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 8px;cursor:pointer}
</style></head><body>
<canvas id="c"></canvas>
<button id="gear" title="3D view settings">&#9881; settings</button><div id="mini"></div><div id="labels"></div><div id="hud"></div>
<div id="ui">
 <b>Point cloud</b> &middot; <a href="/">2D view</a>
 <label>Colour <select id="mode"><option value="rgb">camera</option><option value="depth">depth</option><option value="height">height above table</option></select></label>
 <label>Density <select id="step"><option>2</option><option>3</option><option selected>4</option><option>6</option><option>8</option></select></label>
 <label>Max dist <input id="maxd" type="range" min="0.3" max="8" step="0.1" value="3"><span id="maxv"></span></label>
 <label>Point size <input id="psize" type="range" min="1" max="8" step="0.5" value="2"></label>
 <label>Pause <input id="pause" type="checkbox"></label>
 <button id="reset">Reset view</button>
 <div id="stat">waiting for frames&hellip;</div>
 <div id="help">drag: rotate &middot; right/shift-drag: pan &middot; scroll: zoom</div>
</div>
<script>
// Plain WebGL, no libraries, so it works on an offline robot.
const $ = id => document.getElementById(id);
if (new URLSearchParams(location.search).has('embed')) {
  document.body.classList.add('embed');
  $('gear').onclick = () => document.body.classList.toggle('open');
  $('mode').value = 'height';      // side pane: show physical height by default
}
const cv = $('c'), gl = cv.getContext('webgl', {antialias:true});
function sh(type, s){ const o = gl.createShader(type); gl.shaderSource(o, s); gl.compileShader(o);
  if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(o); return o; }
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER,
 'attribute vec3 p; attribute vec3 c; uniform mat4 m; uniform float s; varying vec3 v;'+
 'void main(){ gl_Position = m*vec4(p,1.0); gl_PointSize = s; v = c; }'));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,
 'precision mediump float; varying vec3 v; void main(){ gl_FragColor = vec4(v,1.0); }'));
gl.linkProgram(prog); gl.useProgram(prog);
const aP = gl.getAttribLocation(prog,'p'), aC = gl.getAttribLocation(prog,'c');
const uM = gl.getUniformLocation(prog,'m'), uS = gl.getUniformLocation(prog,'s');
gl.enableVertexAttribArray(aP); gl.enableVertexAttribArray(aC);
gl.enable(gl.DEPTH_TEST); gl.clearColor(0.067,0.067,0.067,1);

const pBuf = gl.createBuffer(), cBuf = gl.createBuffer(), axP = gl.createBuffer(), axC = gl.createBuffer();
const L = 0.1;   // axes at the sensor: red x(right) green y(up) blue z(back toward viewer)
gl.bindBuffer(gl.ARRAY_BUFFER, axP); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0,0,L,0,0, 0,0,0,0,L,0, 0,0,0,0,0,L]), gl.STATIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER, axC); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array([255,0,0,255,0,0, 0,255,0,0,255,0, 60,120,255,60,120,255]), gl.STATIC_DRAW);

// ---- camera (orbit around a target) ----
const cam = {tx:0, ty:0, tz:-1.5, yaw:0, pitch:0, dist:1.55};
function resetView(zc){ zc = zc || 1.5; Object.assign(cam, {tx:0, ty:0, tz:-zc, yaw:0, pitch:0, dist:zc+0.05}); }
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=a=>{const l=Math.hypot(...a)||1; return [a[0]/l,a[1]/l,a[2]/l];};
function basis(){ const cp=Math.cos(cam.pitch);
  const z=[cp*Math.sin(cam.yaw), Math.sin(cam.pitch), cp*Math.cos(cam.yaw)];
  const x=nrm(cross([0,1,0],z)), y=cross(z,x); return {x,y,z}; }
function matrix(){
  const {x,y,z} = basis(), t=[cam.tx,cam.ty,cam.tz];
  const e=[t[0]+z[0]*cam.dist, t[1]+z[1]*cam.dist, t[2]+z[2]*cam.dist];
  const V=[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1];
  const a=cv.width/cv.height, f=1/Math.tan(Math.PI/6), n=0.01, F=100, nf=1/(n-F);
  const P=[f/a,0,0,0, 0,f,0,0, 0,0,(F+n)*nf,-1, 0,0,2*F*n*nf,0], M=new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++){ let s=0; for (let k=0;k<4;k++) s+=P[k*4+r]*V[c*4+k]; M[c*4+r]=s; }
  return M;
}
let drag=null;
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('pointerdown', e => { drag={x:e.clientX, y:e.clientY, pan:e.button===2||e.shiftKey}; cv.setPointerCapture(e.pointerId); cv.style.cursor='grabbing'; });
cv.addEventListener('pointerup', () => { drag=null; cv.style.cursor='grab'; });
cv.addEventListener('pointermove', e => {
  if (!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if (drag.pan){ const {x,y}=basis(), k=cam.dist*0.0015;
    cam.tx-=(x[0]*dx-y[0]*dy)*k; cam.ty-=(x[1]*dx-y[1]*dy)*k; cam.tz-=(x[2]*dx-y[2]*dy)*k; }
  else { cam.yaw-=dx*0.005; cam.pitch=Math.max(-1.55, Math.min(1.55, cam.pitch+dy*0.005)); }
});
cv.addEventListener('wheel', e => { e.preventDefault(); cam.dist=Math.max(0.05, Math.min(30, cam.dist*Math.exp(e.deltaY*0.001))); }, {passive:false});
$('reset').onclick = () => resetView(lastMedian);

// ---- data ----
let n=0, cap=0, pos, col, lastMedian=0, fitted=false;
function hue(h){ // h 0..1 -> rgb (red..blue), near = warm, far = cool
  const k=n=>(n+h*6)%6, f=n=>255*(1-Math.max(0,Math.min(k(n),4-k(n),1))); return [f(5),f(3),f(1)]; }
function apply(buf){
  const cnt = new DataView(buf).getUint32(0, true);
  const xyz = new Float32Array(buf, 4, cnt*3), rgb = new Uint8Array(buf, 4+cnt*12, cnt*3);
  if (cnt > cap){ cap = Math.ceil(cnt*1.25); pos = new Float32Array(cap*3); col = new Uint8Array(cap*3); }
  const byDepth = $('mode').value === 'depth', byHeight = $('mode').value === 'height', far = +$('maxd').value;
  for (let j=0; j<cnt*3; j+=3){
    pos[j]=xyz[j]; pos[j+1]=-xyz[j+1]; pos[j+2]=-xyz[j+2];   // camera frame (y down, z fwd) -> y up, -z fwd
    if (byHeight && plane){ const H = plane.n[0]*xyz[j]+plane.n[1]*xyz[j+1]+plane.n[2]*xyz[j+2]+plane.d;
      if (H < 0.005){ col[j]=col[j+1]=col[j+2]=H < -0.02 ? 25 : 60; }
      else { const c=hue(0.66*(1-Math.min(1,(H-0.005)/0.095))); col[j]=c[0]; col[j+1]=c[1]; col[j+2]=c[2]; } }
    else if (byDepth){ const c=hue(0.66*Math.min(1, xyz[j+2]/far)); col[j]=c[0]; col[j+1]=c[1]; col[j+2]=c[2]; }
    else { col[j]=rgb[j]; col[j+1]=rgb[j+1]; col[j+2]=rgb[j+2]; }
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.bufferData(gl.ARRAY_BUFFER, pos.subarray(0, cnt*3), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.bufferData(gl.ARRAY_BUFFER, col.subarray(0, cnt*3), gl.DYNAMIC_DRAW);
  n = cnt;
  if (cnt){ const zs = []; for (let i=0;i<cnt;i+=Math.max(1,cnt>>9)) zs.push(xyz[i*3+2]); zs.sort((a,b)=>a-b);
    lastMedian = zs[zs.length>>1]; if (!fitted){ fitted=true; resetView(lastMedian); } }
  return cnt;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- table plane (fitted by the server) + measurement marks, refreshed every second ----
let plane=null, marks=[], gridN=0;
const gP = gl.createBuffer(), gC = gl.createBuffer(), mP = gl.createBuffer(), mC = gl.createBuffer();
const flip = p => [p[0], -p[1], -p[2]];       // camera frame -> view frame, as for the points
function buildGrid(){
  // 5 cm grid, 80x80 cm, centred under the image centre ray, lying in the table plane
  const n = plane.n, d = plane.d, t = d / Math.max(0.2, -n[2]);   // ray (0,0,1)*t hits n.p+d=0 at t=-d/n_z
  const c = [0, 0, Math.abs(t)];
  const proj = p => { const h = dot(n,p)+d; return [p[0]-n[0]*h, p[1]-n[1]*h, p[2]-n[2]*h]; };
  const o = proj(c), u = nrm(cross(n, [0,1,0])), v = cross(n, u);
  const P = [], C = [], R = 0.4;
  for (let k=-8; k<=8; k++){
    const a = k*0.05, major = k%2===0 ? 1 : 0.55;
    for (const [A,B] of [[u,v],[v,u]]){
      const p1 = [o[0]+A[0]*a-B[0]*R, o[1]+A[1]*a-B[1]*R, o[2]+A[2]*a-B[2]*R];
      const p2 = [o[0]+A[0]*a+B[0]*R, o[1]+A[1]*a+B[1]*R, o[2]+A[2]*a+B[2]*R];
      P.push(...flip(p1), ...flip(p2)); const g = Math.round(150*major); C.push(40,g,g+30, 40,g,g+30);
    }
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, gP); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, gC); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(C), gl.STATIC_DRAW);
  gridN = P.length/3;
}
function buildMarks(){
  const P = [], C = [];
  for (const m of marks){ P.push(...flip(m.below), ...flip(m.xyz)); C.push(255,220,60, 255,220,60);
    // small cross at the top
    const t = flip(m.xyz), e = 0.008;
    P.push(t[0]-e,t[1],t[2], t[0]+e,t[1],t[2], t[0],t[1],t[2]-e, t[0],t[1],t[2]+e); C.push(...Array(12).fill(255)); }
  gl.bindBuffer(gl.ARRAY_BUFFER, mP); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, mC); gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(C), gl.DYNAMIC_DRAW);
}
const cm = v => (v*100).toFixed(1)+' cm';
(async function sideLoop(){
  for(;;){
    try{
      const t = await (await fetch('/table', {cache:'no-store'})).json();
      if (t.normal){ plane = {n:t.normal, d:t.d}; buildGrid();
        $('hud').innerHTML = `table found · camera ${cm(t.camera_height_m)} above it<br>grid 5 cm` +
          ($('mode').value==='height' ? `<br>height <span class="bar"></span> 10 cm` : ''); }
      else { plane = null; gridN = 0; $('hud').textContent = 'no table plane in view'; }
      marks = (await (await fetch('/marks', {cache:'no-store'})).json()).marks || []; buildMarks();
    }catch(e){}
    await sleep(1000);
  }
})();
function placeLabels(M){
  const box = $('labels'), w = cv.clientWidth, h = cv.clientHeight;
  while (box.children.length < marks.length) box.appendChild(Object.assign(document.createElement('div'), {className:'tag'}));
  [...box.children].forEach((el, i) => {
    const m = marks[i]; if (!m){ el.style.display='none'; return; }
    const p = flip(m.xyz), x = M[0]*p[0]+M[4]*p[1]+M[8]*p[2]+M[12], y = M[1]*p[0]+M[5]*p[1]+M[9]*p[2]+M[13],
          wq = M[3]*p[0]+M[7]*p[1]+M[11]*p[2]+M[15];
    if (wq <= 0){ el.style.display='none'; return; }
    el.style.display=''; el.style.left = ((x/wq+1)/2*w)+'px'; el.style.top = ((1-y/wq)/2*h)+'px';
    el.textContent = cm(m.h) + ' up' + (m.above != null ? ` · ${cm(m.above)} above surface` : '');
  });
}
let frames=0, t0=performance.now();
(async function loop(){
  for(;;){
    if ($('pause').checked){ await sleep(200); continue; }
    try{
      const r = await fetch(`/cloud.bin?step=${$('step').value}&max=${$('maxd').value}`, {cache:'no-store'});
      if (r.status !== 200){ $('stat').textContent = 'no depth frame yet ('+r.status+')'; await sleep(500); continue; }
      const cnt = apply(await r.arrayBuffer()); frames++;
      const dt=(performance.now()-t0)/1000;
      if (dt>1){ $('stat').textContent = $('mini').textContent = `${cnt.toLocaleString()} points · ${(frames/dt).toFixed(1)} fps · drag to rotate`; frames=0; t0=performance.now(); }
    }catch(e){ $('stat').textContent = 'error: '+e.message; await sleep(1000); }
  }
})();
$('maxd').oninput = () => $('maxv').textContent = (+$('maxd').value).toFixed(1)+' m'; $('maxd').oninput();

function draw(){
  const w=Math.round(cv.clientWidth*devicePixelRatio), h=Math.round(cv.clientHeight*devicePixelRatio);
  if (cv.width!==w || cv.height!==h){ cv.width=w; cv.height=h; }
  gl.viewport(0,0,w,h); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const M = matrix(); gl.uniformMatrix4fv(uM, false, M); gl.uniform1f(uS, +$('psize').value*devicePixelRatio);
  const bind=(pb,cb)=>{ gl.bindBuffer(gl.ARRAY_BUFFER,pb); gl.vertexAttribPointer(aP,3,gl.FLOAT,false,0,0);
                         gl.bindBuffer(gl.ARRAY_BUFFER,cb); gl.vertexAttribPointer(aC,3,gl.UNSIGNED_BYTE,true,0,0); };
  if (n){ bind(pBuf,cBuf); gl.drawArrays(gl.POINTS, 0, n); }
  bind(axP,axC); gl.drawArrays(gl.LINES, 0, 6);
  if (gridN){ bind(gP,gC); gl.drawArrays(gl.LINES, 0, gridN); }
  if (marks.length){ gl.disable(gl.DEPTH_TEST); bind(mP,mC); gl.drawArrays(gl.LINES, 0, marks.length*6); gl.enable(gl.DEPTH_TEST); }
  placeLabels(M);
  requestAnimationFrame(draw);
}
draw();
</script></body></html>
"""
