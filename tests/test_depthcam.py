"""Offline test for depthcam.py: pure helpers, the daemon's HTTP API with fake frames, and the MCP
tools talking to it. No camera needed.

usage: python tests/test_depthcam.py <normacore-mcp dir>
"""
import asyncio, json, os, sys, threading, time, urllib.request
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
os.environ["DEPTHCAM_PORT"] = "8799"
os.environ["DEPTHCAM_URL"] = "http://127.0.0.1:8799"
import numpy as np
import depthcam as D
from http.server import ThreadingHTTPServer

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# --- pure helpers ---
depth = np.zeros((48, 64), np.uint16)
depth[20:30, 30:40] = 700          # a 0.7 m patch (scale 0.001)
depth[25, 35] = 0                  # one hole inside it
z, n = D.median_depth(depth, 35, 25, 2, 0.001)
check("median ignores zero (no-depth) pixels", abs(z - 0.7) < 1e-9 and n == 24, (z, n))
check("no valid depth -> None", D.median_depth(depth, 2, 2, 1, 0.001) == (None, 0))
try:
    D.median_depth(depth, 64, 0, 1, 0.001); check("pixel outside frame rejected", False)
except ValueError:
    check("pixel outside frame rejected", True)
intr = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}
check("principal point deprojects to the optical axis", D.deproject(intr, 320, 240, 1.0) == [0.0, 0.0, 1.0])
check("deproject x/y scale with depth", D.deproject(intr, 380, 180, 2.0) == [0.2, -0.2, 2.0])

# --- daemon HTTP API with a fake frame source ---
store = D.FrameStore()
store.info = {"name": "Fake D435", "serial": "123", "width": 64, "height": 48, "fps": 30,
              "depth_scale_m": 0.001, "intrinsics": {"fx": 60.0, "fy": 60.0, "ppx": 32.0, "ppy": 24.0}}
httpd = ThreadingHTTPServer(("127.0.0.1", 8799), D.make_handler(store))
httpd.daemon_threads = True
threading.Thread(target=httpd.serve_forever, daemon=True).start()

def get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8799" + path, timeout=3) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")

code, body, _ = get("/color.jpg")
check("no frame yet -> 503", code == 503, code)
color = np.full((48, 64, 3), 200, np.uint8)
store.put(color, depth, np.zeros((48, 64, 3), np.uint8))
code, body, ctype = get("/color.jpg")
check("colour JPEG served", code == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8", (code, ctype))
check("depth JPEG served", get("/depth.jpg")[0] == 200)
code, body, _ = get("/depth_at?x=35&y=25&r=2")
j = json.loads(body)
check("depth_at returns metres + 3D point", code == 200 and abs(j["depth_m"] - 0.7) < 1e-6 and j["point_m"] == [0.035, 0.0117, 0.7], j)
code, body, _ = get("/depth_at?x=999&y=1")
check("depth_at outside frame -> 400", code == 400, code)
check("depth_at without x/y -> 400", get("/depth_at")[0] == 400)
check("page served", b"/stream/color" in get("/")[1])
st = json.loads(get("/status")[1])
check("status has frame count and age", st["frames"] == 1 and st["age_s"] < 1.0, st)

# MJPEG stream: read a couple of frames while new ones arrive
def feeder():
    for _ in range(5):
        time.sleep(0.05)
        store.put(color, depth, np.zeros((48, 64, 3), np.uint8))
threading.Thread(target=feeder, daemon=True).start()
with urllib.request.urlopen("http://127.0.0.1:8799/stream/color", timeout=3) as r:
    data = r.read(4000)
check("MJPEG stream sends multipart frames", r.headers.get_content_type() == "multipart/x-mixed-replace" and data.count(b"--frame") >= 2, data[:80])

# --- MCP tools against the daemon ---
async def mcp_tests():
    m = D.build_mcp()
    tools = sorted(t.name for t in await m.list_tools())
    check("MCP tools registered", tools == ["depth_at", "depthcam_status", "look_depth"], tools)
    store.put(color, depth, np.zeros((48, 64, 3), np.uint8))
    out = await m.call_tool("look_depth", {})
    items = out[0] if isinstance(out, tuple) else out
    check("look_depth: label + colour + depth images", len(items) == 3 and "Fake D435" in items[0].text
          and "STALE" not in items[0].text and items[1].type == "image" and items[2].type == "image", items[0].text)
    store.stamp = time.time() - 10
    items = await m.call_tool("look_depth", {})
    items = items[0] if isinstance(items, tuple) else items
    check("old frame flagged STALE", "STALE" in items[0].text, items[0].text)
    r = await m.call_tool("depth_at", {"x": 35, "y": 25})
    data = r[1] if isinstance(r, tuple) else json.loads(r[0].text)
    data = data.get("result", data)
    check("depth_at tool", abs(data["depth_m"] - 0.7) < 1e-6, data)
    D.URL = "http://127.0.0.1:1"
    try:
        await m.call_tool("depthcam_status", {}); check("daemon down -> clear error", False)
    except Exception as e:
        check("daemon down -> clear error", "depthcam.py serve" in str(e), str(e))

asyncio.run(mcp_tests())
httpd.shutdown()
print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
