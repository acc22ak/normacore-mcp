"""Web panel: Station's own UI (unchanged, embedded from :8889) with a command panel beside it.

  .venv/bin/python src/app.py --web          # then open http://localhost:8890

The panel sends each line to the same App.on_line() the terminal uses, so 'stop', 'relax',
'record', 'done', 'repeat' and free-form requests behave exactly as typed in the terminal, and it
shows the same output lines, streamed live (server-sent events).
"""
from __future__ import annotations

import asyncio
import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import app as A
from usage import METER  # noqa: E402

STATION_PORT = 8889


def make_web(app: "A.App") -> Starlette:
    async def page(_: Request):
        return HTMLResponse(PAGE.replace("__STATION_PORT__", str(STATION_PORT)))

    async def say(req: Request):
        try:
            text = str((await req.json()).get("text", "")).strip()[:500]
        except (ValueError, AttributeError):
            return JSONResponse({"ok": False, "error": "bad request"}, status_code=400)
        if text:
            A.out(f"› {text}")
            app.on_line(text)
        return JSONResponse({"ok": True})

    async def events(req: Request):
        q: asyncio.Queue = asyncio.Queue()
        A.LISTENERS.append(q)

        async def stream():
            try:
                for line in list(A.HISTORY):
                    yield f"data: {json.dumps(line)}\n\n"
                while not await req.is_disconnected():
                    try:
                        line = await asyncio.wait_for(q.get(), 15)
                        yield f"data: {json.dumps(line)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                A.LISTENERS.remove(q)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store"})

    async def status(_: Request):
        busy = bool(app.task and not app.task.done())
        return JSONResponse({"busy": busy, "recording": app.recorder.recording, "io": app.io.name,
                             "bus": getattr(app.io, "follower", None),
                             "views": [v.name for v in app.views], "views_version": app.views_version,
                             "usage": METER.summary()})

    async def view(req: Request):
        name = req.path_params["name"]
        for v in app.views:
            if v.name == name:
                return Response(v.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        return Response(status_code=404)

    return Starlette(routes=[Route("/", page), Route("/say", say, methods=["POST"]),
                             Route("/events", events), Route("/status", status),
                             Route("/view/{name}.jpg", view)])


async def serve(app: "A.App", host: str, port: int) -> None:
    import uvicorn
    cfg = uvicorn.Config(make_web(app), host=host, port=port, log_level="warning", lifespan="off")
    await uvicorn.Server(cfg).serve()


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NormaCore station + arm chat</title>
<style>
  :root { --bg:#111827; --panel:#1a2230; --line:#2b3546; --fg:#e5e7eb; --dim:#9aa4b2;
          --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; --accent:#60a5fa; --say:#c4b5fd; }
  * { box-sizing:border-box }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
               font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif }
  .wrap { display:flex; height:100vh }
  iframe { flex:1; border:0; min-width:0; background:var(--bg) }
  #cloudpane { width:440px; display:flex; flex-direction:column; background:#111;
               border-left:1px solid var(--line); min-width:0 }
  #cloudpane.hidden { display:none }
  #cloudpane header { padding:8px 12px }
  #cloudpane iframe { flex:1; border:0; background:#111 }
  @media (max-width:1400px) { #cloudpane { width:340px } }
  aside { width:380px; display:flex; flex-direction:column; background:var(--panel);
          border-left:1px solid var(--line) }
  aside.hidden { display:none }
  header { display:flex; align-items:center; gap:8px; padding:10px 12px; border-bottom:1px solid var(--line) }
  header h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.02em }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--dim) }
  .dot.live { background:var(--ok) } .dot.busy { background:var(--warn) } .dot.rec { background:var(--bad) }
  #state { color:var(--dim); font-size:12px; margin-left:auto }
  #log { flex:1; overflow-y:auto; padding:10px 12px; font:12.5px/1.5 ui-monospace, "DejaVu Sans Mono", monospace;
         white-space:pre-wrap; word-break:break-word }
  #log div { padding:1px 0 }
  .you { color:var(--accent); margin-top:8px } .said { color:var(--say) } .ok { color:var(--ok) }
  .bad { color:var(--bad) } .warn { color:var(--warn) } .dim { color:var(--dim) }
  .quick { display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; padding:10px 12px;
           border-top:1px solid var(--line) }
  button { font:inherit; font-size:13px; color:var(--fg); background:#243044; border:1px solid var(--line);
           border-radius:6px; padding:7px 6px; cursor:pointer }
  button:hover { background:#2d3b52 }
  #stop { grid-column:1 / -1; background:#7f1d1d; border-color:#b91c1c; font-weight:700; letter-spacing:.06em;
          padding:10px }
  #stop:hover { background:#991b1b }
  form { display:flex; gap:6px; padding:0 12px 12px }
  input { flex:1; min-width:0; font:inherit; color:var(--fg); background:#0f1622; border:1px solid var(--line);
          border-radius:6px; padding:9px 10px }
  input:focus { outline:2px solid var(--accent); outline-offset:-1px }
  #toggle { position:fixed; right:10px; bottom:10px; z-index:5; display:none }
  .hint { color:var(--dim); font-size:11.5px; padding:0 12px 10px }
  #views { display:flex; gap:6px; padding:8px 12px 0; overflow-x:auto }
  #views:empty { display:none }
  #views figure { margin:0; flex:0 0 auto }
  #views img { height:92px; border-radius:4px; border:1px solid var(--line); display:block; cursor:zoom-in }
  #views figcaption { font-size:11px; color:var(--dim) }
  #views img.big { height:260px }
  #mode { white-space:nowrap }
  #mode.goal { background:#1e3a5f; border-color:var(--accent); color:#dbeafe }
  .goalline { color:#93c5fd } .done { color:var(--ok); font-weight:600 }
  #mic { width:40px; padding:0; font-size:16px }
  #mic.on { background:#7f1d1d; border-color:#ef4444; animation:pulse 1.1s ease-in-out infinite }
  #mic:disabled { opacity:.4; cursor:not-allowed }
  @keyframes pulse { 50% { box-shadow:0 0 0 5px rgba(239,68,68,.25) } }
  #cost { display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; padding:8px 12px 0;
          border-top:1px solid var(--line); font-variant-numeric:tabular-nums }
  #cost div { background:#0f1622; border:1px solid var(--line); border-radius:6px; padding:5px 7px; min-width:0 }
  #cost b { display:block; font-size:15px; font-weight:600 }
  #cost span { display:block; color:var(--dim); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
  #cost small { color:var(--dim); font-size:10.5px; text-transform:uppercase; letter-spacing:.06em }
</style></head>
<body>
<div class="wrap">
  <iframe id="station" title="NormaCore station"></iframe>
  <section id="cloudpane" aria-label="3D point cloud">
    <header><h1>3D · RealSense</h1><span id="cloudnote" class="dim" style="font-size:12px;margin-left:auto"></span>
      <button id="cloudhide" title="Hide the 3D view" style="padding:2px 8px">×</button></header>
    <iframe id="cloud" title="RealSense 3D point cloud"></iframe>
  </section>
  <aside id="panel">
    <header><span class="dot" id="dot"></span><h1>Arm chat</h1><span id="state">connecting…</span>
      <button id="cloudshow" title="Show the 3D view" style="padding:2px 8px">3D</button>
      <button id="hide" title="Hide panel" style="padding:2px 8px">›</button></header>
    <div id="views" title="What Claude saw on the last goal turn (click to enlarge)"></div>
    <div id="log" aria-live="polite"></div>
    <div id="cost" title="Estimated from token counts at API list prices; the Anthropic Console has the real bill">
      <div><small>This request</small><b id="c-req">$0.000</b><span id="c-req-t">–</span></div>
      <div><small>Session</small><b id="c-ses">$0.000</b><span id="c-ses-t">–</span></div>
      <div><small>Today</small><b id="c-day">$0.000</b><span id="c-day-t">–</span></div>
    </div>
    <div class="quick">
      <button id="stop" data-cmd="stop">STOP</button>
      <button data-cmd="open the gripper">Open gripper</button>
      <button data-cmd="close the gripper">Close gripper</button>
      <button data-cmd="relax">Relax</button>
      <button data-cmd="record">Record</button>
      <button data-cmd="done">Done</button>
      <button data-cmd="repeat">Repeat</button>
    </div>
    <form id="f"><button type="button" id="mic" title="Speak a request (click, talk, pause)">🎤</button>
      <input id="in" placeholder="Ask or tell the arm anything…" autocomplete="off">
      <button type="submit">Send</button></form>
    <div class="hint">Esc = stop · ↑ = previous request · 🎤 or Alt+V = speak · Claude answers, moves, or keeps looking and moving until a task is done</div>
  </aside>
</div>
<button id="toggle" title="Show panel">Arm chat ‹</button>
<script>
const station = document.getElementById("station");
station.src = location.protocol + "//" + location.hostname + ":__STATION_PORT__/";
// RealSense 3D view (depthcam daemon, listens on 127.0.0.1 only)
const cloudHost = location.hostname === "localhost" ? "127.0.0.1" : location.hostname;
const cloudPane = document.getElementById("cloudpane"), cloud = document.getElementById("cloud");
function setCloud(show) {
  cloudPane.classList.toggle("hidden", !show);
  if (show && !cloud.src) cloud.src = "http://" + cloudHost + ":8791/cloud?embed=1";
  if (!show) cloud.removeAttribute("src");   // stop fetching point clouds while hidden
  try { localStorage.setItem("cloud", show ? "1" : "0"); } catch {}
}
let cloudPref = "1"; try { cloudPref = localStorage.getItem("cloud") ?? "1"; } catch {}
setCloud(cloudPref !== "0");
document.getElementById("cloudhide").onclick = () => setCloud(false);
document.getElementById("cloudshow").onclick = () => setCloud(cloudPane.classList.contains("hidden"));
// another port = another site: the browser hides the reply, but a refused connection still fails
fetch("http://" + cloudHost + ":8791/status", {mode: "no-cors", cache: "no-store"})
  .then(() => document.getElementById("cloudnote").textContent = "")
  .catch(() => document.getElementById("cloudnote").textContent = "depth camera server not running");
const log = document.getElementById("log"), input = document.getElementById("in");
const dot = document.getElementById("dot"), state = document.getElementById("state");
const sent = []; let back = 0;

function cls(t) {
  if (t.startsWith("›")) return "you";
  if (t.startsWith("◎") || t.startsWith("👁")) return "goalline";
  if (t.startsWith("★")) return "done";
  if (t.trim().startsWith("“")) return "said";
  if (/✓/.test(t)) return "ok";
  if (/✗|■ Holding|Not recording|Can't/.test(t)) return "bad";
  if (/■|●|busy|WARNING/.test(t)) return "warn";
  if (/⏱|peak current|🎤/.test(t)) return "dim";
  return "";
}
function add(t) {
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  const d = document.createElement("div"); d.textContent = t; d.className = cls(t); log.appendChild(d);
  while (log.children.length > 400) log.firstChild.remove();
  if (stick) log.scrollTop = log.scrollHeight;
}
function send(text) {
  text = text.trim(); if (!text) return;
  fetch("/say", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})})
    .catch(() => add("✗ could not reach the app"));
}
function connect() {
  const es = new EventSource("/events");
  es.onmessage = e => add(JSON.parse(e.data));
  es.onerror = () => { dot.className = "dot"; state.textContent = "reconnecting…"; };
}
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    dot.className = "dot " + (s.recording ? "rec" : s.busy ? "busy" : "live"); showViews(s);
    state.textContent = (s.recording ? "recording · " : s.busy ? "working · " : "ready · ") + (s.bus || s.io);
    if (s.usage) showUsage(s.usage);
  } catch { dot.className = "dot"; state.textContent = "app not running"; }
}
document.getElementById("f").onsubmit = e => {
  e.preventDefault(); const t = input.value; if (!t.trim()) return;
  sent.push(t); back = sent.length;
  send(t); input.value = "";
};
let viewsVersion = -1; const viewsEl = document.getElementById("views");
function showViews(s) {
  if (s.views_version === viewsVersion) return; viewsVersion = s.views_version;
  viewsEl.textContent = "";
  for (const n of s.views) {
    const f = document.createElement("figure"), img = document.createElement("img"), c = document.createElement("figcaption");
    img.src = "/view/" + n + ".jpg?v=" + viewsVersion; img.alt = n + " camera"; img.onclick = () => img.classList.toggle("big");
    c.textContent = n; f.append(img, c); viewsEl.appendChild(f);
  }
}
input.addEventListener("keydown", e => {
  if (e.key === "ArrowUp" && back > 0) { input.value = sent[--back]; e.preventDefault(); }
  if (e.key === "ArrowDown") { back = Math.min(sent.length, back + 1); input.value = sent[back] || ""; }
});
document.addEventListener("keydown", e => { if (e.key === "Escape") send("stop"); });
document.querySelectorAll("[data-cmd]").forEach(b => b.onclick = () => send(b.dataset.cmd));
const panel = document.getElementById("panel"), toggle = document.getElementById("toggle");
document.getElementById("hide").onclick = () => { panel.classList.add("hidden"); toggle.style.display = "block"; };
toggle.onclick = () => { panel.classList.remove("hidden"); toggle.style.display = "none"; };
// ---- token use and estimated cost ----
const kt = n => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
const usd = v => v < 1 ? "$" + v.toFixed(3) : "$" + v.toFixed(2);
function showUsage(u) {
  for (const [id, t] of [["req", u.request], ["ses", u.session], ["day", u.today]]) {
    document.getElementById("c-" + id).textContent = usd(t.usd);
    const inTok = t.input + t.cache_read + t.cache_write;
    document.getElementById("c-" + id + "-t").textContent = t.calls
      ? `${kt(inTok)} in · ${kt(t.output)} out · ${t.calls} call${t.calls > 1 ? "s" : ""}` : "no calls";
  }
  const p = u.prices_per_mtok;
  document.getElementById("cost").title = `${u.model}: $${p[0]} in / $${p[1]} out / $${p[2]} cached per million tokens. ` +
    "Estimated from token counts at list prices; the Anthropic Console has the real bill.";
}

// ---- voice input: the browser turns speech into text, which is sent like a typed request ----
const mic = document.getElementById("mic");
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
if (!SR) {
  mic.disabled = true;
  mic.title = "Voice input needs Chrome or Edge (this browser has no speech recognition)";
} else {
  rec = new SR();
  rec.lang = navigator.language || "en-US"; rec.interimResults = true; rec.continuous = false;
  rec.onstart = () => { listening = true; mic.classList.add("on"); input.placeholder = "Listening…"; };
  rec.onend = () => { listening = false; mic.classList.remove("on"); input.placeholder = "Ask or tell the arm anything…"; };
  rec.onresult = e => {
    let text = "", final = false;
    for (const r of e.results) { text += r[0].transcript; final = final || r.isFinal; }
    input.value = text;
    if (final && text.trim()) {
      add("🎤 heard: " + text.trim());
      sent.push(text); back = sent.length; send(text); input.value = "";
    }
  };
  rec.onerror = e => {
    const why = {"not-allowed": "microphone blocked: allow it for this page in the browser",
                 "service-not-allowed": "microphone blocked: allow it for this page in the browser",
                 "network": "speech service unreachable (needs internet)",
                 "no-speech": "didn't hear anything", "audio-capture": "no microphone found"}[e.error] || e.error;
    add("✗ voice: " + why);
  };
  mic.onclick = () => { if (listening) rec.stop(); else { try { rec.start(); } catch {} } };
  document.addEventListener("keydown", e => { if (e.altKey && (e.key === "v" || e.key === "V")) { e.preventDefault(); mic.click(); } });
}

connect(); poll(); setInterval(poll, 1000); input.focus();
</script>
</body></html>
"""
