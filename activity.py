"""Live feed of what the connected agent is doing, for the station-viewer's "agent" panel.

The MCP server reports each tool call here (start and end) and the client that connected. Browsers
read it as Server-Sent Events from GET /events on the MCP server's HTTP port: on connect they get the
current agent and the recent calls, then every new event as it happens. Display only; nothing reads
it back, so a slow or missing viewer never affects the robot.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from collections import deque
from typing import Any

# MCP clients say which app they are (e.g. claude-code 2.1), never which model drives them, so the
# model name shown in the panel is configured here.
AGENT_LABEL = os.environ.get("MCP_AGENT_LABEL", "Claude Opus 5.5")
RECENT_CALLS = 30
ARGS_MAX_CHARS = 90
HEARTBEAT_S = 15.0

_ids = itertools.count(1)
_recent: deque[dict] = deque(maxlen=RECENT_CALLS)
_subscribers: set[asyncio.Queue] = set()
_agent: dict[str, Any] = {"label": AGENT_LABEL, "client": None, "version": None, "seen": None}


def _publish(event: dict) -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass  # a stalled browser just misses events


def note_client(name: str | None, version: str | None) -> None:
    """Record the MCP client of the current request (announced once per change)."""
    changed = (name, version) != (_agent["client"], _agent["version"])
    _agent.update(client=name, version=version, seen=time.time())
    if changed:
        _publish({"type": "agent", **_agent})


def _short_args(arguments: dict[str, Any]) -> str:
    parts = [f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in arguments.items() if v is not None]
    text = ", ".join(parts)
    return text if len(text) <= ARGS_MAX_CHARS else text[: ARGS_MAX_CHARS - 1] + "…"


def call_started(tool: str, arguments: dict[str, Any]) -> int:
    call = {"id": next(_ids), "tool": tool, "args": _short_args(arguments or {}), "started": time.time(),
            "status": "running", "ms": None}
    _recent.append(call)
    _publish({"type": "call", **call})
    return call["id"]


def call_finished(call_id: int, ok: bool) -> None:
    for call in reversed(_recent):
        if call["id"] == call_id:
            call["status"] = "ok" if ok else "failed"
            call["ms"] = round((time.time() - call["started"]) * 1000)
            _publish({"type": "call", **call})
            return


def result_ok(result: Any) -> bool:
    """False when a tool reported failure in its JSON ("success": false), else True."""
    # FastMCP returns content blocks, or (content blocks, structured dict) for tools with an output schema.
    blocks: list = []
    for part in result if isinstance(result, (list, tuple)) else [result]:
        blocks.extend(part if isinstance(part, (list, tuple)) else [part])
    for block in blocks:
        text = getattr(block, "text", None) if not isinstance(block, (str, dict)) else block
        if isinstance(text, dict):
            return text.get("success") is not False
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except ValueError:
                return True
            return not (isinstance(data, dict) and data.get("success") is False)
    return True


async def events_endpoint(request):
    """GET /events: Server-Sent Events (snapshot first, then live)."""
    from starlette.responses import StreamingResponse

    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.add(queue)

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'snapshot', 'agent': _agent, 'calls': list(_recent)})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_S)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield 'data: {"type": "ping"}\n\n'  # an event, not a comment: the page's watchdog sees it
        finally:
            _subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",  # the viewer runs on another port (Vite :5173)
    })


async def clear_endpoint(request):
    """POST /events/clear: forget the recent calls (the panel's Clear button, e.g. before recording)."""
    from starlette.responses import JSONResponse

    _recent.clear()
    _publish({"type": "clear"})
    return JSONResponse({"cleared": True}, headers={"Access-Control-Allow-Origin": "*"})
