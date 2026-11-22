"""Step 0 of plan.html: raw I/O cost, MCP vs direct Python, no model.

Read-only: never sends a motor command.

Arms (N trials each, median / p90 in ms):
  direct    station_py read straight from Station queues (what station_io.py would do)
  handler   the MCP server's own tool functions, called in-process (no transport)
  mcp       the same tools over MCP stdio, server spawned exactly like Claude Code spawns it
  net       TCP+TLS connect and one HTTPS round trip to api.anthropic.com (no key, 401 expected)

Usage:  ~/normacore-mcp/.venv/bin/python bench/measure_io.py [--n 30] [--out logs/io_bench.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import ssl
import statistics
import sys
import time
from pathlib import Path

MCP_DIR = Path(__file__).resolve().parents[2]
ENV = {
    "NORMA_CORE_REPO": str(Path.home() / "norma-core"),
    "STATION_HOST": "localhost",
    "FOLLOWER_BUS": "5B8E115950",
    "LEADER_BUS": "5B91043428",
}
os.environ.update(ENV)
sys.path.insert(0, str(MCP_DIR))


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 2), "p90_ms": round(p90, 2),
            "min_ms": round(xs[0], 2), "max_ms": round(xs[-1], 2)}


async def timeit(fn, n: int, warmup: int = 2) -> tuple[list[float], object]:
    last = None
    for _ in range(warmup):
        last = await fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        last = await fn()
        out.append((time.perf_counter() - t) * 1000)
    return out, last


async def bench_inprocess(n: int) -> tuple[dict, dict]:
    import server as S  # the MCP server module; importing it does not start the server

    cams, _ = await S.cameras()
    live = [c for c in cams if c["jpeg"] is not None]
    info = {"cameras": [{"queue": c["queue"], "product": c["product"], "age_s": c["age_s"],
                         "jpeg_bytes": len(c["jpeg"]) if c["jpeg"] else 0} for c in cams]}
    if not live:
        raise SystemExit("no camera with frames; is the gripper cam plugged in?")
    cam = next((c for c in live if "FHD C3" in (c["product"] or "")), live[0])
    info["bench_camera"] = cam["product"]
    q = cam["queue"]

    res = {}
    # direct: one joint snapshot, one camera entry (what a lean station_io would read)
    res["direct.joints(1 entry)"], _ = await timeit(lambda: S.read_last("st3215/inference", 1), n)
    res["direct.camera(1 entry)"], _ = await timeit(lambda: S.read_last(q, 1), n)
    # handler: the tool bodies, in process
    res["handler.get_robot_state"], _ = await timeit(S.get_robot_state, n)
    res["handler.look(cam)"], out = await timeit(lambda: S.look(cam["camera_id"]), n)
    res["handler.look(all)"], _ = await timeit(lambda: S.look("all"), n)
    return {k: stats(v) for k, v in res.items()}, {**info, "look_cam_id": cam["camera_id"]}


async def bench_mcp(n: int, cam_id: str) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=str(MCP_DIR / ".venv/bin/python"),
                                   args=[str(MCP_DIR / "server.py")], env={**os.environ, **ENV})
    res = {}
    t0 = time.perf_counter()
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res["mcp.spawn+initialize"] = [(time.perf_counter() - t0) * 1000]
            t = time.perf_counter()
            await s.list_tools()
            res["mcp.list_tools"] = [(time.perf_counter() - t) * 1000]

            async def call(name, args):
                r = await s.call_tool(name, args)
                if r.isError:
                    raise RuntimeError(f"{name}: {r.content}")
                return r

            res["mcp.get_robot_state"], _ = await timeit(lambda: call("get_robot_state", {}), n)
            res["mcp.look(cam)"], last = await timeit(lambda: call("look", {"camera": cam_id}), n)
            res["mcp.look(all)"], _ = await timeit(lambda: call("look", {"camera": "all"}), n)
    return {k: stats(v) for k, v in res.items()}


def bench_net(n: int, host: str = "api.anthropic.com") -> dict:
    """Connect cost and warm round trip. No API key is sent; a 401 is the expected answer."""
    import http.client
    connect, rtt = [], []
    for _ in range(n):
        c = http.client.HTTPSConnection(host, timeout=10)
        t = time.perf_counter(); c.connect(); connect.append((time.perf_counter() - t) * 1000)
        for i in range(3):  # requests 2 and 3 reuse the warm connection
            t = time.perf_counter()
            c.request("GET", "/v1/models"); r = c.getresponse(); r.read()
            if i:
                rtt.append((time.perf_counter() - t) * 1000)
        c.close()
    return {"net.tcp+tls_connect": stats(connect), "net.https_rtt_warm": stats(rtt), "net.status": r.status}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "logs" / "io_bench.json"))
    a = ap.parse_args()

    results, info = await bench_inprocess(a.n)
    results.update(await bench_mcp(a.n, info["look_cam_id"]))
    try:
        results.update(bench_net(min(a.n, 15)))
    except OSError as e:
        results["net.error"] = str(e)

    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n": a.n, "info": info, "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2))
    w = max(len(k) for k in results)
    print(f"{'arm':<{w}}  {'median':>8}  {'p90':>8}  {'n':>3}")
    for k, v in results.items():
        if isinstance(v, dict) and v.get("n"):
            print(f"{k:<{w}}  {v['median_ms']:>8.1f}  {v['p90_ms']:>8.1f}  {v['n']:>3}")
        else:
            print(k, v)
    print(json.dumps(info, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    asyncio.run(main())
