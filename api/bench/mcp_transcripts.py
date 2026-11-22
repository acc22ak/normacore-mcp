"""Timestamp past MCP runs from Claude Code transcripts (~/.claude/projects/*/*.jsonl).

An "episode" = one real user instruction and everything until the next one. For each
episode that used normacore tools we report:
  turns        assistant API calls (distinct message ids)
  model_s      time Claude spent generating (prev event -> assistant block logged)
  tool_s       tool_use -> tool_result (MCP + Station + any permission prompt wait)
  wall_s       instruction -> last event
Per tool we also report the distribution of tool_use -> tool_result.

Caveat: tool_s includes time waiting for the human to approve a permission prompt.
Calls longer than PROMPT_SUSPECT_S are flagged separately.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROMPT_SUSPECT_S = 20.0


def ts(e) -> float:
    return datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()


def is_real_user(e) -> bool:
    if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain"):
        return False
    c = e.get("message", {}).get("content")
    if isinstance(c, str):
        return not c.startswith("<")
    return any(b.get("type") == "text" and not b["text"].startswith("<") for b in c or [])


def episodes(path: Path):
    ev = []
    for l in path.open(errors="replace"):
        try:
            ev.append(json.loads(l))
        except ValueError:
            pass
    ev = [e for e in ev if "timestamp" in e and e.get("type") in ("user", "assistant") and not e.get("isSidechain")]
    cur = None
    for e in ev:
        if is_real_user(e):
            if cur:
                yield cur
            c = e["message"]["content"]
            text = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if b.get("type") == "text")
            cur = {"file": path.name, "start": ts(e), "prompt": text[:90].replace("\n", " "), "events": []}
        elif cur:
            cur["events"].append(e)
    if cur:
        yield cur


def analyze(ep):
    last_t = ep["start"]
    msg_ids, model_s, tool_s = set(), 0.0, 0.0
    pending = {}  # tool_use_id -> (name, t)
    calls = []
    for e in ep["events"]:
        t = ts(e)
        if e["type"] == "assistant":
            mid = e["message"].get("id")
            if mid not in msg_ids:
                msg_ids.add(mid)
                model_s += max(0.0, t - last_t)
            for b in e["message"].get("content", []):
                if b.get("type") == "tool_use":
                    pending[b["id"]] = (b["name"], t)
            last_t = t
        else:
            for b in e["message"].get("content", []) if isinstance(e["message"].get("content"), list) else []:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                    name, t0 = pending.pop(b["tool_use_id"])
                    calls.append((name, t - t0))
            last_t = t
    nc = [c for c in calls if c[0].startswith("mcp__normacore__") or c[0].startswith("mcp__depthcam__")]
    if not nc:
        return None
    tool_s = sum(d for _, d in calls)
    return {"file": ep["file"][:8], "prompt": ep["prompt"], "turns": len(msg_ids),
            "tool_calls": len(calls), "normacore_calls": len(nc),
            "wall_s": round(last_t - ep["start"], 1), "model_s": round(model_s, 1),
            "tool_s": round(tool_s, 1), "calls": calls}


def main(paths):
    rows, per_tool = [], defaultdict(list)
    for p in paths:
        for ep in episodes(Path(p)):
            r = analyze(ep)
            if r:
                rows.append(r)
                for n, d in r["calls"]:
                    per_tool[n.replace("mcp__normacore__", "nc.").replace("mcp__depthcam__", "dc.")].append(d)
    rows.sort(key=lambda r: -r["wall_s"])
    print(f"{len(rows)} episodes with normacore/depthcam calls\n")
    print(f"{'session':8} {'turns':>5} {'calls':>5} {'wall_s':>7} {'model_s':>7} {'tool_s':>7}  prompt")
    for r in rows[:25]:
        print(f"{r['file']:8} {r['turns']:>5} {r['tool_calls']:>5} {r['wall_s']:>7} {r['model_s']:>7} {r['tool_s']:>7}  {r['prompt'][:70]}")
    # per-turn model time = model_s / turns
    pt = [r["model_s"] / r["turns"] for r in rows if r["turns"]]
    print(f"\nmodel seconds per turn: median {statistics.median(pt):.1f}  (over {len(pt)} episodes)")
    print(f"turns per episode: median {statistics.median(r['turns'] for r in rows)}")
    print(f"\n{'tool':28} {'n':>4} {'median_s':>9} {'p90_s':>7} {'>'+str(int(PROMPT_SUSPECT_S))+'s':>5}")
    for n, ds in sorted(per_tool.items(), key=lambda kv: -len(kv[1])):
        if not (n.startswith("nc.") or n.startswith("dc.")):
            continue
        ds2 = sorted(ds)
        p90 = ds2[min(len(ds2) - 1, int(round(0.9 * (len(ds2) - 1))))]
        print(f"{n:28} {len(ds):>4} {statistics.median(ds):>9.2f} {p90:>7.2f} {sum(d > PROMPT_SUSPECT_S for d in ds):>5}")
    out = Path(__file__).resolve().parents[1] / "logs" / "mcp_transcripts.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([{k: v for k, v in r.items()} for r in rows], indent=1))
    print("\nsaved", out)


if __name__ == "__main__":
    main(sys.argv[1:])
