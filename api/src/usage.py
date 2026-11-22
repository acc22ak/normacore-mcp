"""Token use and estimated cost of every Claude call, for the web panel.

Each call is appended to logs/usage.jsonl; totals are kept for the last request, this session and
today (today is re-read from the log at start, so it survives restarts).

Prices are US$ per million tokens (Claude API list prices, from the claude-api reference). The
cache-write price for Opus 5.5 isn't listed there: 1.25x input is assumed (the usual 5-minute
cache-write rate). The figures are estimates; the Anthropic Console is the bill.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "usage.jsonl"

PRICES = {   # input, output, cache read, cache write ($/MTok)
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
    # OpenAI: cache reads $0.40; cache write $10 (30-min) - aggregator figures, Nov 2026
    "gpt-6.2-astra": (8.00, 40.00, 0.40, 10.00),
}
DEFAULT = "claude-opus-5-5"
FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def price_for(model: str) -> tuple[float, float, float, float]:
    for name, p in PRICES.items():   # served ids can carry a date suffix
        if model and model.startswith(name):
            return p
    return PRICES[DEFAULT]


def cost(model: str, usage: dict) -> float:
    pi, po, pr, pw = price_for(model)
    return (usage.get("input_tokens", 0) * pi + usage.get("output_tokens", 0) * po
            + usage.get("cache_read_input_tokens", 0) * pr
            + usage.get("cache_creation_input_tokens", 0) * pw) / 1e6


def _zero() -> dict:
    return {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "usd": 0.0}


def _add(tot: dict, usage: dict, usd: float) -> None:
    tot["calls"] += 1
    tot["input"] += usage.get("input_tokens", 0)
    tot["output"] += usage.get("output_tokens", 0)
    tot["cache_read"] += usage.get("cache_read_input_tokens", 0)
    tot["cache_write"] += usage.get("cache_creation_input_tokens", 0)
    tot["usd"] += usd


class Meter:
    def __init__(self, log: Path = LOG):
        self.log = log
        self.session, self.today, self.request = _zero(), _zero(), _zero()
        self.last: dict | None = None
        self.day = time.strftime("%Y-%m-%d")
        try:
            for line in self.log.read_text().splitlines():
                r = json.loads(line)
                if r.get("when", "").startswith(self.day):
                    _add(self.today, r["usage"], r["usd"])
        except (OSError, ValueError, KeyError):
            pass

    def new_request(self) -> None:
        """A new instruction from the person: 'this request' starts again from zero."""
        self.request = _zero()

    def add(self, kind: str, model: str, usage) -> float:
        """Record one API call. usage = the SDK's Usage object or a dict. Returns its cost in US$."""
        u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
        u = {k: int(u.get(k) or 0) for k in FIELDS}
        usd = cost(model, u)
        if time.strftime("%Y-%m-%d") != self.day:
            self.day, self.today = time.strftime("%Y-%m-%d"), _zero()
        for tot in (self.session, self.today, self.request):
            _add(tot, u, usd)
        self.last = {"kind": kind, "model": model, **u, "usd": usd}
        try:
            self.log.parent.mkdir(exist_ok=True)
            with self.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "model": model,
                                    "usage": u, "usd": round(usd, 6)}) + "\n")
        except OSError:
            pass
        return usd

    def summary(self) -> dict:
        r = lambda t: {**t, "usd": round(t["usd"], 4)}
        return {"request": r(self.request), "session": r(self.session), "today": r(self.today),
                "last": self.last and {**self.last, "usd": round(self.last["usd"], 4)},
                "model": (self.last or {}).get("model", DEFAULT), "prices_per_mtok": price_for((self.last or {}).get("model", DEFAULT))}


METER = Meter()


def fmt(usage: dict, usd: float) -> str:
    """'3.1k in (2.4k cached) · 180 out · $0.0123'"""
    k = lambda n: f"{n / 1000:.1f}k" if n >= 1000 else str(n)
    cached = usage.get("cache_read_input_tokens", 0)
    tot_in = usage.get("input_tokens", 0) + cached + usage.get("cache_creation_input_tokens", 0)
    return f"{k(tot_in)} in" + (f" ({k(cached)} cached)" if cached else "") + \
        f" · {k(usage.get('output_tokens', 0))} out · ${usd:.4f}"
