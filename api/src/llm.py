"""Which model answers, and one structured-output call that works for Claude or GPT.

Models are switched from the web panel (POST /model) and the choice is remembered in
logs/model.json. Both providers get the same system prompt, images, text and JSON schema, and
return the same thing: the JSON text (streamed to on_text as it arrives), token usage in the
Anthropic shape (input_tokens excludes cached tokens), the model that answered, and whether the
answer finished normally.

Keys come from .env: ANTHROPIC_API_KEY for Claude, OPENAI_API_KEY for GPT.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

STATE = Path(__file__).resolve().parents[1] / "logs" / "model.json"


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    provider: str           # anthropic | openai
    timeout_s: float        # one command-mode answer; goal turns get 1.5x (more images)


MODELS = {m.id: m for m in (
    Model("claude-fable-5-2", "Claude Fable 5.2", "anthropic", 25.0),
    Model("claude-opus-5-5", "Claude Opus 5.5", "anthropic", 12.0),
    Model("gpt-6.2-astra", "GPT-6.2 Astra", "openai", 25.0),
)}
DEFAULTS = {"anthropic": "claude-opus-5-5", "openai": "gpt-6.2-astra"}
EFFORT = "low"


def available(m: Model) -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY" if m.provider == "anthropic" else "OPENAI_API_KEY"))


def current() -> Model:
    try:
        mid = json.loads(STATE.read_text()).get("model")
        if mid in MODELS:
            return MODELS[mid]
    except (OSError, ValueError):
        pass
    return MODELS[DEFAULTS["anthropic"]]


def select(mid: str) -> Model:
    if mid not in MODELS:
        raise ValueError(f"unknown model {mid!r}")
    m = MODELS[mid]
    if not available(m):
        raise ValueError(f"no {'OPENAI' if m.provider == 'openai' else 'ANTHROPIC'}_API_KEY in .env")
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"model": mid}))
    return m


def listing() -> list[dict]:
    cur = current().id
    return [{"id": m.id, "label": m.label, "provider": m.provider, "available": available(m), "current": m.id == cur}
            for m in MODELS.values()]


class LLMError(Exception):
    """API failure of either provider, with a short readable reason."""


@dataclass
class Answer:
    text: str
    usage: dict              # input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
    model: str
    ok: bool                 # finished normally (else: refusal / cut off / incomplete)
    stop: str                # provider's reason when not ok


_clients: dict = {}


def _client(provider: str, timeout_s: float):
    key = (provider, timeout_s)
    if key not in _clients:
        if provider == "anthropic":
            _clients[key] = anthropic.AsyncAnthropic(timeout=timeout_s, max_retries=1)
        else:
            import openai
            _clients[key] = openai.AsyncOpenAI(timeout=timeout_s, max_retries=1)
    return _clients[key]


async def ask(system: str, parts: list, schema: dict, on_text: Callable[[str], None] | None = None,
              model: Model | None = None, timeout_s: float | None = None, max_tokens: int = 4000) -> Answer:
    """parts: a list of str (text) and bytes (JPEG images), in order. Claude caches the system
    prompt between calls; OpenAI caches repeated prefixes by itself."""
    m = model or current()
    t = timeout_s or m.timeout_s
    try:
        if m.provider == "anthropic":
            return await _ask_claude(m, t, system, parts, schema, on_text, max_tokens)
        return await _ask_gpt(m, t, system, parts, schema, on_text, max_tokens)
    except anthropic.APIError as e:
        raise LLMError(f"{m.label}: {type(e).__name__}: {str(e)[:200]}") from e
    except Exception as e:
        if type(e).__module__.startswith("openai"):
            raise LLMError(f"{m.label}: {type(e).__name__}: {str(e)[:200]}") from e
        raise


async def _ask_claude(m, t, system, parts, schema, on_text, max_tokens) -> Answer:
    content = []
    for p in parts:
        if isinstance(p, bytes):
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(p).decode()}})
        else:
            content.append({"type": "text", "text": p})
    buf = ""
    async with _client("anthropic", t).beta.messages.stream(
            model=m.id, max_tokens=max_tokens,
            # the system prompt is the same on every call: cache it (camera images change every time)
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default") as stream:
        async for text in stream.text_stream:
            buf += text
            if on_text:
                on_text(text)
        msg = await stream.get_final_message()
    u = msg.usage.model_dump()
    usage = {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                             "cache_creation_input_tokens")}
    return Answer(buf, usage, msg.model, msg.stop_reason == "end_turn", str(msg.stop_reason))


async def _ask_gpt(m, t, system, parts, schema, on_text, max_tokens) -> Answer:
    content = []
    for p in parts:
        if isinstance(p, bytes):
            content.append({"type": "input_image", "detail": "auto",
                            "image_url": "data:image/jpeg;base64," + base64.standard_b64encode(p).decode()})
        else:
            content.append({"type": "input_text", "text": p})
    stream = await _client("openai", t).responses.create(
        model=m.id, instructions=system, input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "arm_reply", "schema": schema, "strict": True}},
        reasoning={"effort": EFFORT}, max_output_tokens=max_tokens, stream=True, store=False)
    buf, final = "", None
    async for ev in stream:
        if ev.type == "response.output_text.delta":
            buf += ev.delta
            if on_text:
                on_text(ev.delta)
        elif ev.type in ("response.completed", "response.incomplete", "response.failed"):
            final = ev.response
        elif ev.type == "error":
            raise LLMError(f"{m.label}: {getattr(ev, 'message', ev)}")
    if final is None:
        raise LLMError(f"{m.label}: the stream ended without a final response")
    u = final.usage
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0 if u else 0
    written = getattr(getattr(u, "input_tokens_details", None), "cache_write_tokens", 0) or 0 if u else 0
    usage = {"input_tokens": max(0, (u.input_tokens if u else 0) - cached - written),
             "output_tokens": u.output_tokens if u else 0,      # includes reasoning tokens (billed as output)
             "cache_read_input_tokens": cached, "cache_creation_input_tokens": written}
    ok = final.status == "completed"
    reason = final.status
    if not ok and getattr(final, "incomplete_details", None):
        reason = f"{final.status}: {final.incomplete_details.reason}"
    return Answer(buf, usage, final.model or m.id, ok, reason)
