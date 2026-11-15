"""One place for LLM access: provider switch, JSON-schema output, graceful failure (issue #26).

Provider is chosen by `FIF_LLM=gemini|anthropic|none`; when unset: gemini if GEMINI_API_KEY is set,
else anthropic if ANTHROPIC_API_KEY is set, else none. With none, `json_call` raises `LLMUnavailable`
and every caller falls back to the extractive / rule-based path — nothing crashes without a key.

Gemini (free tier): `FIF_MODEL` or GEMINI_MODELS[0]; on quota errors (429 / RESOURCE_EXHAUSTED) the
next model in GEMINI_MODELS is tried, then `LLMUnavailable`. Free-tier prompts may be used by Google to
improve its products — the app and the AI disclosure must say that user questions are sent to Google.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from .config import MODEL, has_api_key, load_env

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]   # default + fallback on quota / 503
GEMINI_TIMEOUT_MS = 25_000
ANTHROPIC_MODEL = "claude-opus-5-5"
PROVIDER_LABEL = {"gemini": "Gemini (Google)", "anthropic": "Claude (Anthropic)", "none": "no LLM"}


class LLMUnavailable(RuntimeError):
    """No key / SDK, quota exhausted, network error or the model declined. Callers fall back."""


def provider() -> str:
    """'gemini' | 'anthropic' | 'none' — from FIF_LLM, else from which key exists."""
    load_env()
    forced = (os.environ.get("FIF_LLM") or "").strip().lower()
    if forced in ("gemini", "anthropic", "none"):
        if forced != "none" and not has_api_key(forced):
            return "none"
        return forced
    if has_api_key("gemini"):
        return "gemini"
    if has_api_key("anthropic"):
        return "anthropic"
    return "none"


def model_name() -> str:
    p = provider()
    if p == "gemini":
        return os.environ.get("FIF_MODEL") or GEMINI_MODELS[0]
    if p == "anthropic":
        return os.environ.get("FIF_MODEL") or MODEL or ANTHROPIC_MODEL
    return "none"


def json_call(system: str, user: str, schema: dict, *, effort: str = "medium", max_tokens: int = 16000,
              budget_s: float | None = None) -> dict:
    """One generation constrained to `schema`; returns the parsed JSON object or raises LLMUnavailable.
    `budget_s`: hard wall-clock budget for all attempts (live answers use 19 s; batch jobs leave it None)."""
    p = provider()
    if p == "gemini":
        return _gemini_json_call(system, user, schema, max_tokens=max_tokens, budget_s=budget_s)
    if p == "anthropic":
        return _anthropic_json_call(system, user, schema, effort=effort, max_tokens=max_tokens)
    raise LLMUnavailable("no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env)")


# ----------------------------------------------------------------------------- Gemini
@lru_cache(maxsize=1)
def _gemini_client():
    load_env()
    if not has_api_key("gemini"):
        raise LLMUnavailable("no GEMINI_API_KEY in .env")
    try:
        from google import genai
    except ImportError as e:
        raise LLMUnavailable("google-genai package not installed") from e
    from google.genai import types
    try:
        # Short timeout + one retry: on a 503 "high demand" we'd rather switch model (below) than
        # let the SDK's default backoff hold the UI for ~30 s.
        return genai.Client(http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=2.0)))
    except Exception as e:
        raise LLMUnavailable(f"cannot create Gemini client: {e}") from e


def _gemini_models() -> list[str]:
    first = os.environ.get("FIF_MODEL") or GEMINI_MODELS[0]
    return [first] + [m for m in GEMINI_MODELS if m != first]


def _is_quota_error(e) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e).upper()
    return code == 429 or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg


def _attempt_timeouts(budget_s: float | None, n_models: int) -> list[float]:
    """Seconds allowed per model attempt: an even split of the budget (min 3 s), or the default per model."""
    if budget_s is None:
        return [GEMINI_TIMEOUT_MS / 1000] * n_models
    return [max(3.0, budget_s / n_models)] * n_models


def _gemini_json_call(system: str, user: str, schema: dict, *, max_tokens: int, budget_s: float | None = None) -> dict:
    client = _gemini_client()  # raises LLMUnavailable first, before any SDK import below
    from google.genai import errors, types

    import time as _time

    models = _gemini_models()
    timeouts = _attempt_timeouts(budget_s, len(models))
    deadline = None if budget_s is None else _time.monotonic() + budget_s
    reasons = []  # why each model was skipped; transient problems move on to the next model
    for model, t_s in zip(models, timeouts):
        if deadline is not None:
            left = deadline - _time.monotonic()
            if left < 2.0:
                reasons.append(f"{model}: skipped, time budget used up")
                break
            t_s = min(t_s, left)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=max_tokens,
            temperature=0.2,
            # per-attempt timeout; with a budget there are no SDK retries inside it (we switch model instead)
            http_options=types.HttpOptions(timeout=int(t_s * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1) if deadline else None),
        )
        try:
            resp = client.models.generate_content(model=model, contents=user, config=config)
        except errors.ClientError as e:
            if _is_quota_error(e):
                reasons.append(f"{model}: quota exhausted (429)")
                continue  # free-tier quota for this model used up -> try the lighter model
            raise LLMUnavailable(f"Gemini rejected the request ({getattr(e, 'code', '?')}): {str(e)[:120]}") from e
        except errors.ServerError as e:  # 503 "high demand", 500 — try the other model
            reasons.append(f"{model}: server error {getattr(e, 'code', '')} "
                           + ("high demand" if "demand" in str(e).lower() else str(e)[:60]))
            continue
        except errors.APIError as e:
            raise LLMUnavailable(f"Gemini API error: {str(e)[:120]}") from e
        except Exception as e:  # network / timeout
            reasons.append(f"{model}: cannot reach Gemini ({e.__class__.__name__})")
            continue
        text = getattr(resp, "text", None)
        if not text:
            fb = getattr(resp, "prompt_feedback", None)
            raise LLMUnavailable(f"Gemini returned no text ({getattr(fb, 'block_reason', None) or 'empty'})")
        try:
            out = json.loads(text)
        except ValueError as e:
            raise LLMUnavailable("Gemini returned invalid JSON") from e
        out["_model"] = model
        return out
    if all("quota" in r for r in reasons):
        raise LLMUnavailable("Gemini free-tier quota exhausted for " + ", ".join(_gemini_models()))
    raise LLMUnavailable("Gemini unavailable — " + "; ".join(reasons))


# ----------------------------------------------------------------------------- Anthropic
@lru_cache(maxsize=1)
def _client():
    load_env()
    if not has_api_key("anthropic"):
        raise LLMUnavailable("no ANTHROPIC_API_KEY in .env")
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("anthropic package not installed") from e
    try:
        return anthropic.Anthropic()
    except Exception as e:  # missing credentials
        raise LLMUnavailable(str(e)) from e


def _anthropic_json_call(system: str, user: str, schema: dict, *, effort: str, max_tokens: int) -> dict:
    client = _client()  # raises LLMUnavailable if the key or the SDK is missing — must run first
    import anthropic
    try:
        resp = client.beta.messages.create(
            model=os.environ.get("FIF_MODEL") or MODEL or ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("invalid ANTHROPIC_API_KEY") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("Anthropic rate limit reached") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("cannot reach the Anthropic API") from e

    if resp.stop_reason == "refusal":
        raise LLMUnavailable("the model declined this request")
    if resp.stop_reason == "max_tokens":
        raise LLMUnavailable("response hit max_tokens before finishing")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    out = json.loads(text)
    out["_model"] = resp.model
    return out
