"""Thin wrapper around the Anthropic SDK: one place for model, fallbacks, JSON output and errors."""
from __future__ import annotations

import json
from functools import lru_cache

from .config import MODEL, load_env


class LLMUnavailable(RuntimeError):
    """No API key / SDK, or the model refused. Callers fall back to non-LLM behaviour."""


@lru_cache(maxsize=1)
def _client():
    load_env()
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("anthropic package not installed") from e
    try:
        return anthropic.Anthropic()
    except Exception as e:  # missing credentials
        raise LLMUnavailable(str(e)) from e


def json_call(system: str, user: str, schema: dict, *, effort: str = "medium", max_tokens: int = 16000) -> dict:
    """One Messages call constrained to `schema`; returns the parsed JSON object."""
    import anthropic

    client = _client()
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("invalid ANTHROPIC_API_KEY") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("cannot reach the Anthropic API") from e

    if resp.stop_reason == "refusal":
        raise LLMUnavailable("the model declined this request")
    if resp.stop_reason == "max_tokens":
        raise LLMUnavailable("response hit max_tokens before finishing")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)
