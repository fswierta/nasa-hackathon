"""Shared paths and settings for the AI layer. Run everything from the repo root."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "project-data-1-pc1" / "data"
PROCESSED_DIR = ROOT / "processed"
AI_OUT = PROCESSED_DIR / "ai"
INDEX_DIR = AI_OUT / "index"
PROFILES_DIR = AI_OUT / "profiles"
CHUNKS_FILE = AI_OUT / "chunks.jsonl"

MODEL = os.environ.get("FIF_MODEL", "claude-opus-5-5")  # Anthropic default; Gemini default lives in ai.llm
COLLECTION = "psi_docs"

KEY_VARS = {"gemini": ("GEMINI_API_KEY",), "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}


def load_env() -> None:
    """Load ANTHROPIC_API_KEY etc. from the repo-root .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def has_api_key(provider: str | None = None) -> bool:
    """True if a non-empty key exists for `provider` ('gemini' | 'anthropic'), or for any provider."""
    load_env()
    providers = [provider] if provider else list(KEY_VARS)
    return any((os.environ.get(v) or "").strip() for p in providers for v in KEY_VARS.get(p, ()))
