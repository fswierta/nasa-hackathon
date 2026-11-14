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

MODEL = os.environ.get("FIF_MODEL", "claude-opus-5-5")
COLLECTION = "psi_docs"


def load_env() -> None:
    """Load ANTHROPIC_API_KEY etc. from the repo-root .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def has_api_key() -> bool:
    load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
