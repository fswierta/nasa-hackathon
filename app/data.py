"""Cached loaders for everything the dashboard reads. Pages import from here only.

Files come from main's ETL (`processed/*.json|csv`) and pc2's AI layer (`processed/ai/`).
Missing optional files return None / empty so pages can show "not available yet".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "processed"
INVESTIGATIONS = PROCESSED / "investigations.json"
TESTS = PROCESSED / "tests.csv"
COVERAGE = PROCESSED / "coverage.json"
ETL_LOG = PROCESSED / "etl_log.csv"
CHUNKS = PROCESSED / "ai" / "chunks.jsonl"

PSI_REPO = "https://psi.nasa.gov/physci/repo/data/investigations/"


def _mtime(p: Path) -> float:
    """Cache key part so a re-run of the ETL shows up without restarting Streamlit."""
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _investigations(_m: float) -> list[dict]:
    return json.loads(INVESTIGATIONS.read_text(encoding="utf-8"))


def investigations() -> list[dict]:
    """processed/investigations.json (contract 1), sorted by PSI number."""
    return _investigations(_mtime(INVESTIGATIONS))


def investigation(pid: str) -> dict | None:
    return next((i for i in investigations() if i["id"] == pid), None)


def label(inv: dict) -> str:
    """'BASS (PSI-26)'; ground studies have no acronym -> short title."""
    name = inv.get("acronym") or (inv.get("title") or inv["id"])[:48]
    return f"{name} ({inv['id']})"


@st.cache_data(show_spinner=False)
def _tests(_m: float) -> pd.DataFrame:
    return pd.read_csv(TESTS, dtype={"test_id": str}, keep_default_na=True, low_memory=False)


def tests() -> pd.DataFrame:
    """processed/tests.csv (contract 2) — one row per test record."""
    return _tests(_mtime(TESTS))


@st.cache_data(show_spinner=False)
def _json_or_none(path: str, _m: float):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def coverage() -> dict | None:
    """processed/coverage.json (main, M3) or None."""
    return _json_or_none(str(COVERAGE), _mtime(COVERAGE))


def insights() -> dict | list | None:
    """processed/ai/insights.json (pc2, P2-4) or None until it exists."""
    p = PROCESSED / "ai" / "insights.json"
    return _json_or_none(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _etl_log(_m: float) -> pd.DataFrame:
    return pd.read_csv(ETL_LOG, dtype=str, keep_default_na=False) if ETL_LOG.exists() else pd.DataFrame()


def etl_log() -> pd.DataFrame:
    return _etl_log(_mtime(ETL_LOG))


def profile(pid: str) -> dict | None:
    """AI profile (contract 3) or None if pc2 has not generated it yet."""
    from ai.profiles import PROFILES_DIR, load_profile

    p = PROFILES_DIR / f"{pid}.json"
    return _cached_profile(pid, _mtime(p)) if p.exists() else None


@st.cache_data(show_spinner=False)
def _cached_profile(pid: str, _m: float):
    from ai.profiles import load_profile

    return load_profile(pid)


@st.cache_data(show_spinner=False)
def _doc_stats(_m: float) -> dict:
    files, per_inv = set(), {}
    if CHUNKS.exists():
        with CHUNKS.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if c["file"].startswith("metadata/"):
                    continue
                files.add(c["file"])
                per_inv.setdefault(c["psi_id"], set()).add(c["file"])
    return {"n_documents": len(files), "per_investigation": {k: len(v) for k, v in per_inv.items()}}


def doc_stats() -> dict:
    """Science documents / reports indexed for the AI (from processed/ai/chunks.jsonl)."""
    return _doc_stats(_mtime(CHUNKS))


@st.cache_resource(show_spinner="Loading the document index…")
def warmup() -> bool:
    """Load the retrieval index once per server process (see ai.retrieval.warmup)."""
    try:
        from ai import retrieval

        retrieval.warmup()
        return True
    except Exception:  # the app must still work without the AI layer
        return False


def doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def fmt_range(rng, unit: str = "", scale: float = 1.0, digits: int = 2) -> str:
    """[0.12, 0.34] -> '0.12–0.34 unit'; None -> 'no data'."""
    if not rng or rng[0] is None:
        return "no data"
    lo, hi = (round(v * scale, digits) for v in rng)
    txt = f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"
    return f"{txt} {unit}".strip()
