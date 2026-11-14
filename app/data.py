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
    """Cache key part so a re-run of the ETL shows up without restarting Streamlit.
    The cached functions take it as `mtime`, not `_mtime`: st.cache_data leaves parameters that
    start with an underscore out of the cache key, which made every loader ignore file changes."""
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _investigations(mtime: float) -> list[dict]:
    invs = json.loads(INVESTIGATIONS.read_text(encoding="utf-8"))
    return sorted(invs, key=lambda i: int(i["id"].split("-")[1]))


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
def _tests(mtime: float) -> pd.DataFrame:
    return pd.read_csv(TESTS, dtype={"test_id": str}, keep_default_na=True, low_memory=False)


def tests() -> pd.DataFrame:
    """processed/tests.csv (contract 2) — one row per test record."""
    return _tests(_mtime(TESTS))


def table_rows(pid: str) -> tuple[int, str]:
    """(number of tests.csv rows, what a row is). investigations.json `n_tests` counts only
    one-row-per-test tables, so day-log tables (ACME, SLICE) would show 0 there."""
    t = tests()
    rows = t[t["investigation_id"] == pid]
    kinds = rows["row_kind"].dropna().unique().tolist() if "row_kind" in rows else []
    kind = {"test": "tests", "test_day": "test days", "session": "sessions",
            "test_matrix": "test-matrix rows"}.get(kinds[0], "rows") if len(kinds) == 1 else "rows"
    return len(rows), kind


@st.cache_data(show_spinner=False)
def _json_or_none(path: str, mtime: float):
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
def _etl_log(mtime: float) -> pd.DataFrame:
    return pd.read_csv(ETL_LOG, dtype=str, keep_default_na=False) if ETL_LOG.exists() else pd.DataFrame()


def etl_log() -> pd.DataFrame:
    return _etl_log(_mtime(ETL_LOG))


def profile(pid: str) -> dict | None:
    """AI profile (contract 3) or None if pc2 has not generated it yet."""
    from ai.profiles import PROFILES_DIR

    p = PROFILES_DIR / f"{pid}.json"
    return _cached_profile(str(p), _mtime(p)) if p.exists() else None


@st.cache_data(show_spinner=False)
def _cached_profile(path: str, mtime: float):
    """Keyed on the file path + mtime, so a regenerated (or different) profile file is re-read."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _doc_stats(mtime: float) -> dict:
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


# ---------------------------------------------------------------- analysis (main, #23 / #24)
ANALYSIS = PROCESSED / "analysis"


def flex_boundary() -> dict | None:
    """processed/analysis/flex_boundary.json: model card + P(extinction) grids, or None."""
    p = ANALYSIS / "flex_boundary.json"
    return _json_or_none(str(p), _mtime(p))


def similarity() -> dict | None:
    """processed/analysis/similarity.json (#24): 2-D map, clusters, neighbours, or None."""
    p = ANALYSIS / "similarity.json"
    return _json_or_none(str(p), _mtime(p))


@st.cache_resource(show_spinner="Training the FLEX outcome model (once per server, ~15 s)…")
def flex_model_ready() -> str | None:
    """Fit the classifier once per server process. Returns None when ready, else why it can't run."""
    try:
        from analysis.flex_classifier import model_card
        model_card()
        return None
    except ImportError as e:
        return f"the analysis package is not installed ({e.name}); run pip install -r requirements.txt"
    except Exception as e:  # keep the page alive
        return f"the model could not be trained ({type(e).__name__}: {e})"


# ---------------------------------------------------------------- provenance labels (#16)
# Every text shown must say how it was produced. Never call something AI-written unless a
# language model actually wrote it in this run.

ANSWER_LABELS = {
    "gemini": ("✨", "Answer generated by Google Gemini (free tier) only from the PSI passages below; "
                     "uncited claims removed. Questions are sent to Google."),
    "llm": ("✨", "Answer generated by Anthropic Claude only from the PSI passages below; "
                  "uncited claims removed. Questions are sent to Anthropic."),
    "extractive": ("🔎", "Answer quoted from a PSI document by a local QA model (no text generated)."),
    "no-llm": ("🔎", "Most relevant PSI passages (local AI search): found by a local embedding model + keyword "
                     "search in PSI documents, shown unedited with file and page."),
}


def answer_label(mode: str | None) -> tuple[str, str] | None:
    """(icon, text) for an ask() result, or None for an unknown mode."""
    return ANSWER_LABELS.get(mode or "")


def written_by_rules(gen: dict | None) -> bool:
    """True when a profile / insights file was built without a language model."""
    by = ((gen or {}).get("by") or "").lower()
    return "rule" in by or "no llm" in by


def profile_label(prof: dict) -> str:
    gen = prof.get("generated") or {}
    if written_by_rules(gen):
        return ("Summary from PSI metadata (quoted). Findings are sentences quoted from PSI documents, "
                "picked by keyword rules — no AI-written text.")
    model = gen.get("model") or gen.get("by") or "a language model"
    return f"✨ Written by AI ({model}, {gen.get('date', '')}) from PSI documents — check the cited sources."


def insights_label(raw: dict | list | None) -> str:
    gen = raw.get("generated") if isinstance(raw, dict) else None
    if written_by_rules(gen):
        return ("Computed from the PSI tables and backed by sentences quoted from PSI documents "
                "(no AI-written text). Each insight lists its sources.")
    if gen and gen.get("how"):
        return f"✨ {gen['how']}"
    return "✨ Written with AI from PSI documents; check the cited sources."


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
