"""Cached loaders for everything the dashboard reads. Pages import from here only.

Files come from main's ETL (`processed/*.json|csv`) and pc2's AI layer (`processed/ai/`).
Missing optional files return None / empty so pages can show "not available yet".
"""
from __future__ import annotations

import json
import re
import sys
import threading
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


O2_ROLE_LABELS = {"ambient": "atmosphere around the flame",
                  "burner": "oxidizer flowing out of the burner (inverse flame)"}


def o2_role(t: pd.DataFrame) -> pd.Series:
    """'ambient' | 'burner' per tests.csv row (None where O2 is empty).
    In inverse flames (ACME Flame Design PSI-10, CFI-G PSI-159) the burner blows O2 into a fuel
    atmosphere, so XO2 (up to 85 %) is the burner stream, not the cabin air. Uses main's `o2_role`
    column once the ETL writes it; until then the PSI table name says which flames are inverse."""
    if "o2_role" in t.columns:
        role = t["o2_role"].where(t["o2_role"].isin(list(O2_ROLE_LABELS)))
    else:
        inverse = t["source_file"].fillna("").str.contains("inverse", case=False)
        role = pd.Series("ambient", index=t.index).mask(inverse, "burner")
    return role.where(t["o2_frac"].notna())


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


def habitat_coverage(name: str) -> dict:
    """coverage.json entry for one habitat ({} if there is none)."""
    return next((h for h in (coverage() or {}).get("habitats", []) if h["name"] == name), {})


def no_solid_evidence(h: dict) -> bool:
    """True when no solid-material test is near this atmosphere at all — not even one at a matching O2
    whose table lacks pressure (ISS: BASS-II / Saffire ran at ~21 % O2, pressure unrecorded)."""
    near = (h.get("by_fuel_phase") or {}).get("solid", 0)
    o2_only = ((h.get("o2_match_pressure_unknown") or {}).get("by_fuel_phase") or {}).get("solid", 0)
    return near == 0 and o2_only == 0


def gaps_for(name: str) -> list[dict]:
    """The coverage.json gaps that name this habitat (main lists them in gap["habitats"])."""
    return [g for g in (coverage() or {}).get("gaps", [])
            if name in (g.get("habitats") or []) or (not g.get("habitats") and name.lower() in g["text"].lower())]


def key_number(name: str, fallback=None):
    """A headline number from processed/key_numbers.json (main writes it, so Home, About, slides and the
    video all quote the same figure). `fallback` (computed here) is used until the file has the key."""
    p = PROCESSED / "key_numbers.json"
    entry = (_json_or_none(str(p), _mtime(p)) or {}).get(name)
    return entry.get("value") if isinstance(entry, dict) and entry.get("value") is not None else fallback


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
    """Profile (contract 3): the LLM one if generated, else the rule-based one, else None.

    Same order as ai.profiles.load_profile(), but mtime-cached. The folders are read from the module at
    call time so tests can point them elsewhere. Provenance is in prof["generated"]["by"]."""
    from ai import profiles

    for d in (profiles.PROFILES_DIR, getattr(profiles, "PROFILES_RULES_DIR", None)):
        p = d / f"{pid}.json" if d else None
        if p and p.exists():
            return _cached_profile(str(p), _mtime(p))
    return None


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


# ---------------------------------------------------------------- raw PSI files (#34)
DATA_DIR = ROOT / "project-data-1-pc1" / "data"
DOC_TYPES = {"Reports": "Report", "Science Documents": "Science document", "Presentations": "Presentation"}


@st.cache_data(show_spinner=False)
def raw_table(name: str) -> pd.DataFrame:
    """An experimental table exactly as PSI ships it (all cells as text, nothing parsed)."""
    p = DATA_DIR / "experimental_tables" / name
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(p, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="latin-1")


@st.cache_data(show_spinner=False)
def file_index(pid: str) -> pd.DataFrame:
    """Every file PSI lists for one investigation (name, category, size) — for links, never downloads."""
    p = DATA_DIR / "file_index" / f"{pid}_files.csv"
    if not p.exists():
        return pd.DataFrame(columns=["category", "subcategory", "file_name", "size", "bytes"])
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")
    return df


def documents(pid: str) -> list[dict]:
    """Science documents / reports in the repo for one investigation (downloadable; ~275 MB in total)."""
    folder = DATA_DIR / "documents" / pid
    pages = indexed_pages()
    out = []
    for f in sorted(folder.glob("*")) if folder.exists() else []:
        if not f.is_file():
            continue
        parts = f.stem.split("_", 2)          # PSI-26 _ Science Documents _ SRD_BASS
        kind = parts[1] if len(parts) > 2 else ""
        title = (parts[2] if len(parts) > 2 else f.stem).replace("_", " ").strip()
        out.append({"title": title, "type": DOC_TYPES.get(kind, kind or "Document"), "file": f.name,
                    "format": f.suffix.lstrip(".").upper(), "bytes": f.stat().st_size, "path": str(f),
                    "pages_indexed": pages.get(f"documents/{pid}/{f.name}", 0)})
    return out


@st.cache_data(show_spinner=False)
def _indexed_pages(mtime: float) -> dict:
    seen: dict[str, set] = {}
    if CHUNKS.exists():
        with CHUNKS.open(encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                if c["file"].startswith("documents/"):
                    seen.setdefault(c["file"], set()).add(c.get("page"))
    return {k: len(v) for k, v in seen.items()}


def indexed_pages() -> dict:
    """documents/PSI-XX/<file> -> number of pages the AI search can cite from it."""
    return _indexed_pages(_mtime(CHUNKS))


@st.cache_data(show_spinner=False, max_entries=4)
def file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def human_size(n) -> str:
    if n is None or pd.isna(n):
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ---------------------------------------------------------------- analysis (main, #23 / #24)
ANALYSIS = PROCESSED / "analysis"


def similarity_method(sim: dict | None) -> str:
    """Short human description of how similarity.json was computed (its 'method' is a dict)."""
    m = (sim or {}).get("method")
    if isinstance(m, dict):
        return "60 % document text (MiniLM embeddings) + 40 % test conditions" if m.get("combined") else             "; ".join(str(v) for v in m.values())[:160]
    return str(m or "method not stated")


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

# Generative providers: (display name, who receives the question, extra privacy note).
PROVIDERS = {
    "gemini": ("Gemini", "Google's free API",
               " Free-tier content may be used by Google to improve its products."),
    "anthropic": ("Claude", "Anthropic's API", ""),
    "llm": ("Claude", "Anthropic's API", ""),          # older ai/ versions report "llm" for Claude
}
DEFAULT_MODEL = {"gemini": "Gemini", "anthropic": "Claude", "llm": "Claude"}

ANSWER_LABELS = {
    "extractive": ("🔎", "Answer quoted from a PSI document by a local QA model (no text generated)."),
    "no-llm": ("🔎", "Most relevant PSI passages (local AI search): found by a local embedding model + keyword "
                     "search in PSI documents, shown unedited with file and page."),
}


def answer_label(res: dict | str | None) -> tuple[str, str] | None:
    """(icon, text) describing how an ask() result was produced, or None for an unknown mode.
    Generated answers name the model and say where the question went; cached ones say no new request was made."""
    if not isinstance(res, dict):
        res = {"mode": res}
    mode = res.get("mode") or ""
    if mode in PROVIDERS:
        name, dest, privacy = PROVIDERS[mode]
        model = res.get("model") or DEFAULT_MODEL[mode]
        text = f"Generated by {name} (`{model}`) from the PSI passages below; uncited claims removed."
        if res.get("cached"):
            text += f" Saved answer — this question was sent to {dest} earlier, not now."
        else:
            text += f" Your question is sent to {dest}.{privacy}"
        return "✨", text
    return ANSWER_LABELS.get(mode)


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


def _load_index(state: dict) -> None:
    try:
        from ai import retrieval

        retrieval.warmup()
        state["ok"] = True
    except Exception:  # the app must still work without the AI layer
        state["ok"] = False


@st.cache_resource(show_spinner=False)
def _warm() -> tuple[threading.Thread, dict]:
    """One background thread per server process loads the retrieval index (~7 s: chunks, BM25, vectors,
    ONNX encoder), so the first page renders at once instead of waiting for it (#42)."""
    state: dict = {"ok": None}
    th = threading.Thread(target=_load_index, args=(state,), name="ignis-warmup", daemon=True)
    th.start()
    return th, state


def warmup() -> None:
    """Start loading the document index in the background (returns immediately)."""
    _warm()


def wait_for_index(timeout: float | None = None) -> bool | None:
    """Block until the index is loaded (Ask calls this before the first search). True/False = loaded/failed."""
    th, state = _warm()
    if th.is_alive():
        with st.spinner("Loading the document index (first question only, a few seconds)…"):
            th.join(timeout)
    return state["ok"]


# ---------------------------------------------------------------- partial gravity
# The challenge text says 0 g and 1 g results "could bound" intermediate gravity. NASA's newer work
# says the opposite (lunar gravity may be the MOST flammable case), so the app never repeats the
# bounding claim. Older processed files from main/pc2 still contain it; partial_g_fix() rewrites
# that one sentence at display time until they are regenerated.
FERKUL_2026 = "https://ntrs.nasa.gov/citations/20260001966"
RESEARCHERS_GUIDE = "https://www.nasa.gov/missions/station/a-researchers-guide-to-combustion-science/"
GRAVITY_OPTIONS = {"micro": "µg (orbit)", "lunar": "Moon (1/6 g)", "mars": "Mars (3/8 g)", "earth": "1 g (Earth)"}

PARTIAL_G_SHORT = ("0 g and 1 g results do **not** safely bound the Moon or Mars: NASA's newer analysis finds "
                   "lunar gravity may be close to the *most* flammable condition.")
PARTIAL_G_NOTE = (
    "**Partial gravity may be worse than both 0 g and 1 g.** PSI has no combustion tests at lunar (1/6 g) "
    "or Martian (3/8 g) gravity, and the common assumption that microgravity and 1 g results *bound* those "
    "cases is not safe:\n"
    "- “Numerical and experimental evidence suggests that Lunar gravity is nearly the most flammable "
    "condition.” The same slide: the oxygen level needed for flammability reaches a minimum near lunar g, and "
    "downward flame spread rate peaks in partial g. — Ferkul, Kralik, Liao & Johnston, *Material Flammability "
    f"at Lunar and Martian Gravity*, 57th LPSC 2026, NASA NTRS [20260001966]({FERKUL_2026}), slide 17 "
    "(outside PSI).\n"
    "- “In the low-flow conditions of microgravity or partial gravity, some materials are flammable at lower "
    "oxygen concentrations than in normal gravity.” — NASA, *A Researcher's Guide to ISS Combustion Science* "
    f"(2015), p. 18 ([nasa.gov]({RESEARCHERS_GUIDE}); also in the PSI-20 documents).\n\n"
    "NASA plans to measure this directly with the Flammability of Materials on the Moon (FM2) experiment."
)
_BOUND_RE = re.compile(r"(?:Microgravity|0 g|µg)[^.]*?(?:normal-gravity|normal gravity|1 g|1-g)[^.]*?"
                       r"\bbound[^.]*\.", re.I)


def partial_g_fix(text: str | None) -> str:
    """Replace the outdated 'microgravity and 1 g results bound partial-g' sentence with the correction."""
    return _BOUND_RE.sub(PARTIAL_G_SHORT, text) if text else ""


VAR_LABELS = {"o2": "O₂", "pressure": "pressure", "flow": "airflow", "fuel": "fuel", "diluent": "diluent",
              "geometry": "sample geometry", "sample_thickness": "sample thickness",
              "flow_direction": "flow direction", "droplet_diameter": "droplet diameter"}


def pretty_vars(keys) -> str:
    """['o2', 'sample_thickness'] -> 'O₂, sample thickness'; empty -> '—'."""
    return ", ".join(VAR_LABELS.get(k, str(k).replace("_", " ")) for k in keys or []) or "—"


def doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def fmt_range(rng, unit: str = "", scale: float = 1.0, digits: int = 2) -> str:
    """[0.12, 0.34] -> '0.12–0.34 unit'; None -> 'no data'."""
    if not rng or rng[0] is None:
        return "no data"
    lo, hi = (round(v * scale, digits) for v in rng)
    txt = f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"
    return f"{txt} {unit}".strip()
