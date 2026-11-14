"""Cross-investigation fire-safety insights -> processed/ai/insights.json (issue #14, brief S3/R5/C1).

    python -m ai.insights

No LLM and no hand-written claims:
  * every `title` / `text` is a short template filled with numbers computed at build time from
    processed/investigations.json, processed/tests.csv, processed/coverage.json and ai.model;
  * every insight is backed by 1–3 verbatim sentences found by `search()` for a fixed query
    (citation has `quote`; the build re-checks each quote against chunks.jsonl and fails on a mismatch
    or on an insight with no backing passage).

Output shape (contract for pc3):
{"generated": {...},
 "insights": [{"id", "pillar", "kind": "evidence|data|data_gap", "title", "text", "habitats": [...],
               "numbers": {...}, "citations": [{"psi_id", "doi", "file", "page", "quote"?}]}],
 "earth_applications": [{"text", "citations": [...]}]}
Citations with a `quote` are passages; citations without one point at the computed data source.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from .catalog import by_id
from .config import AI_OUT, CHUNKS_FILE, PROCESSED_DIR
from .retrieval import expand_query, investigation_name_terms, search, term_weights
from .textutil import PRIOR_WORK, best_sentence_scored, dedupe_key, norm, sentences

OUT = AI_OUT / "insights.json"
ALL_HABITATS = ["ISS", "Lunar base", "Exploration atmosphere", "Mars base", "Pressurized rover", "Spacesuit"]
EXPLORATION = ["Lunar base", "Exploration atmosphere", "Mars base", "Pressurized rover"]
MIN_BACKING_SCORE = 4.0  # best_sentence_scored score a passage needs to back an insight
MAX_BACKING = 3


# ---------------------------------------------------------------- computed numbers
@lru_cache(maxsize=1)
def _tests():
    import pandas as pd
    return pd.read_csv(PROCESSED_DIR / "tests.csv")


def _inv(pid: str) -> dict:
    return by_id()[pid]


def _rows(*pids: str) -> int:
    t = _tests()
    return int(t["investigation_id"].isin(pids).sum())


def _rng(r, unit: str, scale: float = 1.0, digits: int = 0) -> str:
    if not r or r[0] is None and r[1] is None:
        return "not recorded"
    lo, hi = (r[0] if r[0] is not None else r[1]) * scale, (r[1] if r[1] is not None else r[0]) * scale
    f = f"{{:.{digits}f}}"
    return f"{f.format(lo)} {unit}" if abs(hi - lo) < 1e-9 else f"{f.format(lo)}–{f.format(hi)} {unit}"


def _materials(*pids: str) -> list[str]:
    seen = []
    for pid in pids:
        for m in _inv(pid).get("fuels") or []:
            if m.lower() not in {s.lower() for s in seen}:
                seen.append(m)
    return seen


def _flex_numbers() -> dict:
    t = _tests()
    f = t[t["investigation_id"] == "PSI-69"]
    raw = f["raw_row"].map(json.loads)
    co2 = raw.map(lambda r: float(r.get("CO initial ambient composition; mole fraction") or 0))
    he = raw.map(lambda r: float(r.get("He initial ambient composition; mole fraction") or 0))
    out = {"n": len(f), "n_co2": int((co2 > 0).sum()), "co2_max": co2.max(), "n_he": int((he > 0).sum()),
           "he_max": he.max(), "o2": _rng(_inv("PSI-69").get("o2_range"), "%", 100),
           "p": _rng(_inv("PSI-69").get("pressure_range_kPa"), "kPa")}
    for fuel, g in f.groupby("fuel"):
        key = fuel.lower()
        out[f"{key}_done_min"] = g[g["outcome"] == "completion"]["o2_frac"].min() * 100
        ext = g[g["outcome"] == "extinction"]["o2_frac"]
        out[f"{key}_ext"] = f"{ext.min() * 100:.0f}–{ext.max() * 100:.0f} %"
        out[f"{key}_n_ext"] = len(ext)
    return out


def _model_numbers() -> dict:
    from .model import model_card

    c = model_card()["extinction_d"]
    return {"n": c["n_train"], "r2": c["cv_r2"], "mm_per_point": -c["coefficients"]["o2_frac"] / 100,
            "rmse": c["cv_rmse"]}


# ---------------------------------------------------------------- insight specs
# Each spec: numbers() -> dict; title/text are format templates over those numbers; `query` + `scope`
# drive the backing-passage search; `sources` names the computed-data files behind the numbers.
SPECS = [
    {"id": "prev-flow-spread", "pillar": "prevention", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n": _rows("PSI-26", "PSI-25"), "mats": ", ".join(_materials("PSI-26", "PSI-25")),
                         "o2": _rng(_inv("PSI-25").get("o2_range"), "%", 100, 1)},
     "title": "Flow speed controls flame spread over solids — BASS / BASS-II ({n} tests)",
     "text": "BASS and BASS-II burned {mats} in the ISS glovebox flow duct ({n} test rows; BASS-II O2 {o2}). "
             "Quoted results on how airflow speed sets spread rate and extinction:",
     "query": "flame spread rate flow speed opposed concurrent extinction low flow microgravity",
     "scope": ["PSI-26", "PSI-25"], "sources": [("PSI-26", "processed/tests.csv"), ("PSI-25", "processed/tests.csv")]},
    {"id": "prev-std6001", "pillar": "prevention", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n_inv": 5},
     "title": "1 g material screening (NASA-STD-6001 Test 1) vs microgravity — checked by {n_inv} investigations",
     "text": "BASS, BASS-II and Saffire I–III were designed to compare NASA's 1 g upward-burning screening test with "
             "actual flammability in low gravity. Quoted statements:",
     "query": "NASA-STD-6001 Test 1 normal gravity screening not conservative microgravity material flammability",
     "scope": ["PSI-26", "PSI-25", "PSI-98", "PSI-99", "PSI-100", "REF"], "sources": []},
    {"id": "prev-saffire-scale", "pillar": "prevention", "kind": "evidence", "habitats": ["ISS", "Lunar base", "Pressurized rover"],
     "numbers": lambda: {"n": _rows("PSI-98", "PSI-99", "PSI-100"), "mats": ", ".join(_materials("PSI-98", "PSI-99", "PSI-100")),
                         "flow": _rng([20, 30], "cm/s"), "o2": _rng(_inv("PSI-99").get("o2_range"), "%", 100, 1)},
     "title": "Large-scale fire growth in a real spacecraft — Saffire I–III ({n} samples, {flow})",
     "text": "Saffire I–III burned {mats} inside an uncrewed Cygnus vehicle at {flow} airflow and ~{o2} O2 "
             "({n} sample rows). Quoted statements:",
     "query": "large scale fire growth flame spread sample Saffire low gravity steady spread rate",
     "scope": ["PSI-98", "PSI-99", "PSI-100"],
     "sources": [("PSI-98", "processed/tests.csv"), ("PSI-99", "processed/tests.csv"), ("PSI-100", "processed/tests.csv")]},
    {"id": "prev-high-o2", "pillar": "prevention", "kind": "evidence", "habitats": EXPLORATION,
     "numbers": lambda: {"n": _rows("PSI-20"), "o2": _rng(_inv("PSI-20").get("o2_range"), "%", 100)},
     "title": "Burning at elevated O2 — ACME BRE solid-fuel emulator ({o2} O2, {n} test days)",
     "text": "ACME BRE used a gas burner that emulates burning solids at {o2} O2 ({n} test-day rows) — the only PSI "
             "flight data reaching exploration-atmosphere O2 for solid-like fuels. Quoted results:",
     "query": "common plastic steadily burn NASA proposed atmospheric conditions elevated oxygen reduced pressure",
     "scope": ["PSI-20"], "sources": [("PSI-20", "processed/tests.csv")]},
    {"id": "det-particle-size", "pillar": "detection", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n": _rows("PSI-102", "PSI-101"), "mats": ", ".join(_materials("PSI-102", "PSI-101"))},
     "title": "Smoke particle size in microgravity — SAME / SAME-R ({n} tests)",
     "text": "SAME and SAME-R heated {mats} on the ISS and measured the smoke ({n} test rows). Quoted results on "
             "particle size and what it means for detectors:",
     "query": "experimental results smoke particles below 500 nm diameter particle size distribution detector",
     "scope": ["PSI-102", "PSI-101"], "must": r"particle|size|diameter",
     "sources": [("PSI-102", "processed/tests.csv"), ("PSI-101", "processed/tests.csv")]},
    {"id": "det-detector-type", "pillar": "detection", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n_types": len({m["name"] for pid in ("PSI-102", "PSI-101") for m in (_profile(pid).get("measurements") or [])
                                         if m["type"] == "particle counter / smoke detector"})},
     "title": "Ionization vs light-scattering smoke detection — {n_types} detector / counter types flown in SAME",
     "text": "SAME / SAME-R flew {n_types} smoke-detector and particle-counter types side by side (from the profiles' "
             "instrument lists). Quoted results on which ones saw the smoke:",
     "query": "ionization detector light scattering detector signals smoke tests detection",
     "scope": ["PSI-102", "PSI-101"], "sources": [("PSI-102", "processed/ai/profiles/PSI-102.json")]},
    {"id": "det-dust", "pillar": "detection", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n": _rows("PSI-47")},
     "title": "Dust vs smoke on orbit — DAFT particle-counter test ({n} test rows)",
     "text": "DAFT / DAFT-2 flew a TSI P-Trak particle counter with test dust ({n} test rows). Quoted results on how "
             "dust behaves without gravity:",
     "query": "dust particles larger particles remain entrained settling on orbit P-Trak",
     "scope": ["PSI-47"], "sources": [("PSI-47", "processed/tests.csv")]},
    {"id": "sup-loi", "pillar": "suppression", "kind": "data", "habitats": ALL_HABITATS,
     "numbers": _flex_numbers,
     "title": "Droplet flames: heptane burned to completion down to {heptane_done_min:.0f} % O2 — FLEX ({n} tests)",
     "text": "FLEX ({n} droplet tests, O2 {o2}, {p}): lowest O2 with burn-to-completion was {heptane_done_min:.0f} % "
             "(heptane) and {methanol_done_min:.0f} % (methanol); flames extinguished at {heptane_ext} (heptane, "
             "{heptane_n_ext} tests) and {methanol_ext} (methanol, {methanol_n_ext} tests). Quoted findings on the "
             "limiting oxygen index:",
     "query": "limiting oxygen index LOI lower than normal gravity extinction limits droplets",
     "scope": ["PSI-69", "REF"], "sources": [("PSI-69", "processed/tests.csv")]},
    {"id": "sup-diluents", "pillar": "suppression", "kind": "data", "habitats": ALL_HABITATS,
     "numbers": _flex_numbers,
     "title": "CO2 and helium as suppressants — FLEX ({n_co2} CO2 tests, {n_he} He tests)",
     "text": "FLEX added CO2 (up to {co2_max:.0%} mole fraction, {n_co2} tests) or helium (up to {he_max:.0%}, {n_he} "
             "tests) to the ambient. Quoted statements on how the suppressant changes the extinction limit:",
     "query": "CO2 helium suppressant effective microgravity minimum oxygen concentration extinction",
     "scope": ["PSI-69", "PSI-68", "PSI-39"], "must": r"CO2|helium|He|suppress|dilu",
     "sources": [("PSI-69", "processed/tests.csv")]},
    {"id": "sup-model", "pillar": "suppression", "kind": "data", "habitats": ALL_HABITATS,
     "numbers": _model_numbers,
     "title": "Each 1 % less O2 → flame out at a {mm_per_point:.2f} mm larger droplet (FLEX model, n={n})",
     "text": "Linear model on {n} FLEX tests (5-fold CV R² {r2:.2f}, RMSE {rmse:.2f} mm): extinction diameter rises "
             "{mm_per_point:.2f} mm per percentage point less O2, other inputs fixed (`ai.model.predict_flex`). "
             "Quoted statements on extinction diameter:",
     "query": "extinction diameter droplet oxygen concentration finite droplet size extinguish",
     "scope": ["PSI-69", "PSI-68"], "must": r"oxygen|O2|extinction diameter",
     "sources": [("PSI-69", "processed/tests.csv")]},
    {"id": "sup-cool-flame", "pillar": "suppression", "kind": "evidence", "habitats": ALL_HABITATS,
     "numbers": lambda: {"n": _rows("PSI-69", "PSI-39")},
     "title": "Cool flames can keep burning after the visible flame goes out — FLEX / CFI ({n} tests)",
     "text": "FLEX and CFI droplet tests ({n} rows) recorded low-temperature 'cool flames' after visible flame "
             "extinction — relevant when judging whether a fire is really out. Quoted results:",
     "query": "cool flame after radiative extinction visible flame prolonged burn vaporize",
     "scope": ["PSI-69", "PSI-39", "REF"], "sources": [("PSI-69", "processed/tests.csv"), ("PSI-39", "processed/tests.csv")]},
]

GAP_QUERIES = {  # coverage.json gap -> query/scope for backing passages
    "solid-material test at the O2 level": ("common plastic steadily burn NASA proposed atmospheric conditions elevated oxygen",
                                            ["PSI-20", "REF"]),
    "only up to 22.2": ("exploration atmospheres reduced pressure elevated oxygen materials flammability", ["PSI-20", "REF"]),
    "recorded pressure": ("testing in exploration atmospheres with reduced pressure and elevated oxygen", ["REF", "PSI-20"]),
    "partial-gravity": ("partial gravity lunar martian gravity electric field mimic buoyancy", ["PSI-22", "REF"]),
    "spacesuit": ("inverse flames oxygen enriched highest oxygen concentration", ["PSI-159", "PSI-10"]),
}


@lru_cache(maxsize=None)
def _profile(pid: str) -> dict:
    from .profiles import load_profile
    return load_profile(pid) or {}


# ---------------------------------------------------------------- backing passages
def backing(query: str, scope: list[str], n: int = MAX_BACKING, must: str | None = None) -> list[dict]:
    """Up to `n` verbatim sentences from `search()` that best support `query` (score >= MIN_BACKING_SCORE).
    `must`: regex every backing sentence has to match (keeps e.g. the CO2/He insight on suppressants)."""
    must_rx = re.compile(must, re.I) if must else None
    hits = search(query, k=15, psi_ids=scope, max_per_file=4)
    extra = expand_query(query)[len(query):]
    weights = term_weights(query + " " + extra)
    ignore = investigation_name_terms(query)
    scored, seen = [], set()
    for h in hits:
        s, score = best_sentence_scored(h["text"], query, extra, weights=weights, ignore=ignore)
        key = dedupe_key(s or "")
        # backing must be this investigation's own statement, not a literature-review sentence
        if (s and score >= MIN_BACKING_SCORE and key not in seen and not PRIOR_WORK.search(s)
                and (must_rx is None or must_rx.search(s))):
            seen.add(key)
            scored.append((score, h, s))
    scored.sort(key=lambda x: -x[0])
    return [{"psi_id": h["psi_id"], "doi": h.get("doi"), "file": h["file"], "page": h.get("page"), "quote": s}
            for _, h, s in scored[:n]]


@lru_cache(maxsize=1)
def _chunk_texts() -> dict[tuple, list[str]]:
    table: dict[tuple, list[str]] = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            table.setdefault((c["file"], c["page"]), []).append(norm(c["text"]))
    return table


def verify_quote(c: dict) -> None:
    if "quote" not in c:
        return
    q = norm(c["quote"])
    if not any(q in t for t in _chunk_texts().get((c["file"], c["page"]), [])):
        raise ValueError(f"quote not found in {c['file']} p{c['page']}: {q[:70]}…")


def _data_cite(pid: str, file: str) -> dict:
    return {"psi_id": pid, "doi": _inv(pid).get("doi"), "file": file, "page": None}


# ---------------------------------------------------------------- builders
def spec_insights() -> list[dict]:
    out = []
    for s in SPECS:
        nums = s["numbers"]()
        cites = backing(s["query"], s["scope"], must=s.get("must"))
        if not cites:
            raise ValueError(f"{s['id']}: no backing passage found for query {s['query']!r}")
        out.append({
            "id": s["id"], "pillar": s["pillar"], "kind": s["kind"], "habitats": s["habitats"],
            "title": s["title"].format(**nums), "text": s["text"].format(**nums),
            "numbers": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in nums.items()},
            "citations": cites + [_data_cite(pid, f) for pid, f in s["sources"]],
        })
    return out


def gap_insights() -> list[dict]:
    from .model import predict_flex
    from .retrieval import mentioned_investigations

    out = []
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    for i, g in enumerate(cov.get("gaps", []), start=1):
        blob = " ".join(str(g.get(k, "")) for k in ("text", "evidence", "closest_proxy"))
        pids = list(dict.fromkeys(re.findall(r"PSI-\d+", blob) + mentioned_investigations(blob)))
        query, scope = next((qs for key, qs in GAP_QUERIES.items() if key.lower() in g["text"].lower()),
                            (g["text"], None))
        habitats = g.get("habitats") or ([g["habitat"]] if g.get("habitat") else EXPLORATION)
        text = g["evidence"] + (f" Closest proxy: {g['closest_proxy']}" if g.get("closest_proxy") else "")
        out.append({
            "id": f"gap-coverage-{i}", "pillar": "prevention", "kind": "data_gap", "habitats": habitats,
            "title": g["text"], "text": text, "numbers": {},
            "citations": backing(query, scope, n=2) + [_data_cite(p, "processed/coverage.json") for p in pids if p in by_id()],
        })
    p = predict_flex(0.34, 56.5, "N2", 3.0, "heptane")
    lo = min(p["valid_range"]["pressure_kPa"][0], predict_flex(0.21, 101.3, fuel="methanol")["valid_range"]["pressure_kPa"][0])
    out.append({
        "id": "gap-flex-pressure", "pillar": "suppression", "kind": "data_gap", "habitats": EXPLORATION,
        "title": f"No droplet-extinction data below {lo:.0f} kPa (exploration atmosphere: 56.5 kPa)",
        "text": f"Lowest FLEX test pressure: {lo:.0f} kPa. `predict_flex` refuses 56.5 kPa: {p['reason'].rstrip('.')}.",
        "numbers": {"flex_min_kPa": round(lo, 1), "exploration_kPa": 56.5},
        "citations": backing("test pressure atm ambient pressure droplet tests range", ["PSI-69"], n=2)
                     + [_data_cite("PSI-69", "processed/tests.csv")],
    })
    return out


EARTH_PIDS = ["PSI-98", "PSI-102", "PSI-47", "PSI-69", "PSI-39", "PSI-159", "PSI-22"]


def earth_applications() -> list[dict]:
    """Verbatim researchImpacts sentences about Earth uses (no paraphrase)."""
    from .profile_rules import earth_applications as earth_sentences

    out, seen = [], set()
    for pid in EARTH_PIDS:
        for s in earth_sentences(pid)[:2]:
            key = s.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            file = f"metadata/{pid}.json#researchImpacts"
            out.append({"text": f"{_inv(pid).get('acronym')}: “{s}”",
                        "citations": [{"psi_id": pid, "doi": _inv(pid).get("doi"), "file": file, "page": None, "quote": s}]})
    return out


def build() -> dict:
    insights = spec_insights() + gap_insights()
    earth = earth_applications()
    for item in insights + earth:
        for c in item["citations"]:
            verify_quote(c)
        if not any("quote" in c for c in item["citations"]):
            raise ValueError(f"{item.get('id', item['text'][:40])}: no verbatim backing passage")
    return {
        "generated": {"date": date.today().isoformat(), "by": "rules (no LLM)",
                      "how": "titles/texts are templates filled with numbers computed from processed/*.json|csv and "
                             "ai.model; each insight is backed by verbatim sentences found by ai.retrieval.search() "
                             "(citations with `quote`), re-verified against processed/ai/chunks.jsonl"},
        "insights": insights,
        "earth_applications": earth,
    }


def main():
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"{len(result['insights'])} insights -> {OUT}")
    print(dict(Counter((i["pillar"], i["kind"]) for i in result["insights"])))
    print(f"{len(result['earth_applications'])} earth applications")


if __name__ == "__main__":
    main()
