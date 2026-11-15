"""Explainable scenario ranking of investigations (AGENT_BRIEF §6.2, contract 4 in CLAUDE.md).

score = Σ w_k · s_k / Σ w_k, every component s_k in [0, 1].
Deterministic and instant (no LLM) so the "why" is always reproducible.
Missing data scores 0 for that component and is reported, never guessed.

Atmosphere (#64 item 1): ONE joint score from the test rows, not separate O2 / pressure range overlaps (those
gave 1.0 + 1.0 to investigations whose O2 and pressure were never tested together). It uses the definition of
"near" of processed/coverage.json (etl/build_coverage.py) and app/data.py::near_atmosphere: ambient rows only
(never inverse-flame burner O2), O2 within ±0.06 and pressure within ±25 %, a row with a range (test matrix,
day log) counts if its box overlaps. d_min = distance of the nearest test in units of that window (0 when the habitat lies inside a
test's box); score = 1 - d_min / 2 inside the window (0.5-1), 0.5 * exp(-(d_min - 1)) outside — so an
investigation with tests near scores >= 0.5 and one with none scores < 0.5 (never 1.0).
"""
from __future__ import annotations

import math
from functools import lru_cache

from .catalog import load_investigations

# Public weights (contract 4). "o2" + "pressure" TOGETHER weight the one joint "atmosphere" component (#64) - kept
# as two keys so existing weight sliders keep working; rank() also accepts weights={"atmosphere": x} directly.
DEFAULT_WEIGHTS = {
    "o2": 0.20,
    "pressure": 0.15,
    "flow": 0.10,
    "material": 0.20,
    "pillar": 0.10,
    "tier": 0.15,
    "richness": 0.10,
    "gravity": 0.10,
}

# Gravity (#37, #54): 1.0 only where the investigation has test data AT the scenario's gravity, else None
# ("no data") — never partial credit. Microgravity: every flight investigation (all PSI flight tests ran in
# microgravity). Earth (1 g): only the investigation(s) the scenario engine lists in
# analysis.scenario.gravity_evidence() (Saffire-II's 1 g ground comparison). Lunar / Mars: PSI has no
# partial-gravity data, so None for everyone and the engine's own statement goes into `gaps`.
PARTIAL_G_PROXY = {"PSI-22": "ACME E-FIELD used an electric field on the flame as a partial-gravity proxy "
                             "(no real partial-gravity test) — a note, not a score."}

PILLAR_FOCUS_WEIGHT = 0.45  # pillar weight when the scenario names a pillar

TIER_SCORE = {"A": 1.0, "B": 0.6, "C": 0.3}
SAFETY_PILLARS = {"prevention", "detection", "suppression"}

# Decay scale: how far outside the tested range a scenario can be before the component hits 0.
FLOW_SCALE = 15.0        # cm/s
NEAR_O2, NEAR_P = 0.06, 0.25  # = etl/build_coverage.py O2_TOL / P_TOL and app/data.py NEAR_O2 / NEAR_P

# Material aliases so "cotton/fiberglass fabric" matches the SIBAL samples, etc.
ALIASES = {
    "sibal": {"sibal", "cotton", "fiberglass", "fabric"},
    "pmma": {"pmma", "acrylic", "plexiglas"},
    "nomex": {"nomex"},
    "silicone": {"silicone"},
    "kapton": {"kapton"},
    "teflon": {"teflon", "ptfe"},
    "ultem": {"ultem"},
}


def _range_score(value, rng, scale, log=False):
    """1 inside [lo, hi], linear decay to 0 over `scale` outside. None if no data."""
    if value is None or not rng:
        return None
    lo, hi = rng[0], rng[1]
    if lo is None and hi is None:
        return None
    lo = hi if lo is None else lo
    hi = lo if hi is None else hi
    if log:
        if value <= 0 or lo <= 0:
            return None
        value, lo, hi = math.log(value), math.log(lo), math.log(hi)
    if lo <= value <= hi:
        return 1.0
    dist = lo - value if value < lo else value - hi
    return max(0.0, 1.0 - dist / scale)


@lru_cache(maxsize=1)
def _ambient_boxes() -> dict:
    """investigation id -> list of (o2_lo, o2_hi, p_lo, p_hi) for every combustion row with an ambient O2 AND a
    pressure (session rows and inverse-flame burner O2 excluded, as in coverage.json)."""
    import csv
    from .config import PROCESSED_DIR

    def f(v):
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None
    out = {}
    with (PROCESSED_DIR / "tests.csv").open(encoding="utf-8", newline="") as fh:
        for t in csv.DictReader(fh):
            if t.get("row_kind") == "session" or t.get("o2_role") == "burner":
                continue
            o2, p = f(t.get("o2_frac")), f(t.get("pressure_kPa"))
            if o2 is None or p is None:
                continue
            o2_hi, p_hi = f(t.get("o2_frac_max")), f(t.get("pressure_kPa_max"))
            out.setdefault(t["investigation_id"], []).append((o2, o2 if o2_hi is None else o2_hi, p, p if p_hi is None else p_hi))
    return out


@lru_cache(maxsize=256)
def atmosphere_near(o2: float, p: float) -> dict:
    """Per investigation with O2 + pressure rows: {"n": tests near, "d_min": distance of the nearest test in units
    of the near window, "nearest": "30 % O₂ / 70.4 kPa"}. Investigations without such rows are absent."""
    out = {}
    for pid, boxes in _ambient_boxes().items():
        n, best = 0, None
        for lo_o, hi_o, lo_p, hi_p in boxes:
            if lo_o <= o2 + NEAR_O2 and hi_o >= o2 - NEAR_O2 and lo_p <= (1 + NEAR_P) * p and hi_p >= (1 - NEAR_P) * p:
                n += 1
            d_o = max(lo_o - o2, o2 - hi_o, 0.0) / NEAR_O2
            d_p = max(lo_p - p, p - hi_p, 0.0) / (NEAR_P * p)
            d = math.hypot(d_o, d_p)
            if best is None or d < best[0]:
                best = (d, lo_o, hi_o, lo_p, hi_p)
        d, lo_o, hi_o, lo_p, hi_p = best

        def rng(lo, hi, scale, fmt):
            return f"{lo * scale:{fmt}}" if lo == hi else f"{lo * scale:{fmt}}–{hi * scale:{fmt}}"
        out[pid] = {"n": n, "d_min": round(d, 4),
                    "nearest": f"{rng(lo_o, hi_o, 100, '.3g')} % O₂ / {rng(lo_p, hi_p, 1, '.4g')} kPa"}
    return out


def _atmosphere_score(scenario: dict, inv: dict):
    o2, p = scenario.get("o2_frac"), scenario.get("pressure_kPa")
    if o2 is None or not p:
        return None, None
    info = atmosphere_near(round(float(o2), 4), round(float(p), 2)).get(inv["id"])
    if info is None:
        return None, None
    return atmosphere_score(info["d_min"]), info


def atmosphere_score(d_min: float) -> float:
    """>= 0.5 when the nearest test is inside the near window (d_min <= 1), < 0.5 outside, 1.0 only at d_min 0."""
    return 1.0 - d_min / 2 if d_min <= 1 else 0.5 * math.exp(-(d_min - 1))


@lru_cache(maxsize=None)
def _family(name: str) -> str | None:
    n = name.lower()
    for fam, words in ALIASES.items():
        if any(w in n for w in words):
            return fam
    return None


def _material_score(materials, inv):
    if not materials:
        return None
    fuels = inv.get("fuels") or []
    if not fuels:
        return 0.3 if inv.get("fuel_phase") == "solid" else None
    inv_fams = {_family(f) for f in fuels} - {None}
    want = {_family(m) for m in materials} - {None}
    if want and inv_fams:
        return len(want & inv_fams) / len(want) if want & inv_fams else (0.3 if inv.get("fuel_phase") == "solid" else 0.0)
    return 0.3 if inv.get("fuel_phase") == "solid" else 0.0


def _pillar_score(pillar, inv):
    pillars = set(inv.get("pillars") or [])
    if not pillars:
        return None
    if pillar:
        return 1.0 if pillar in pillars else 0.0
    return 1.0 if pillars & SAFETY_PILLARS else 0.5


def _richness(inv):
    n = inv.get("n_tests") or 0
    return min(1.0, math.log1p(n) / math.log1p(300))


def _fmt_range(rng, unit, digits=2):
    if not rng:
        return "unknown"
    lo, hi = rng
    f = lambda v: "?" if v is None else f"{v:.{digits}g}"
    return f"{f(lo)} {unit}" if lo == hi else f"{f(lo)}–{f(hi)} {unit}"


def _gaps(scenario, inv, breakdown, near=None):
    gaps = []
    if scenario.get("gravity") in ("lunar", "mars"):
        # #54: the scenario engine's statement (one wording for the whole app; #36/#37: no '0 g / 1 g bound it')
        gaps.append(_gravity_statement(scenario["gravity"]) or
                    "No partial-gravity combustion data exists in PSI; 0 g and 1 g results may not bound this case.")
        if inv["id"] in PARTIAL_G_PROXY:
            gaps.append(PARTIAL_G_PROXY[inv["id"]])
    if breakdown["atmosphere"] is None and scenario.get("o2_frac") is not None:
        gaps.append("No test with both an ambient O2 and a pressure in its PSI table, so it can't be matched to "
                    "this atmosphere.")
    elif near is not None and near["n"] == 0:
        gaps.append(f"No test within ±6 pts O2 and ±25 % pressure of this atmosphere; nearest test at {near['nearest']}.")
    if breakdown["flow"] is None:
        gaps.append("No airflow range on record for this investigation.")
    o2 = scenario.get("o2_frac")
    rng = inv.get("o2_range")
    if o2 is not None and rng and rng[1] is not None and o2 > rng[1]:
        gaps.append(f"Scenario O2 ({o2:.0%}) is above the highest tested ({rng[1]:.0%}).")
    return gaps


@lru_cache(maxsize=1)
def _earth_gravity_ids() -> frozenset:
    """PSI-IDs with 1 g test data, from the scenario engine (computed once: rank() stays ~1 ms)."""
    try:
        from analysis.scenario import gravity_evidence
        return frozenset(g["source"]["psi_id"] for g in gravity_evidence())
    except Exception:  # engine unavailable -> no 1 g evidence claimed
        return frozenset()


def _gravity_statement(gravity: str) -> str | None:
    try:
        from analysis.scenario import GRAVITY
        return GRAVITY.get(gravity, {}).get("statement")
    except Exception:
        return None


def _gravity_score(gravity, inv):
    if gravity == "micro":
        return 1.0 if inv.get("type") == "flight" else None
    if gravity == "earth":
        return 1.0 if inv["id"] in _earth_gravity_ids() else None
    return None  # lunar / mars / unspecified: no data


def _why(inv, breakdown, weights, near=None):
    contrib = {k: (v or 0.0) * weights.get(k, 0) for k, v in breakdown.items()}
    if near is not None and near["n"] == 0:
        contrib["atmosphere"] = 0.0  # never "ranks here for 0 tests near"; the gap says so instead
    top = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    phrases = {
        "atmosphere": (f"{near['n']} tests within ±6 pts O₂ and ±25 % pressure; nearest test at {near['nearest']}"
                       if near else "tested atmosphere"),
        "flow": f"tested airflow {_fmt_range(inv.get('flow_range_cm_s'), 'cm/s', 3)}",
        "material": "materials: " + ", ".join((inv.get("fuels") or [])[:4]) if inv.get("fuels") else "solid-fuel investigation",
        "pillar": "fire-safety focus: " + ", ".join(inv.get("pillars") or []),
        "tier": f"relevance tier {inv.get('relevance_tier')} (directly tied to spacecraft fire safety)" if inv.get("relevance_tier") == "A"
                else f"relevance tier {inv.get('relevance_tier')}",
        "richness": f"{inv.get('n_tests')} recorded tests",
        "gravity": ("flown in microgravity" if inv.get("type") == "flight" else "ground study"),
    }
    name = inv.get("acronym") or inv["id"]
    if not top:
        return f"{name}: little overlap with this scenario."
    return f"{name} ranks here mainly for " + "; ".join(phrases[k] for k in top) + "."


def rank(scenario: dict, weights: dict | None = None) -> list[dict]:
    """Rank all investigations for a habitat scenario.

    scenario: {"name", "o2_frac", "pressure_kPa", "flow_cm_s", "materials": [...],
               "pillar": "prevention|detection|suppression|None", optional "gravity"}
    returns:  [{"id", "acronym", "doi", "score", "breakdown": {...}, "why", "gaps": [...], "near"}] best first.
    """
    w = dict(DEFAULT_WEIGHTS)
    if scenario.get("pillar"):
        w["pillar"] = PILLAR_FOCUS_WEIGHT  # an explicit pillar is a strong user intent
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in DEFAULT_WEIGHTS})
    w["atmosphere"] = float(weights["atmosphere"]) if weights and "atmosphere" in weights else w["o2"] + w["pressure"]
    w = {k: v for k, v in w.items() if k not in ("o2", "pressure")}
    total_w = sum(w.values()) or 1.0

    results = []
    for inv in load_investigations():
        atm, near = _atmosphere_score(scenario, inv)
        breakdown = {
            "atmosphere": atm,
            "flow": _range_score(scenario.get("flow_cm_s"), inv.get("flow_range_cm_s"), FLOW_SCALE),
            "material": _material_score(scenario.get("materials"), inv),
            "pillar": _pillar_score(scenario.get("pillar"), inv),
            "tier": TIER_SCORE.get(inv.get("relevance_tier")),
            "richness": _richness(inv),
            "gravity": _gravity_score(scenario.get("gravity"), inv),
        }
        score = sum((breakdown[k] or 0.0) * w[k] for k in w) / total_w
        results.append({
            "id": inv["id"],
            "acronym": inv.get("acronym"),
            "doi": inv.get("doi"),
            "score": round(score, 4),
            "breakdown": {k: (None if v is None else round(v, 3)) for k, v in breakdown.items()},
            "why": _why(inv, breakdown, w, near),
            "gaps": _gaps(scenario, inv, breakdown, near),
            "near": near,  # {"n", "d_min", "nearest"} or None: the joint-atmosphere evidence behind the score
        })
    results.sort(key=lambda r: -r["score"])
    return results
