"""Explainable scenario ranking of investigations (AGENT_BRIEF §6.2, contract 4 in CLAUDE.md).

score = Σ w_k · s_k / Σ w_k, every component s_k in [0, 1].
Deterministic and instant (no LLM) so the "why" is always reproducible.
Missing data scores 0 for that component and is reported, never guessed.
"""
from __future__ import annotations

import math

from .catalog import load_investigations

DEFAULT_WEIGHTS = {
    "o2": 0.20,
    "pressure": 0.15,
    "flow": 0.10,
    "material": 0.20,
    "pillar": 0.10,
    "tier": 0.15,
    "richness": 0.10,
}

PILLAR_FOCUS_WEIGHT = 0.45  # pillar weight when the scenario names a pillar

TIER_SCORE = {"A": 1.0, "B": 0.6, "C": 0.3}
SAFETY_PILLARS = {"prevention", "detection", "suppression"}

# Decay scales: how far outside the tested range a scenario can be before the component hits 0.
O2_SCALE = 0.10          # mole fraction
PRESSURE_SCALE = math.log(2.0)  # factor of 2 in pressure
FLOW_SCALE = 15.0        # cm/s

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


def _gaps(scenario, inv, breakdown):
    gaps = []
    if scenario.get("gravity") in ("lunar", "mars"):
        gaps.append("PSI has no partial-gravity combustion data; microgravity and 1 g results only bound this case.")
    for key, label in (("o2", "O2"), ("pressure", "pressure"), ("flow", "airflow")):
        if breakdown[key] is None:
            gaps.append(f"No {label} range on record for this investigation.")
    o2 = scenario.get("o2_frac")
    rng = inv.get("o2_range")
    if o2 is not None and rng and rng[1] is not None and o2 > rng[1]:
        gaps.append(f"Scenario O2 ({o2:.0%}) is above the highest tested ({rng[1]:.0%}).")
    return gaps


def _why(inv, breakdown, weights):
    contrib = {k: (v or 0.0) * weights.get(k, 0) for k, v in breakdown.items()}
    top = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][:3]
    phrases = {
        "o2": f"tested O2 {_fmt_range(inv.get('o2_range'), 'mole frac.')}",
        "pressure": f"tested pressure {_fmt_range(inv.get('pressure_range_kPa'), 'kPa', 3)}",
        "flow": f"tested airflow {_fmt_range(inv.get('flow_range_cm_s'), 'cm/s', 3)}",
        "material": "materials: " + ", ".join((inv.get("fuels") or [])[:4]) if inv.get("fuels") else "solid-fuel investigation",
        "pillar": "fire-safety focus: " + ", ".join(inv.get("pillars") or []),
        "tier": f"relevance tier {inv.get('relevance_tier')} (directly tied to spacecraft fire safety)" if inv.get("relevance_tier") == "A"
                else f"relevance tier {inv.get('relevance_tier')}",
        "richness": f"{inv.get('n_tests')} recorded tests",
    }
    name = inv.get("acronym") or inv["id"]
    if not top:
        return f"{name}: little overlap with this scenario."
    return f"{name} ranks here mainly for " + "; ".join(phrases[k] for k in top) + "."


def rank(scenario: dict, weights: dict | None = None) -> list[dict]:
    """Rank all investigations for a habitat scenario.

    scenario: {"name", "o2_frac", "pressure_kPa", "flow_cm_s", "materials": [...],
               "pillar": "prevention|detection|suppression|None", optional "gravity"}
    returns:  [{"id", "acronym", "doi", "score", "breakdown": {...}, "why", "gaps": [...]}] best first.
    """
    w = dict(DEFAULT_WEIGHTS)
    if scenario.get("pillar"):
        w["pillar"] = PILLAR_FOCUS_WEIGHT  # an explicit pillar is a strong user intent
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in DEFAULT_WEIGHTS})
    total_w = sum(w.values()) or 1.0

    results = []
    for inv in load_investigations():
        breakdown = {
            "o2": _range_score(scenario.get("o2_frac"), inv.get("o2_range"), O2_SCALE),
            "pressure": _range_score(scenario.get("pressure_kPa"), inv.get("pressure_range_kPa"), PRESSURE_SCALE, log=True),
            "flow": _range_score(scenario.get("flow_cm_s"), inv.get("flow_range_cm_s"), FLOW_SCALE),
            "material": _material_score(scenario.get("materials"), inv),
            "pillar": _pillar_score(scenario.get("pillar"), inv),
            "tier": TIER_SCORE.get(inv.get("relevance_tier")),
            "richness": _richness(inv),
        }
        score = sum((breakdown[k] or 0.0) * w[k] for k in w) / total_w
        results.append({
            "id": inv["id"],
            "acronym": inv.get("acronym"),
            "doi": inv.get("doi"),
            "score": round(score, 4),
            "breakdown": {k: (None if v is None else round(v, 3)) for k, v in breakdown.items()},
            "why": _why(inv, breakdown, w),
            "gaps": _gaps(scenario, inv, breakdown),
        })
    results.sort(key=lambda r: -r["score"])
    return results
