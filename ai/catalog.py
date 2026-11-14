"""Load the investigation list the AI layer works on.

Preferred source: `processed/investigations.json` (written by main / ETL, contract 1 in CLAUDE.md).
Until that exists we build the same shape from `investigations_catalog.json` + `metadata/*.json`
plus a SEED table transcribed from AGENT_BRIEF §4. Seed values are only what the brief states;
everything else stays None (never guessed). ETL values always win over the seed.
"""
from __future__ import annotations

import json
from functools import lru_cache

from .config import DATA_DIR, PROCESSED_DIR

ATM = 101.325
MMHG = 0.133322

# Transcribed from AGENT_BRIEF.txt §4 (tier, fuel phase/fuels, ranges explicitly stated there).
# Pillar mapping follows the brief's own labels (SOLIDS -> prevention, DETECTION, SUPPRESSION).
SEED: dict[str, dict] = {
    "PSI-26": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["SIBAL fabric", "PMMA", "Nomex", "Ultem", "candle"],
               "flow_range_cm_s": [0, 40], "diluents": ["N2"], "pillars": ["prevention", "suppression"], "hardware_family": "MSG flow duct"},
    "PSI-25": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["SIBAL fabric", "PMMA"],
               "pillars": ["prevention"], "hardware_family": "MSG flow duct"},
    "PSI-98": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["SIBAL fabric"], "o2_range": [0.215, 0.215],
               "flow_range_cm_s": [20, 20], "pillars": ["prevention"], "hardware_family": "Cygnus (Saffire)"},
    "PSI-99": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["silicone", "SIBAL fabric", "PMMA", "Nomex"],
               "o2_range": [0.22, 0.22], "flow_range_cm_s": [20, 20], "pillars": ["prevention"], "hardware_family": "Cygnus (Saffire)"},
    "PSI-100": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["SIBAL fabric"], "flow_range_cm_s": [30, 30],
                "pillars": ["prevention"], "hardware_family": "Cygnus (Saffire)"},
    "PSI-102": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["Teflon", "Kapton", "silicone rubber", "lampwick", "DBP"],
                "pillars": ["detection"], "hardware_family": "SAME hardware"},
    "PSI-101": {"relevance_tier": "A", "fuel_phase": "solid", "fuels": ["Kapton", "silicone", "lampwick", "Pyrell", "Teflon"],
                "pillars": ["detection"], "hardware_family": "SAME hardware"},
    "PSI-69": {"relevance_tier": "A", "fuel_phase": "droplet", "fuels": ["methanol", "heptane"], "o2_range": [0.12, 0.34],
               "diluents": ["CO2", "He"], "pillars": ["suppression"], "hardware_family": "CIR + MDCA"},
    "PSI-68": {"relevance_tier": "B", "fuel_phase": "droplet", "fuels": ["n-heptane", "iso-octane", "decane", "ethanol"],
               "o2_range": [0.17, 0.30], "pressure_range_kPa": [round(0.5 * ATM, 1), round(3 * ATM, 1)],
               "flow_range_cm_s": [0, 5], "pillars": ["suppression"], "hardware_family": "CIR + MDCA"},
    "PSI-39": {"relevance_tier": "B", "fuel_phase": "droplet", "fuels": ["n-dodecane", "iso-dodecane"],
               "pressure_range_kPa": [round(0.25 * ATM, 1), round(5 * ATM, 1)], "diluents": ["He"],
               "pillars": ["prevention"], "hardware_family": "CIR + MDCA"},
    "PSI-47": {"relevance_tier": "B", "fuel_phase": None, "fuels": [], "pillars": ["detection"], "hardware_family": "SAME hardware"},
    "PSI-20": {"relevance_tier": "B", "fuel_phase": "gas", "fuels": ["gas burner emulating solids"], "o2_range": [0.21, 0.40],
               "pillars": ["prevention"], "hardware_family": "CIR + ACME"},
    "PSI-22": {"relevance_tier": "B", "fuel_phase": "gas", "fuels": [], "pillars": ["prevention"], "hardware_family": "CIR + ACME"},
    "PSI-159": {"relevance_tier": "B", "fuel_phase": "gas", "fuels": ["ethane", "propane", "butane"], "o2_range": [None, 0.85],
                "pressure_range_kPa": [50.0, 300.0], "pillars": ["prevention"], "hardware_family": "CIR + ACME"},
    "PSI-10": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": ["ethylene", "methane"], "pillars": ["modeling"], "hardware_family": "CIR + ACME"},
    "PSI-21": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": [], "pillars": ["modeling"], "hardware_family": "CIR + ACME"},
    "PSI-23": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": [], "pillars": ["modeling"], "hardware_family": "CIR + ACME"},
    "PSI-107": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": ["methane", "propylene"], "pillars": ["detection"], "hardware_family": "MSG flow duct"},
    "PSI-106": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": [], "pillars": ["modeling"], "hardware_family": "MSG flow duct"},
    "PSI-62": {"relevance_tier": "B", "fuel_phase": "solid", "fuels": [], "pillars": ["modeling", "prevention"], "hardware_family": None},
    "PSI-115": {"relevance_tier": "B", "fuel_phase": None, "fuels": [], "pillars": ["modeling", "detection"], "hardware_family": None},
    "PSI-117": {"relevance_tier": "C", "fuel_phase": "droplet", "fuels": [], "pillars": ["modeling"], "hardware_family": None},
    "PSI-142": {"relevance_tier": "C", "fuel_phase": "droplet", "fuels": [], "pillars": ["modeling"], "hardware_family": None},
    "PSI-60": {"relevance_tier": "C", "fuel_phase": "gas", "fuels": [], "pillars": ["modeling"], "hardware_family": None},
}

FIELDS = ["id", "acronym", "title", "type", "platform", "hardware_family", "fuel_phase", "fuels",
          "sample_geometry", "o2_range", "pressure_range_kPa", "flow_range_cm_s", "diluents",
          "measurements", "varied_variables", "fixed_variables", "n_tests", "relevance_tier",
          "related_ids", "doi", "landing_page", "objective", "publications"]


def _read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _from_raw() -> list[dict]:
    catalog = _read_json(DATA_DIR / "investigations_catalog.json")
    out = []
    for row in catalog:
        pid = row["accession"]
        meta_path = DATA_DIR / "metadata" / f"{pid}.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        seed = SEED.get(pid, {})
        is_flight = "Microgravity" in (row.get("projectType") or "")
        inv = {k: None for k in FIELDS}
        inv.update({
            "id": pid,
            "acronym": row.get("acronym"),
            "title": row.get("title"),
            "type": "flight" if is_flight else "ground",
            "platform": row.get("platform") or None,
            "n_tests": row.get("experimentalTableRows") or None,
            "related_ids": (meta.get("relatedInvestigations") or row.get("related") or "").split() or [],
            "doi": row.get("doi"),
            "landing_page": row.get("landingPage"),
            "objective": meta.get("objective"),
            "publications": [{"title": p.get("title"), "doi": p.get("doi")} for p in meta.get("publications", [])],
        })
        for k in ("relevance_tier", "fuel_phase", "fuels", "o2_range", "pressure_range_kPa",
                  "flow_range_cm_s", "diluents", "hardware_family"):
            if k in seed:
                inv[k] = seed[k]
        out.append(inv)
    return out


@lru_cache(maxsize=1)
def load_investigations() -> tuple[dict, ...]:
    """Investigations in contract-1 shape; ETL output wins, raw catalog + seed fills gaps."""
    base = {inv["id"]: inv for inv in _from_raw()}
    etl_path = PROCESSED_DIR / "investigations.json"
    if etl_path.exists():
        for inv in _read_json(etl_path):
            merged = dict(base.get(inv["id"], {}))
            merged.update({k: v for k, v in inv.items() if v not in (None, [], "")})
            base[inv["id"]] = merged
    for inv in base.values():
        inv["pillars"] = SEED.get(inv["id"], {}).get("pillars", [])
    return tuple(sorted(base.values(), key=lambda i: int(i["id"].split("-")[1])))


def by_id() -> dict[str, dict]:
    return {inv["id"]: inv for inv in load_investigations()}
