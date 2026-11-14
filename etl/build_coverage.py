"""Build processed/coverage.json: which conditions the PSI tests cover, and where the gaps are.

Shape is the contract in TASKS.md (main, M3). Habitat atmospheres come from ai/habitats.py so the
condition map and the ranker use the same numbers. Every gap carries the evidence (counts) it is
based on, computed from processed/tests.csv - nothing here is typed in by hand except the wording.

Run from the repo root (after parse_tests.py and build_investigations.py):
    py etl/build_coverage.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "processed"
sys.path.insert(0, str(ROOT))

from ai.habitats import HABITATS  # noqa: E402

O2_TOL = 0.06        # +- mole fraction
P_TOL = 0.25         # +- relative pressure


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def row_range(r, col):
    lo, hi = f(r[col]), f(r.get(col + "_max"))
    if lo is None:
        return None
    return lo, hi if hi is not None else lo


def overlaps(rng, lo, hi):
    return rng is not None and rng[0] <= hi and rng[1] >= lo


def phase_txt(d: dict) -> str:
    return ", ".join(f"{n} {k}" for k, n in d.items())


def main() -> None:
    tests = list(csv.DictReader((OUT / "tests.csv").open(encoding="utf-8")))
    invs = json.loads((OUT / "investigations.json").read_text(encoding="utf-8"))
    inv = {i["id"]: i for i in invs}
    phase = {i["id"]: i["fuel_phase"] or "unknown" for i in invs}
    combustion = [t for t in tests if t["row_kind"] != "session"]
    # inverse flames: O2 is the burner stream, not the atmosphere -> never matched to a cabin atmosphere
    ambient = [t for t in combustion if t.get("o2_role") != "burner"]

    # ---- habitats
    habitats = []
    for name, h in HABITATS.items():
        o2, p = h["o2_frac"], h["pressure_kPa"]
        near = [t for t in ambient
                if overlaps(row_range(t, "o2_frac"), o2 - O2_TOL, o2 + O2_TOL)
                and overlaps(row_range(t, "pressure_kPa"), p * (1 - P_TOL), p * (1 + P_TOL))]
        # O2 matches but the table has no pressure (all solid-fuel tables): reported separately
        o2_only = [t for t in ambient if not t["pressure_kPa"]
                   and overlaps(row_range(t, "o2_frac"), o2 - O2_TOL, o2 + O2_TOL)]
        by_inv = Counter(t["investigation_id"] for t in near)
        by_phase = Counter(phase[t["investigation_id"]] for t in near)
        o2_phase = Counter(phase[t["investigation_id"]] for t in o2_only)
        habitats.append({
            "name": name, "o2_frac": o2, "pressure_kPa": p, "gravity": h.get("gravity"),
            "n_tests_near": len(near),
            "by_investigation": dict(by_inv.most_common()),
            "by_fuel_phase": {k: by_phase.get(k, 0) for k in ("solid", "droplet", "gas")},
            "o2_match_pressure_unknown": {
                "n": len(o2_only),
                "by_investigation": dict(Counter(t["investigation_id"] for t in o2_only).most_common()),
                "by_fuel_phase": {k: o2_phase.get(k, 0) for k in ("solid", "droplet", "gas")},
            },
            "source": h.get("source"),
        })

    # ---- per investigation
    per_inv = []
    for i in invs:
        rows = [t for t in tests if t["investigation_id"] == i["id"]]
        per_inv.append({
            "id": i["id"], "acronym": i["acronym"], "fuel_phase": i["fuel_phase"],
            "relevance_tier": i["relevance_tier"], "table_rows": len(rows),
            "has_o2": sum(bool(t["o2_frac"]) for t in rows),
            "has_pressure": sum(bool(t["pressure_kPa"]) for t in rows),
            "has_flow": sum(bool(t["flow_cm_s"]) for t in rows),
        })

    # ---- gaps (each with evidence computed from the data)
    solids = [t for t in combustion if phase[t["investigation_id"]] == "solid"]
    solid_o2 = [x for t in solids for x in (f(t["o2_frac"]), f(t["o2_frac_max"])) if x is not None]
    solid_p = [t for t in solids if t["pressure_kPa"]]
    all_o2 = [x for t in ambient for x in (f(t["o2_frac"]), f(t["o2_frac_max"])) if x is not None]
    flight = [i for i in invs if i["type"] == "flight"]
    gaps = []

    bre = [t for t in combustion if t["investigation_id"] == "PSI-20"]
    bre_o2 = [x for t in bre for x in (f(t["o2_frac"]), f(t["o2_frac_max"])) if x is not None]

    def bre_note(o2):
        if not bre_o2:
            return None
        days = sum(1 for t in bre if overlaps(row_range(t, "o2_frac"), o2 - O2_TOL, o2 + O2_TOL))
        return (f"ACME BRE (PSI-20) - a gas burner that emulates burning solids - ran at "
                f"{min(bre_o2) * 100:g}–{max(bre_o2) * 100:g} % O2 ({days} of {len(bre)} test days whose O2 range reaches "
                f"{o2 * 100:g} % ±{O2_TOL * 100:g} points); its pressures are not in the table.")

    # one gap per distinct atmosphere (Exploration atm / Mars base / rover share 34 % / 56.5 kPa)
    groups: dict[tuple, list[dict]] = {}
    for h in habitats:
        if h["name"] not in ("ISS", "Spacesuit"):
            groups.setdefault((h["o2_frac"], h["pressure_kPa"]), []).append(h)
    for (o2, p), hs in groups.items():
        h = hs[0]
        solid_any = h["by_fuel_phase"]["solid"] + h["o2_match_pressure_unknown"]["by_fuel_phase"]["solid"]
        if solid_any == 0:
            names = ", ".join(x["name"] for x in hs)
            gaps.append({
                "text": f"No solid-material test at the O2 level of {names} ({o2 * 100:g} % O2, {p:g} kPa).",
                "evidence": f"{h['n_tests_near']} tests near this atmosphere: {phase_txt(h['by_fuel_phase'])} "
                            f"(only droplets and gas flames); 0 solid-fuel rows within ±{O2_TOL * 100:g} percentage points of O2 even "
                            "ignoring pressure. Solids (fabrics, plastics) are the main fire load in a habitat.",
                "closest_proxy": bre_note(o2),
                "habitats": [x["name"] for x in hs],
            })
    if solid_o2:
        gaps.append({
            "text": f"Solid materials were tested only up to {max(solid_o2) * 100:.1f} % O2, "
                    f"below every planned exploration atmosphere (30–34 % O2).",
            "evidence": f"{len(solid_o2)} O2 values in "
                        f"{len({t['investigation_id'] for t in solids if t['o2_frac']})} solid-fuel investigations "
                        f"with an O2 column; range {min(solid_o2) * 100:.1f}–{max(solid_o2) * 100:.1f} %. "
                        f"The BASS (PSI-26) table records no O2 at all. Closest proxy: {bre_note(0.34)}",
        })
    gaps.append({
        "text": "No solid-material test has a recorded pressure in the PSI tables.",
        "evidence": f"{len(solid_p)} of {len(solids)} solid-fuel rows have a pressure value "
                    "(BASS, BASS-II, SAFFIRE and SAME tables have no pressure column).",
    })
    gaps.append({
        "text": "No partial-gravity (Moon 1/6 g, Mars 3/8 g) combustion data in PSI — and 0 g / 1 g results "
                "may not bound it.",
        "evidence": f"All {len(flight)} flight investigations ran in microgravity; ACME E-FIELD (PSI-22) uses electric "
                    "fields only as a proxy. The challenge text says 0 g and 1 g results 'could bound' partial gravity, "
                    "but that is not guaranteed: in microgravity, concurrent flame spread is most vigorous at an "
                    "intermediate, low flow speed (the 'Goldilocks' flammability zone of Olson and Ferkul, PSI-62), and "
                    "the weak buoyant flow at lunar or Martian gravity is also low speed, so a partial-gravity case "
                    "could be worse than both. Planned NASA experiments LUCI and FM2 are meant to measure it.",
    })
    sp = next(h for h in habitats if h["name"] == "Spacesuit")
    gaps.append({
        "text": f"Nothing was tested at spacesuit conditions ({sp['o2_frac'] * 100:g} % O2, {sp['pressure_kPa']:g} kPa).",
        "evidence": f"Highest atmosphere O2 in any test: {max(all_o2) * 100:g} %. (Inverse flames in PSI-10 / PSI-159 "
                    f"blow up to 85 % O2 out of the burner into a fuel atmosphere - that is not a cabin atmosphere.) "
                    f"{sp['n_tests_near']} tests near spacesuit conditions.",
        "habitat": "Spacesuit",
    })

    out = {
        "definition": f"near = O2 within ±{O2_TOL} mole fraction and pressure within ±{P_TOL * 100:g} % of the habitat value; "
                      "rows with a range count if the range overlaps.",
        "habitats": habitats, "gaps": gaps, "per_investigation": per_inv,
        "totals": {"rows": len(tests), "combustion_rows": len(combustion),
                   "rows_with_o2_and_pressure": sum(bool(t["o2_frac"] and t["pressure_kPa"]) for t in combustion)},
    }
    (OUT / "coverage.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote processed/coverage.json: {len(habitats)} habitats, {len(gaps)} gaps")
    for h in habitats:
        print(f"  {h['name']:24s} near={h['n_tests_near']:4d} {h['by_fuel_phase']}")
    for g in gaps:
        print("  GAP:", g["text"])


if __name__ == "__main__":
    main()
