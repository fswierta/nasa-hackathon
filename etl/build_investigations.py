"""Build processed/investigations.json (contract 1 in CLAUDE.md).

M1 version: merges investigations_catalog.json + metadata/PSI-XX.json + the hand-curated
facts from AGENT_BRIEF.txt section 4 (relevance tier, flight/ground, hardware family,
fuel phase). Condition ranges, fuels, varied/fixed variables are left null here and are
filled in by the table parsers (M2) via fill_from_tests().

Run from the repo root:  py etl/build_investigations.py   (or python ...)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "project-data-1-pc1" / "data"
OUT_DIR = ROOT / "processed"

# From AGENT_BRIEF.txt section 4. tier: A = core spacecraft fire safety, B = related, C = other.
# fuel_phase None where the brief does not state it clearly (never guess).
CURATED: dict[str, dict] = {
    "PSI-26":  {"type": "flight", "tier": "A", "family": "MSG flow duct", "phase": "solid"},
    "PSI-25":  {"type": "flight", "tier": "A", "family": "MSG flow duct", "phase": "solid"},
    "PSI-98":  {"type": "flight", "tier": "A", "family": "Cygnus (Saffire)", "phase": "solid"},
    "PSI-99":  {"type": "flight", "tier": "A", "family": "Cygnus (Saffire)", "phase": "solid"},
    "PSI-100": {"type": "flight", "tier": "A", "family": "Cygnus (Saffire)", "phase": "solid"},
    "PSI-102": {"type": "flight", "tier": "A", "family": "SAME hardware", "phase": "solid"},
    "PSI-101": {"type": "flight", "tier": "A", "family": "SAME hardware", "phase": "solid"},
    "PSI-69":  {"type": "flight", "tier": "A", "family": "MDCA in CIR", "phase": "droplet"},
    "PSI-68":  {"type": "flight", "tier": "B", "family": "MDCA in CIR", "phase": "droplet"},
    "PSI-39":  {"type": "flight", "tier": "B", "family": "MDCA in CIR", "phase": "droplet"},
    "PSI-47":  {"type": "flight", "tier": "B", "family": "SAME hardware", "phase": None},
    "PSI-20":  {"type": "flight", "tier": "B", "family": "ACME in CIR", "phase": "gas"},
    "PSI-22":  {"type": "flight", "tier": "B", "family": "ACME in CIR", "phase": "gas"},
    "PSI-159": {"type": "flight", "tier": "B", "family": "ACME in CIR", "phase": "gas"},
    "PSI-10":  {"type": "flight", "tier": "C", "family": "ACME in CIR", "phase": "gas"},
    "PSI-21":  {"type": "flight", "tier": "C", "family": "ACME in CIR", "phase": "gas"},
    "PSI-23":  {"type": "flight", "tier": "C", "family": "ACME in CIR", "phase": "gas"},
    "PSI-107": {"type": "flight", "tier": "C", "family": "MSG flow duct", "phase": "gas"},
    "PSI-106": {"type": "flight", "tier": "C", "family": "MSG flow duct", "phase": "gas"},
    # Ground studies have no PSI acronym; "label" is our own short name (shown as acronym).
    "PSI-62":  {"type": "ground", "tier": "B", "family": None, "phase": "solid", "label": "BASS FGM model (ground)"},
    "PSI-115": {"type": "ground", "tier": "B", "family": None, "phase": None, "label": "SAME smoke DNS (ground)"},
    "PSI-117": {"type": "ground", "tier": "C", "family": None, "phase": "droplet", "label": "FLEX cool-flame CFD (ground)"},
    "PSI-142": {"type": "ground", "tier": "C", "family": "PI lab hardware", "phase": None, "label": "Counterflow cool flames (ground)"},
    "PSI-60":  {"type": "ground", "tier": "C", "family": None, "phase": "gas", "label": "SPICE soot analysis (ground)"},
}

# Fields the table parsers (M2) fill in; null until then.
FROM_TESTS = ["fuels", "sample_geometry", "o2_range", "pressure_range_kPa", "flow_range_cm_s",
              "diluents", "measurements", "varied_variables", "fixed_variables"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text or None


def count_table_rows(files: list[str]) -> int | None:
    total, found = 0, False
    for name in files:
        path = DATA_DIR / "experimental_tables" / name
        if not path.exists():
            continue
        found = True
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        total += sum(1 for r in rows[1:] if any(c.strip() for c in r))
    return total if found else None


def build() -> list[dict]:
    catalog = load_json(DATA_DIR / "investigations_catalog.json")
    out = []
    for row in catalog:
        pid = row["accession"]
        meta_path = DATA_DIR / "metadata" / f"{pid}.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        cur = CURATED.get(pid)
        if cur is None:
            print(f"WARNING: {pid} not in CURATED table, tier/type left null", file=sys.stderr)
            cur = {"type": None, "tier": None, "family": None, "phase": None}

        related = (meta.get("relatedInvestigations") or row.get("related") or "").replace(",", " ").split()
        inv = {
            "id": pid,
            "acronym": row.get("acronym") or meta.get("investigationAcronym") or cur.get("label"),
            "title": row.get("title") or meta.get("title"),
            "type": cur["type"],
            "platform": row.get("platform") or meta.get("flightProgram") or None,
            "hardware": row.get("hardware") or meta.get("experimentHardware") or None,
            "hardware_family": cur["family"],
            "sub_research_area": row.get("subResearchArea") or meta.get("subResearchArea"),
            "fuel_phase": cur["phase"],
            **{k: None for k in FROM_TESTS},
            # non-blank table rows (may still include footnote rows); exact count from M2 parsers
            "n_tests": count_table_rows(meta.get("experimentalTableFiles") or []),
            "relevance_tier": cur["tier"],
            "related_ids": sorted(set(related) - {pid}),
            "doi": row.get("doi") or meta.get("doi"),
            "landing_page": row.get("landingPage") or meta.get("landingPage"),
            "start": row.get("start") or meta.get("investigationStartDate"),
            "end": row.get("end") or meta.get("investigationEndDate"),
            "keywords": clean(row.get("keywords") or meta.get("keywords")),
            "objective": clean(meta.get("objective")),
            "hypothesis": clean(meta.get("hypothesis")),
            "approach": clean(meta.get("approach")),
            "research_impacts": clean(meta.get("researchImpacts")),
            "contacts": meta.get("contacts") or [],
            "publications": [{"title": clean(p.get("title")), "authors": clean(p.get("authors")),
                              "doi": p.get("doi") or None} for p in meta.get("publications") or []],
            "license": meta.get("licenseName") or "CC0-1.0",
            "total_size_gb": row.get("totalSizeGB"),
            "n_files": row.get("nFiles"),
            "experimental_table_files": meta.get("experimentalTableFiles") or [],
        }
        out.append(inv)

    tier_order = {"A": 0, "B": 1, "C": 2, None: 3}
    out.sort(key=lambda i: (tier_order[i["relevance_tier"]], int(i["id"].split("-")[1])))
    return out


def fill_from_tests(invs: list[dict]) -> None:
    """Fill FROM_TESTS fields from processed/tests.csv once the M2 parsers write it."""
    tests_path = OUT_DIR / "tests.csv"
    if not tests_path.exists():
        return
    with tests_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["investigation_id"], []).append(r)

    def rng(values):
        nums = []
        for v in values:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        return [min(nums), max(nums)] if nums else None

    for inv in invs:
        tests = by_id.get(inv["id"])
        if not tests:
            continue
        inv["n_tests"] = len(tests)
        inv["o2_range"] = rng(t.get("o2_frac") for t in tests)
        inv["pressure_range_kPa"] = rng(t.get("pressure_kPa") for t in tests)
        inv["flow_range_cm_s"] = rng(t.get("flow_cm_s") for t in tests)
        fuels = sorted({t["fuel"] for t in tests if t.get("fuel")})
        inv["fuels"] = fuels or None
        diluents = sorted({t["diluent"] for t in tests if t.get("diluent")})
        inv["diluents"] = diluents or None
        geoms = sorted({t["geometry"] for t in tests if t.get("geometry")})
        inv["sample_geometry"] = geoms or None
        varied, fixed = [], []
        for col, name in [("o2_frac", "o2"), ("pressure_kPa", "pressure"), ("flow_cm_s", "flow"),
                          ("fuel", "fuel"), ("diluent", "diluent")]:
            vals = {t.get(col) for t in tests if t.get(col) not in (None, "")}
            if len(vals) > 1:
                varied.append(name)
            elif len(vals) == 1:
                fixed.append(name)
        inv["varied_variables"] = varied or None
        inv["fixed_variables"] = fixed or None


def main() -> None:
    invs = build()
    fill_from_tests(invs)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "investigations.json"
    out_path.write_text(json.dumps(invs, indent=1, ensure_ascii=False), encoding="utf-8")
    tiers = {t: sum(i["relevance_tier"] == t for i in invs) for t in "ABC"}
    print(f"wrote {out_path.relative_to(ROOT)}: {len(invs)} investigations, tiers {tiers}")


if __name__ == "__main__":
    main()
