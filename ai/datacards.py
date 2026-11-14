"""Synthetic "data card" chunks: one per investigation, built from the PSI experimental tables (#19).

    python -m ai.datacards            # -> processed/ai/datacards.jsonl (then: python -m ai.vectors)
    python -m ai.datacards --show PSI-26

Why: factual questions ("which materials were tested in BASS?", "what O2 levels did FLEX use?") should hit
the real values from `processed/investigations.json` + `processed/tests.csv`, not a paragraph about an
igniter relay. Each card is plain sentences (so the extractive answer can quote one) with the source
`processed/investigations.json (PSI experimental table)` and the investigation's DOI. Every number is
computed here — re-run after main's ETL changes (the vector fingerprint then changes: rebuild vectors).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .config import AI_OUT, PROCESSED_DIR

CARDS_FILE = AI_OUT / "datacards.jsonl"
SOURCE_FILE = "processed/investigations.json (PSI experimental table)"


def _rng(r, fmt: str, scale: float = 1.0, unit: str = "") -> str | None:
    if not r or (r[0] is None and r[1] is None):
        return None
    lo = (r[0] if r[0] is not None else r[1]) * scale
    hi = (r[1] if r[1] is not None else r[0]) * scale
    if abs(hi - lo) < 1e-9:
        return f"{fmt.format(lo)}{unit}"
    a, b = fmt.format(lo), fmt.format(hi)
    for digits in (1, 2, 3):  # rounding must not hide a real range ("22 to 22" for 21.5–21.7)
        if a != b:
            break
        a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
    return f"{a} to {b}{unit}"


def _list(xs) -> str | None:
    xs = [str(x) for x in (xs or []) if str(x).strip()]
    return ", ".join(xs) if xs else None


def card_text(inv: dict, rows) -> str:
    """Plain-English sentences; every value comes from the ETL outputs."""
    pid, acr = inv["id"], inv.get("acronym") or inv["id"]
    name = f"{acr} ({pid})"
    kind = {"flight": "a flight investigation", "ground": "a ground-based study"}.get(inv.get("type"), "an investigation")
    s = [f"{name}, {inv.get('title')}, is {kind}" + (f" using the {inv['hardware_family']} hardware" if inv.get("hardware_family") else "")
         + (f" on the {inv['platform']}" if inv.get("platform") else "") + "."]
    n = inv.get("n_tests")
    s.append(f"{name} ran {n} tests according to its PSI experimental table ({n} rows)." if n else
             f"PSI has no experimental table for {name}; it is documented through reports and publications only.")
    phase = inv.get("fuel_phase")
    fuels = _list(inv.get("fuels"))
    if fuels:
        s.append(f"The fuels and sample materials tested in {name} were: {fuels}" + (f" ({phase} fuels)." if phase else "."))
    elif phase:
        s.append(f"{name} studied {phase} fuels; the table does not list specific materials.")
    geo = _list(inv.get("sample_geometry"))
    if geo:
        s.append(f"Sample geometries in {name} were: {geo}.")
    o2 = _rng(inv.get("o2_range"), "{:.0f}", 100, " percent")
    o2f = _rng(inv.get("o2_range"), "{:.2f}")
    s.append(f"The oxygen (O2) levels in {name} tests ranged from {o2} (mole fraction {o2f})." if o2 and " to " in o2
             else f"The oxygen (O2) level in {name} tests was {o2} (mole fraction {o2f})." if o2
             else f"The {name} table does not record the oxygen (O2) concentration.")
    p = _rng(inv.get("pressure_range_kPa"), "{:.0f}", 1, " kPa")
    patm = _rng(inv.get("pressure_range_kPa"), "{:.2f}", 1 / 101.325, " atm")
    s.append(f"The ambient pressure in {name} tests ranged from {p} ({patm})." if p and " to " in p
             else f"The ambient pressure in {name} tests was {p} ({patm})." if p
             else f"The {name} table does not record the ambient pressure.")
    fl = _rng(inv.get("flow_range_cm_s"), "{:g}", 1, " cm/s")
    s.append(f"The forced air flow speed (ventilation velocity) in {name} tests ranged from {fl}." if fl and " to " in fl
             else f"The forced air flow speed (ventilation velocity) in {name} tests was {fl}." if fl
             else f"The {name} table does not record the air flow speed.")
    dil = _list(inv.get("diluents"))
    if dil:
        s.append(f"Diluent or suppressant gases used in {name}: {dil}.")
    var, fix = _list(inv.get("varied_variables")), _list(inv.get("fixed_variables"))
    if var:
        s.append(f"Variables that changed between {name} tests: {var}" + (f"; held fixed: {fix}." if fix else "."))
    if rows is not None and len(rows):
        outs = Counter(x for x in rows["outcome"].dropna() if str(x).strip())
        if outs:
            s.append(f"Recorded test outcomes in {name}: " + ", ".join(f"{k} in {v} tests" for k, v in outs.most_common()) + ".")
        # outcome vs O2: answers "what O2 level stopped burning / extinguished the flame?"
        o2_by = rows.dropna(subset=["o2_frac"])
        o2_by = o2_by[o2_by["outcome"].notna()]
        ext = o2_by[o2_by["outcome"].astype(str).str.contains("extinct", case=False)]["o2_frac"]
        done = o2_by[o2_by["outcome"].astype(str).str.contains("complet|burn ?out|burned out", case=False)]["o2_frac"]
        if len(ext):
            # wording kept apart from the tested-range sentence ("levels") so each answers its own question
            s.append(f"In {name}, the flame was extinguished in {len(ext)} tests run at oxygen (O2) concentrations of "
                     f"{ext.min() * 100:.0f} to {ext.max() * 100:.0f} percent.")
        if len(done):
            s.append(f"The lowest oxygen (O2) concentration at which a {name} test still burned to completion was "
                     f"{done.min() * 100:.0f} percent ({len(done)} tests burned to completion).")
        for col, label, unit in (("d0_mm", "initial droplet diameter", " mm"), ("spread_rate_mm_s", "flame spread rate", " mm/s"),
                                 ("burn_time_s", "burn time", " s"), ("extinction_diameter_mm", "flame extinction diameter", " mm"),
                                 ("thickness_mm", "sample thickness", " mm")):
            v = rows[col].dropna() if col in rows else []
            if len(v) >= 2 and v.max() > v.min():
                s.append(f"The {label} in {name} ranged from {v.min():g} to {v.max():g}{unit} over {len(v)} tests.")
    rel = _list(inv.get("related_ids"))
    if rel:
        s.append(f"PSI lists these related investigations for {name}: {rel}.")
    return " ".join(s)


def build() -> list[dict]:
    import pandas as pd

    invs = json.load(open(PROCESSED_DIR / "investigations.json", encoding="utf-8"))
    tests = pd.read_csv(PROCESSED_DIR / "tests.csv")
    cards = []
    for inv in invs:
        rows = tests[tests["investigation_id"] == inv["id"]]
        cards.append({"id": f"datacard-{inv['id']}", "psi_id": inv["id"], "doi": inv.get("doi"), "file": SOURCE_FILE,
                      "page": None, "text": card_text(inv, rows), "kind": "datacard"})
    AI_OUT.mkdir(parents=True, exist_ok=True)
    with open(CARDS_FILE, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return cards


def load_cards() -> list[dict]:
    if not CARDS_FILE.exists():
        return []
    with open(CARDS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", metavar="PSI-ID")
    args = ap.parse_args(argv)
    cards = build()
    print(f"{len(cards)} data cards -> {CARDS_FILE}  (now run: python -m ai.vectors)")
    if args.show:
        print(next(c["text"] for c in cards if c["psi_id"] == args.show))


if __name__ == "__main__":
    main()
