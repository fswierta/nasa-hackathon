"""Parse the 22 PSI experimental tables into processed/tests.csv (contract 2 in CLAUDE.md).

Every table has different columns, units and quirks (AGENT_BRIEF.txt section 5), so each
table has its own small parser below. Rules:
  * Normalize: O2 -> mole fraction (0-1), pressure -> kPa, flow -> cm/s, lengths -> mm.
  * A cell with a list/range ("0.17; 0.21; 0.3", "21.5 - 21.7") -> min in the main column,
    max in the *_max column. row_kind says whether a row is one test or a summary.
  * Never guess. A value not in the table stays empty; anything we could not parse goes to
    processed/etl_log.csv with the reason. The original row is kept in raw_row (JSON).

Run from the repo root:  py etl/parse_tests.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "project-data-1-pc1" / "data" / "experimental_tables"
OUT_DIR = ROOT / "processed"

COLUMNS = [
    "investigation_id", "test_id", "row_kind", "date", "fuel", "material", "geometry",
    "o2_frac", "o2_frac_max", "pressure_kPa", "pressure_kPa_max", "flow_cm_s", "flow_cm_s_max",
    "diluent", "outcome", "burn_time_s", "spread_rate_mm_s", "extinction_diameter_mm",
    "d0_mm", "burning_rate_k_mm2_s", "thickness_mm", "flow_direction", "notes",
    "source_file", "raw_row",
]
# row_kind values: test | test_matrix (planned condition grid) | test_day (day log, several tests)
#                  | session (non-combustion sampling session)

LOG: list[dict] = []

# ---------------------------------------------------------------- helpers

# Numbers not glued to letters, so the "2" in "O2"/"CH4" is not read as a value.
NUM_RE = re.compile(r"(?<![A-Za-z\d.])(?:\d+\.\d*|\.\d+|\d+)(?![A-Za-z]\d)")
FORMULA_RE = re.compile(r"\bC\d*H\d+\b")
VALID_FORMULAS = {"CH4", "C2H2", "C2H4", "C2H6", "C3H6", "C3H8", "C4H10", "C12H26"}
LEADING_ZERO_INT = re.compile(r"^\s*0\d+\s*(mm|cm)?\s*$")  # "037 mm": ambiguous, brief section 5


def log(src: str, row: int, column: str, value, issue: str) -> None:
    LOG.append({"source_file": src, "row": row, "column": column, "value": value, "issue": issue})


def nums(cell) -> list[float]:
    if cell is None:
        return []
    return [float(x) for x in NUM_RE.findall(str(cell))]


def span(cell) -> tuple[float | None, float | None]:
    v = nums(cell)
    return (min(v), max(v)) if v else (None, None)


def one(cell, src="", row=0, column="") -> float | None:
    """Exactly one number in the cell, else None (and log if the cell was not empty)."""
    text = (cell or "").strip()
    if not text or text.lower() in {"n/a", "na", "none", "ramped", "insignificant"} or set(text) <= set("-–— "):
        return None
    v = nums(text)
    if len(v) == 1:
        return v[0]
    log(src, row, column, text, "expected one number")
    return None


def percent_span(cell, src, row, column):
    """O2 given in percent -> (min, max) mole fraction. Fixes Excel '3400%' (= 34 %) artifacts."""
    text = (cell or "").strip()
    vals = []
    for v in nums(text):
        if v > 100 and "%" in text:
            log(src, row, column, text, f"Excel percent-format artifact, {v:g}% read as {v / 100:g}%")
            v /= 100
        if v > 100:
            log(src, row, column, text, "O2 percent > 100, dropped"); continue
        vals.append(v / 100)
    if not vals:
        return None, None
    lo, hi = min(vals), max(vals)
    return round(lo, 6), (None if hi == lo else round(hi, 6))


def clean_fuel(text, src, row, column="Fuel"):
    """Normalize spacing; reject spreadsheet auto-increment formulas (CH5, C2H11, ...)."""
    t = re.sub(r"\s*%", "%", " ".join((text or "").split()))
    t = re.sub(r"%(?=\S)", "% ", t)
    bad = [f for f in FORMULA_RE.findall(t) if f not in VALID_FORMULAS]
    if bad:
        log(src, row, column, text, f"invalid formula {bad} (spreadsheet auto-fill?), fuel left empty")
        return None
    return t or None


def scale(pair, factor):
    return tuple(None if x is None else round(x * factor, 6) for x in pair)


MMHG_TO_KPA, ATM_TO_KPA, BAR_TO_KPA, PSI_TO_KPA = 0.133322, 101.325, 100.0, 6.894757

MATERIAL_KEYS = [
    ("pmma", "PMMA"), ("sibal", "SIBAL"), ("cotton-fiberglass", "SIBAL"), ("nomex", "Nomex"),
    ("ultem", "Ultem"), ("japan wax", "Japan wax"), ("paraffin", "paraffin wax"),
    ("parrafin", "paraffin wax"), ("silicone", "silicone"), ("teflon", "Teflon"),
    ("kapton", "Kapton"), ("lampwick", "lampwick"), ("pyrell", "Pyrell"),
    ("dibutyl", "dibutyl phthalate"),
]
GEOMETRY_KEYS = [("sphere", "sphere"), ("spherical", "sphere"), ("rod", "rod"), ("slab", "slab"),
                 ("film", "thin film"), ("sheet", "sheet"), ("candle", "candle")]


def material_of(text: str) -> str | None:
    t = (text or "").lower()
    found = []
    for key, name in MATERIAL_KEYS:
        if key in t and name not in found:
            found.append(name)
    if not found and "wax" in t:
        found.append("wax")
    return " + ".join(found) or None


def geometry_of(text: str, default: str | None = None) -> str | None:
    t = (text or "").lower()
    for key, name in GEOMETRY_KEYS:
        if key in t:
            return name
    return default


def flow_dir_of(text: str) -> str | None:
    t = (text or "").lower()
    if "oppos" in t or "oppsed" in t:
        return "opposed"
    if "concurrent" in t:
        return "concurrent"
    return None


def read(name: str) -> tuple[list[str], list[list[str]]]:
    raw = (TABLES / name).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # FLEX table is Windows-1252 ("–" = no value)
        text = raw.decode("cp1252", errors="replace")
        log(name, 0, "*", "", "file is not UTF-8, decoded as cp1252")
    rows = list(csv.reader(text.splitlines()))
    header = [" ".join(h.split()) for h in rows[0]]
    body = [r + [""] * (len(header) - len(r)) for r in rows[1:] if any(c.strip() for c in r)]
    return header, body


def getter(header: list[str]):
    """get(row, 'substring', ...) -> value of the first column whose header contains a substring."""
    low = [h.lower() for h in header]

    def get(row, *subs):
        for s in subs:
            for i, h in enumerate(low):
                if s.lower() in h:
                    return row[i].strip()
        return ""
    return get


def base(pid, src, header, row, **fields) -> dict:
    rec = {c: "" for c in COLUMNS}
    rec.update(investigation_id=pid, source_file=src, row_kind="test",
               raw_row=json.dumps(dict(zip(header, row)), ensure_ascii=False))
    for k, v in fields.items():
        rec[k] = "" if v is None else v
    return rec


def dedupe(src, body):
    seen, out = set(), []
    for r in body:
        key = tuple(c.strip() for c in r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if len(out) < len(body):
        log(src, 0, "*", len(body) - len(out), "dropped exact duplicate rows")
    return out


# ---------------------------------------------------------------- per-table parsers
# Each returns a list of records. i = data row number (1-based, after header, blanks removed).

def p_bass(pid, src):  # PSI-26: only date, test number and material
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        text = g(r, "Fuel Sample Material")
        if "soot cleaning" in text.lower():
            log(src, i, "Fuel Sample Material", text, "not a burn test, skipped"); continue
        mat = material_of(text)
        if mat is None:
            log(src, i, "Fuel Sample Material", text, "material not named in table")
        out.append(base(pid, src, h, r, test_id=g(r, "As Run Test"), date=g(r, "Date"),
                        fuel=mat, material=text, geometry=geometry_of(text, "fabric" if mat == "SIBAL" else None),
                        flow_direction=flow_dir_of(text)))
    return out


def p_bass2(pid, src):  # PSI-25: O2 in % by volume; flow only as fan display settings (not cm/s)
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        text = g(r, "Fuel Sample Material"); mat = material_of(text)
        if mat is None and "film" in text.lower():
            mat = None  # e.g. "100 micron film" - material not named
        o2 = one(g(r, "initial O2"), src, i, "initial O2")
        out.append(base(pid, src, h, r, test_id=g(r, "Test #"), date=g(r, "Date"), fuel=mat,
                        material=text, geometry=geometry_of(text, "fabric" if mat == "SIBAL" else None),
                        o2_frac=None if o2 is None else o2 / 100,
                        notes=f"PI {g(r, 'PI')}; fan display {g(r, 'Fan display')}; air display {g(r, 'Air display')}"))
    return out


def p_saffire(pid, src):  # PSI-98/99/100
    h, body = read(src); g = getter(h); out = []
    body = dedupe(src, body)
    for i, r in enumerate(body, 1):
        sid, text = r[0].strip(), g(r, "Material")
        if sid.lower().startswith("note") or "flow visulation" in text.lower() or "flow visualization" in text.lower():
            continue  # footnote / camera-only rows
        thick_cell = r[2].strip()  # header misspelled in SAFFIRE-2 ("Samle Thickness")
        if LEADING_ZERO_INT.match(thick_cell):
            log(src, i, "Sample Thickness", thick_cell, "ambiguous leading-zero value, left empty")
            thick = None
        else:
            t = one(thick_cell, src, i, "Sample Thickness")
            unit_cm = "cm" in thick_cell or "(cm)" in h[2]
            thick = None if t is None else round(t * 10 if unit_cm else t, 4)
        length, width = one(r[3], src, i, h[3]), one(r[4], src, i, h[4])
        geom = f"sheet {length:g} x {width:g} cm" if length and width else "sheet"
        o2 = percent_span(g(r, "O2"), src, i, "Percent O2")
        flow = one(g(r, "Air Flow"), src, i, "Air Flow")
        burn = g(r, "Burn Time")
        # SAFFIRE-2: column 12 is "µg spread length" but holds a rate in mm/s
        spread = None
        if len(h) > 12 and "mm/s" in r[12]:
            spread = one(r[12], src, i, h[12])
        notes = []
        if len(h) > 13 and r[11].strip():
            notes.append(f"ug burn length {r[11].strip()}")
        if len(h) > 14:
            notes.append(f"1g burn length {r[13].strip()}; 1g spread rate {r[14].strip()}")
        mat = material_of(text)
        if not text:
            log(src, i, "Material", "", "material missing in table")
        out.append(base(pid, src, h, r, test_id=sid, fuel=mat, material=text, geometry=geom,
                        thickness_mm=thick, o2_frac=o2[0], o2_frac_max=o2[1],
                        flow_cm_s=flow, burn_time_s=one(burn) if "s" in burn else None,
                        spread_rate_mm_s=spread, flow_direction=flow_dir_of(g(r, "Direction")),
                        notes="; ".join(notes)))
    return out


def p_flex(pid, src):  # PSI-69: droplets, pressure mmHg, mole fractions
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        co2, he = one(g(r, "CO initial")), one(g(r, "He initial"))
        dil = ["N2"] + (["CO2"] if co2 else []) + (["He"] if he else [])
        p = one(g(r, "pressure"), src, i, "pressure")
        if p == 0:
            log(src, i, "Ambient pressure; mmHg", "0", "pressure 0 is physically impossible, left empty")
            p = None
        out.append(base(pid, src, h, r, test_id=g(r, "FLEX Test #"), date=g(r, "Test Date"),
                        fuel=g(r, "Fuel"), geometry="droplet",
                        o2_frac=one(g(r, "O initial"), src, i, "O2"),
                        pressure_kPa=None if p is None else round(p * MMHG_TO_KPA, 3),
                        diluent="+".join(dil), outcome=g(r, "Test end").lower(),
                        burn_time_s=one(g(r, "Burn time")),
                        extinction_diameter_mm=one(g(r, "extinction diameter")),
                        d0_mm=one(g(r, "initial diameter")),
                        burning_rate_k_mm2_s=one(g(r, "Burning rate")),
                        notes=f"FLEX id {g(r, 'FLEX Identifier')}; CO2 frac {co2}; He frac {he}"))
    return out


def p_flex2(pid, src):  # PSI-68: test matrix, every cell is a list/range
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        o2, p, fl, d0 = span(g(r, "Oxygen")), scale(span(g(r, "Pressure")), ATM_TO_KPA), \
            span(g(r, "Flow Velocity")), span(g(r, "Droplet Diameter"))
        mix = g(r, "Gas Mixture")
        out.append(base(pid, src, h, r, test_id=f"matrix-{i}", row_kind="test_matrix",
                        fuel=g(r, "Sample"), geometry="droplet",
                        o2_frac=o2[0], o2_frac_max=o2[1], pressure_kPa=p[0], pressure_kPa_max=p[1],
                        flow_cm_s=fl[0], flow_cm_s_max=fl[1], d0_mm=d0[0],
                        diluent="N2 or He" if "he" in mix.lower() else "N2",
                        notes=f"D0 range {g(r, 'Droplet Diameter')} mm; ambient {g(r, 'Temperature')} C"))
    return out


def p_cfi(pid, src):  # PSI-39: droplet cool flames, pressure "1 atm"
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        p = one(g(r, "Pressure"), src, i, "Pressure")
        he = one(g(r, "Helium"))
        out.append(base(pid, src, h, r, test_id=g(r, "Flex Number"), date=g(r, "Test Date"),
                        fuel=g(r, "Fuel"), geometry="droplet" + (" on fiber" if g(r, "Fiber").lower() == "yes" else ""),
                        o2_frac=one(g(r, "Oxygen"), src, i, "Oxygen"),
                        pressure_kPa=None if p is None else round(p * ATM_TO_KPA, 3),
                        diluent="He" if he else "N2", notes=f"CIR id {g(r, 'CIR')}"))
    return out


def p_same(pid, src):  # PSI-102: smoke from heated (pyrolysed) materials
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        text = g(r, "Sample Material")
        out.append(base(pid, src, h, r, test_id=g(r, "Sample #"), fuel=material_of(text), material=text,
                        flow_cm_s=one(g(r, "Air Flow")),
                        notes=f"weight loss {g(r, 'Weight loss')} mg/min; aging {g(r, 'Aging')} min"
                        if g(r, "Weight loss") else text))
    return out


def p_samer(pid, src):  # PSI-101
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        text = g(r, "Sample Type")
        out.append(base(pid, src, h, r, test_id=g(r, "SAME Test"), date=g(r, "Date"),
                        fuel=material_of(text), material=text, flow_cm_s=one(g(r, "Inlet velocity")),
                        notes=f"{g(r, 'Description')}; Rt/R0 {g(r, 'Rt/R0')}; net aging {g(r, 'Net aging')} s"))
    return out


def p_daft(pid, src):  # PSI-47: dust sampling sessions, no combustion
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(dedupe(src, body), 1):
        out.append(base(pid, src, h, r, test_id=g(r, "Session"), row_kind="session", date=g(r, "Actual Day"),
                        material=g(r, "Sample Material"),
                        notes=f"{g(r, 'Hardware')}; {g(r, 'Sampled Location')}"))
    return out


def p_bre(pid, src):  # PSI-20: one row per test day, O2 list in percent
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        o2 = percent_span(g(r, "Oxygen"), src, i, "Oxygen Conc (percent)")
        out.append(base(pid, src, h, r, test_id=f"day-{i}", row_kind="test_day",
                        date=f"{g(r, 'Year')}-{g(r, 'Month')}-{g(r, 'Date')}",
                        fuel=clean_fuel(g(r, "Fuel Bottle"), src, i), material=g(r, "Fuel Bottle"),
                        geometry=f"porous burner {g(r, 'Burner Size')} mm",
                        o2_frac=o2[0], o2_frac_max=o2[1],
                        diluent="N2", outcome=f"{g(r, 'Flames Ignited')} flames ignited",
                        notes=g(r, "Notes")))
    return out


def p_daylog(pid, src):  # PSI-21 CLD, PSI-22 E-FIELD, PSI-23 s-Flame, PSI-106 SLICE
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(body, 1):
        day = g(r, "Test Day", "Test Date")
        if not day and out:  # continuation row (e.g. extra fuel concentration)
            extra = "; ".join(f"{k}={v.strip()}" for k, v in zip(h, r) if v.strip())
            out[-1]["notes"] = (out[-1]["notes"] + "; " + extra).strip("; ")
            continue
        atm = g(r, "Atmosphere")
        o2 = percent_span(atm.split("O2")[0], src, i, "Atmosphere") if "O2" in atm else (None, None)
        dil = "+".join(d for d in ("N2", "He") if re.search(rf"\b{d}\b", atm, re.I)) or None
        if atm and re.search(r"\bN[3-9]\b", atm):
            log(src, i, "Atmosphere", atm, "invalid diluent (N3..., spreadsheet auto-fill?), diluent left empty")
        fuel_text = g(r, "Fuel Bottle", "Fuel Type")
        notes = [g(r, "Test Type")]
        for label in ("Mesh Polarity", "Fuel Conc", "Fuel MFC", "N2 MFC", "Notes"):
            v = g(r, label)
            if v:
                notes.append(f"{label}: {v}")
        ignited = g(r, "Flames ignited")
        out.append(base(pid, src, h, r, test_id=f"day-{day[:16]}", row_kind="test_day", date=day,
                        fuel=clean_fuel(fuel_text, src, i), material=fuel_text,
                        o2_frac=o2[0], o2_frac_max=o2[1],
                        diluent=dil, outcome=f"{ignited} flames ignited" if ignited else None,
                        notes="; ".join(n for n in notes if n)))
    return out


def p_cfig(pid, src):  # PSI-159 normal + inverse gaseous cool flames, pressure in bar
    h, body = read(src); g = getter(h); out = []
    inverse = "Inverse" in src
    for i, r in enumerate(body, 1):
        p = one(g(r, "approx p"), src, i, "approx p (bar)")
        o2 = one(g(r, "XO2 before", "XO2"), src, i, "XO2")
        hot, term = g(r, "Hot ignition").lower(), g(r, "Self -ext", "Self-ext")
        outcome = ("hot ignition" if hot == "yes" else "no hot ignition" if hot == "no" else hot)
        if term and term != "-":
            outcome += {"SE": ", self-extinguished", "FT": ", flow terminated"}.get(term, f", {term}")
        out.append(base(pid, src, h, r, test_id=g(r, "Test"), date=f"day {g(r, 'Day')}",
                        fuel=clean_fuel(g(r, "Fuel").lower(), src, i),
                        geometry=("inverse" if inverse else "normal") + " flame, spherical burner",
                        o2_frac=o2, pressure_kPa=None if p is None else round(p * BAR_TO_KPA, 3),
                        outcome=outcome, burn_time_s=one(g(r, "burn time")),
                        notes="; ".join(x for x in [
                            f"cool flame: {g(r, 'Cool flame')}" if g(r, "Cool flame") else "",
                            f"color {g(r, 'Color')}" if g(r, "Color") not in ("", "-") else "",
                            g(r, "Comments")] if x)))
    return out


def p_flamedesign(pid, src):  # PSI-10 normal + inverse, pressure atm, values may be "RAMPED"
    h, body = read(src); g = getter(h); out = []
    inverse = "Inverse" in src
    for i, r in enumerate(body, 1):
        p = one(g(r, "pmean")) or one(g(r, "pinit"))
        o2 = one(g(r, "XO2 (init)", "XO2"), src, i, "XO2")
        ramped = any(c.strip().upper() == "RAMPED" for c in r)
        out.append(base(pid, src, h, r, test_id=g(r, "Test"), date=f"day {g(r, 'Day')}",
                        fuel=g(r, "Fuel") or "C2H4",  # inverse table: fuel named in column headers (XC2H4)
                        geometry=("inverse" if inverse else "normal") + " flame, spherical burner",
                        o2_frac=o2, pressure_kPa=None if p is None else round(p * ATM_TO_KPA, 3),
                        diluent="N2", notes="flow rates ramped during test" if ramped else ""))
    return out


def p_spice(pid, src):  # PSI-107: gas jet flames in co-flow air
    h, body = read(src); g = getter(h); out = []
    for i, r in enumerate(dedupe(src, body), 1):
        out.append(base(pid, src, h, r, test_id=f"GMT{g(r, 'GMT date')}-{g(r, 'GMT ignition')}",
                        date=f"GMT day {g(r, 'GMT date')}", fuel=clean_fuel(g(r, "Fuel"), src, i),
                        geometry=f"jet nozzle {g(r, 'Nozzle')} mm", flow_cm_s=one(g(r, "Air Rate")),
                        notes=f"fuel rate {g(r, 'Fuel Rate (mg/s)')} mg/s; air co-flow"))
    return out


PARSERS = {
    "PSI-26_Experimental table_BASS.csv": ("PSI-26", p_bass),
    "PSI-25_Experimental table_BASS-II.csv": ("PSI-25", p_bass2),
    "PSI-98_Experimental table_SAFFIRE-1.csv": ("PSI-98", p_saffire),
    "PSI-99_Experimental table_SAFFIRE-2.csv": ("PSI-99", p_saffire),
    "PSI-100_Experimental table_SAFFIRE-3.csv": ("PSI-100", p_saffire),
    "PSI-69_Experimental table_FLEX.csv": ("PSI-69", p_flex),
    "PSI-68_Experimental table_Flex2.csv": ("PSI-68", p_flex2),
    "PSI-39_Experimental table_CFI.csv": ("PSI-39", p_cfi),
    "PSI-102_Experimental table_SAME.csv": ("PSI-102", p_same),
    "PSI-101_Experimental table_SAME-R.csv": ("PSI-101", p_samer),
    "PSI-47_Experimental table_DAFT.csv": ("PSI-47", p_daft),
    "PSI-20_Experimental table_ACME BRE.csv": ("PSI-20", p_bre),
    "PSI-21_Experimental table_CLD Flame.csv": ("PSI-21", p_daylog),
    "PSI-22_Experimental table_ACME E-FIELD Flames.csv": ("PSI-22", p_daylog),
    "PSI-23_Experimental table_ACME s-Flame.csv": ("PSI-23", p_daylog),
    "PSI-106_Experimental table_Slice.csv": ("PSI-106", p_daylog),
    "PSI-159_Experimental table_CFI-G_Normal Flames.csv": ("PSI-159", p_cfig),
    "PSI-159_Experimental table_CFI-G_Inverse Flames.csv": ("PSI-159", p_cfig),
    "PSI-10_Experimental table_Flame Design_Normal Flame.csv": ("PSI-10", p_flamedesign),
    "PSI-10_Experimental table_Flame Design_Inverse Flame.csv": ("PSI-10", p_flamedesign),
    "PSI-107_Experimental table_Spice.csv": ("PSI-107", p_spice),
}
SKIPPED = {"PSI-142_Experimental table_Data Summary Table.csv":
           "list of published datasets (figure references), not test conditions"}


def main() -> None:
    records = []
    for path in sorted(TABLES.glob("*.csv")):
        name = path.name
        if name in SKIPPED:
            log(name, 0, "*", "", f"skipped: {SKIPPED[name]}"); continue
        if name not in PARSERS:
            log(name, 0, "*", "", "no parser registered"); continue
        pid, fn = PARSERS[name]
        recs = fn(pid, name)
        records.extend(recs)
        print(f"{name:60s} {len(recs):4d} rows")

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "tests.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(records)
    with (OUT_DIR / "etl_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_file", "row", "column", "value", "issue"])
        w.writeheader()
        w.writerows(LOG)
    print(f"wrote processed/tests.csv ({len(records)} rows), processed/etl_log.csv ({len(LOG)} issues)")


if __name__ == "__main__":
    main()
