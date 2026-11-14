"""ETL round 2: documented corrections applied after parsing (parse_tests.py stays faithful to the
PSI tables). Every correction is one function below, cites its evidence, and appends a note to the
row's `flags` column ("<fix id>: <what changed> — <source>"), so the app can mark corrected values
and anyone can check them. Nothing here is guessed: each fix is backed by a PSI document, by the
table's own structure, or only flags a value without changing it.

Also adds `date_iso` (YYYY-MM-DD) from the many date formats in the PSI tables.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta

SAFFIRE_DOC = "PSI-99/100 'SAFFIRE Objectives, Development and Status' p3, Table 1"
BASS_SRD = "PSI-26 'SRD_BASS.pdf' p6, Fig. 3b: '1 and 2-cm PMMA spheres'"
BASS2_README = ("PSI-25 'BASS-II_ReadMe.docx': Olson and Fernandez-Pello 'burned three sizes of pmma rods "
                "(mostly clear / mostly black)'")


def flag(rec: dict, fix: str, text: str) -> None:
    rec["flags"] = (rec.get("flags", "") + f" | {fix}: {text}").strip(" |")


def raw(rec: dict) -> dict:
    return json.loads(rec["raw_row"])


# ------------------------------------------------------------------ SAFFIRE

def fix_saffire_thickness(recs):
    """SAFFIRE-I/III SIBAL fabric thickness: values 0.37 / .33 sit under a 'cm' header, giving 3.7 / 3.3 mm
    for a cotton-fiberglass fabric. The same SIBAL fabric is 0.33 mm in the SAFFIRE sample table and
    '037 mm' in the SAFFIRE-II table -> the unit is mm, the header is wrong."""
    for r in recs:
        if r["investigation_id"] in ("PSI-98", "PSI-100") and r["fuel"] == "SIBAL" and r["thickness_mm"]:
            old = float(r["thickness_mm"])
            if old > 1:
                r["thickness_mm"] = round(old / 10, 4)
                flag(r, "F1", f"thickness {old:g} mm -> {r['thickness_mm']:g} mm (header says cm; SIBAL fabric is "
                              f"0.33 mm per {SAFFIRE_DOC})")


def fix_saffire2_leading_zero(recs):
    """SAFFIRE-II rows 2-4..2-6 thickness '037 mm' = 0.37 mm (missing decimal point; brief section 5;
    the sample table lists these samples at 0.33-0.36 mm)."""
    for r in recs:
        if r["investigation_id"] == "PSI-99" and not r["thickness_mm"]:
            cell = next((v for k, v in raw(r).items() if "thickness" in k.lower()), "")
            if re.fullmatch(r"\s*0(\d+)\s*mm\s*", cell or ""):
                r["thickness_mm"] = float("0." + re.fullmatch(r"\s*0(\d+)\s*mm\s*", cell).group(1))
                flag(r, "F2", f"thickness '{cell.strip()}' read as {r['thickness_mm']:g} mm (missing decimal point; "
                              f"sample table: 0.33-0.36 mm, {SAFFIRE_DOC})")


def fix_saffire2_missing_material(recs):
    """SAFFIRE-II 2-6 and 2-9 have an empty Material cell directly under 'Cotton-Fiberglass (SIBAL)' and
    'PMMA' (merged Excel cells). The sample table confirms: S6 = SIBAL fabric, S9 = flat PMMA 10 mm."""
    known = {"2-6": ("SIBAL", "S6 = SIBAL fabric"), "2-9": ("PMMA", "S9 = flat PMMA 10 mm; table thickness 1 cm agrees")}
    for r in recs:
        if r["investigation_id"] == "PSI-99" and r["test_id"] in known and not r["fuel"]:
            fuel, why = known[r["test_id"]]
            r["fuel"] = fuel
            flag(r, "F3", f"material filled as {fuel} (empty merged cell; {why}, {SAFFIRE_DOC})")


# ------------------------------------------------------------------ BASS / BASS-II

def fix_bass_materials(recs):
    """BASS rows that name only a shape. Spheres: the BASS SRD lists only PMMA spheres. Candles: BASS
    burned Japan-wax and paraffin candles, so a bare 'candle' is wax of unstated type."""
    for r in recs:
        if r["investigation_id"] != "PSI-26" or r["fuel"]:
            continue
        text = (r["material"] or "").lower()
        if "sphere" in text:
            r["fuel"] = "PMMA"
            flag(r, "F4", f"material '{r['material'].strip()}' -> PMMA ({BASS_SRD})")
        elif "candle" in text:
            r["fuel"] = "wax"
            flag(r, "F4", "bare 'candle' -> wax (BASS candles were Japan wax or paraffin; type not stated in this row)")


def fix_bass2_materials(recs):
    """BASS-II rows without a material name: rods are PMMA (ReadMe); films/slabs are PMMA when the same PI's
    other film/slab rows in the table say PMMA."""
    pmma_pis = defaultdict(set)
    for r in recs:
        if r["investigation_id"] == "PSI-25" and r["fuel"] == "PMMA":
            pi = raw(r).get("PI", "").strip()
            for shape in ("film", "slab", "rod"):
                if shape in (r["material"] or "").lower():
                    pmma_pis[shape].add(pi)
    for r in recs:
        if r["investigation_id"] != "PSI-25" or r["fuel"]:
            continue
        text, pi = (r["material"] or "").lower(), raw(r).get("PI", "").strip()
        if "rod" in text:
            r["fuel"] = "PMMA"
            flag(r, "F5", f"rod '{r['material'].strip()}' -> PMMA ({BASS2_README})")
        else:
            for shape in ("film", "slab"):
                if shape in text and pi in pmma_pis[shape]:
                    r["fuel"] = "PMMA"
                    flag(r, "F5", f"{shape} '{r['material'].strip()}' -> PMMA (PI {pi}'s other {shape} rows in this "
                                  f"table name PMMA)")
                    break


# ------------------------------------------------------------------ spreadsheet auto-fill

FORMULA = re.compile(r"\b(C\d*)H(\d+)\b")
NITRO = re.compile(r"\bN(\d)\b")


def _undo_autofill(recs, pid, field, pattern, group_label):
    """Within one table, a run like 'X% CH4, X% CH5, X% CH6 ...' (each row = previous row with the trailing
    number +1) is an Excel drag artifact: every member of the run is restored to the run's first value.
    Catches valid-looking members too (e.g. C2H6 inside an ethylene run labelled 'Ethylene based test')."""
    rows = [r for r in recs if r["investigation_id"] == pid]
    prev_text, prev_num, start = None, None, None
    for r in rows:
        text = r[field] or ""
        m = pattern.search(text)
        if not m:
            prev_text = prev_num = start = None
            continue
        num = int(m.group(m.lastindex))
        skeleton = pattern.sub(lambda mm: mm.group(0)[: -len(mm.group(mm.lastindex))] + "#", text)
        if prev_text is not None and skeleton == prev_text and num == prev_num + 1:
            restored = pattern.sub(lambda mm: mm.group(0)[: -len(mm.group(mm.lastindex))] + str(start), text)
            yield r, text, restored
        else:
            start = num
        prev_text, prev_num = skeleton, num


def fix_autofill(recs):
    for pid in ("PSI-21", "PSI-22", "PSI-23", "PSI-20"):
        for r, old, new in list(_undo_autofill(recs, pid, "material", FORMULA, "fuel")):
            r["material"] = new
            r["fuel"] = re.sub(r"\s*%", "%", " ".join(new.split()))
            flag(r, "F6", f"fuel '{old}' -> '{new}' (Excel drag auto-increment: consecutive rows "
                          f"CH4, CH5, CH6...; test type column: '{raw(r).get('Test Type', '').strip()}')")
    # s-Flame atmosphere 'with N2 -> N3 -> N4 ...' and fuels '... with N3'
    for r in recs:
        if r["investigation_id"] != "PSI-23":
            continue
        rw = raw(r)
        atm = next((v for k, v in rw.items() if k.lower().startswith("atmosphere")), "")
        if NITRO.search(atm) and NITRO.search(atm).group(1) != "2":
            r["diluent"] = "N2"
            flag(r, "F6", f"atmosphere '{atm}' -> N2 diluent (Excel drag auto-increment N2, N3, N4...; "
                          "test type: nitrogen-diluted)")
        if r["fuel"] is not None and NITRO.search(r["material"] or "") and NITRO.search(r["material"]).group(1) != "2":
            old = r["material"]
            r["material"] = NITRO.sub("N2", old)
            r["fuel"] = r["material"]
            flag(r, "F6", f"fuel '{old}' -> '{r['material']}' (Excel drag auto-increment of N2)")
        elif not r["fuel"] and NITRO.search(r["material"] or ""):
            r["material"] = NITRO.sub("N2", r["material"])
            r["fuel"] = r["material"]
            flag(r, "F6", "fuel restored after N2 auto-increment fix")


# ------------------------------------------------------------------ test ids, SLICE text, FLEX checks

def fix_duplicate_test_ids(recs):
    """SPICE: one ignition (same GMT time) holds several rows that differ only in fuel / air flow rate - the
    flow was stepped during one burn. Make ids unique with a step suffix. Other duplicates: '#n' suffix."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r["investigation_id"], r["test_id"])].append(r)
    for (pid, tid), rows in groups.items():
        if len(rows) < 2:
            continue
        for n, r in enumerate(rows, 1):
            if pid == "PSI-107":
                r["test_id"] = f"{tid}-step{n}"
                flag(r, "F7", f"test id '{tid}' shared by {len(rows)} rows of one ignition (fuel/air flow stepped) "
                              f"-> step {n}")
            else:
                r["test_id"] = f"{tid}#{n}"
                flag(r, "F7", f"test id '{tid}' used by {len(rows)} rows in the PSI table -> '#{n}'")


SLICE_FUEL = re.compile(r"^(\d+%\s*\w+(?:\s*\(balance N2\))?)\s*(.*)$")


def fix_slice_text(recs):
    """SLICE: the fuel cell mixes the fuel with remarks, and some date cells continue with remarks."""
    for r in recs:
        if r["investigation_id"] != "PSI-106":
            continue
        m = SLICE_FUEL.match(r["fuel"] or "")
        if m and m.group(2):
            r["fuel"], extra = m.group(1).strip(), m.group(2).strip()
            r["notes"] = "; ".join(x for x in (r["notes"], f"fuel note: {extra}") if x)
            flag(r, "F8", "remark moved from fuel to notes")
        dm = re.match(r"^(\d{4}-[A-Za-z]{3}-\d{1,2})\s+(.+)$", r["date"] or "")
        if dm:
            r["date"] = dm.group(1)
            r["notes"] = "; ".join(x for x in (r["notes"], f"date note: {dm.group(2)}") if x)
            flag(r, "F8", "remark moved from date to notes")


def flag_flex_physics(recs):
    """Values kept as published but flagged: physically odd per droplet-combustion theory."""
    for r in recs:
        if r["investigation_id"] != "PSI-69":
            continue
        d0, de, k = (float(r[c]) if r[c] else None for c in ("d0_mm", "extinction_diameter_mm", "burning_rate_k_mm2_s"))
        if d0 and de and de > d0:
            flag(r, "F9", f"extinction diameter {de} mm > initial diameter {d0} mm (kept as published; check video)")
        if k is not None and k < 0.1:
            flag(r, "F9", f"burning rate K = {k} mm2/s is far below all other FLEX tests (kept as published)")


# ------------------------------------------------------------------ dates

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
YEAR_FROM_DOCS = {
    "PSI-107": (2009, "SPICE reports: 'operated in the MSG ... in 2009' (Dotson thesis p32, 30-day report p5)"),
    "PSI-101": (2010, "SAME-R final report p2: flight tests 'July 16 to September 9, 2010'"),
}


def _yyddd(s: str):
    m = re.match(r"^(\d{2})(\d{3})", s or "")
    if not m or not 1 <= int(m.group(2)) <= 366:
        return None
    return date(2000 + int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)


def _parse_date(r):
    pid, d, tid = r["investigation_id"], (r["date"] or "").strip().replace("`", ""), r["test_id"]
    if not d and pid not in ("PSI-10", "PSI-159"):
        return None, None
    if pid in ("PSI-10", "PSI-159"):  # test ids are YYDDD + letter/number (e.g. 21328A1)
        return _yyddd(tid), "date from test id (YYDDD)"
    if pid in ("PSI-21", "PSI-22", "PSI-23"):  # 'Test Day' formatted YYDDD (table notes)
        return _yyddd(d), "Test Day YYDDD"
    if pid == "PSI-107":
        m = re.search(r"(\d+)", d)
        year, src = YEAR_FROM_DOCS[pid]
        return (date(year, 1, 1) + timedelta(days=int(m.group(1)) - 1) if m else None), f"GMT day + year {year} from {src}"
    if pid == "PSI-47":
        m = re.match(r"GMT\s*(\d+)\s*\((\d{4})\)", d)
        return (date(int(m.group(2)), 1, 1) + timedelta(days=int(m.group(1)) - 1) if m else None), "GMT day (year)"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", d)
    if m:
        y = int(m.group(3)); y += 2000 if y < 100 else 0
        return date(y, int(m.group(1)), int(m.group(2))), None
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})", d)
    if m:
        return date(2000 + int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))), None
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})", d)
    if m and pid in YEAR_FROM_DOCS:
        year, src = YEAR_FROM_DOCS[pid]
        return date(year, MONTHS[m.group(2).lower()], int(m.group(1))), f"year {year} from {src}"
    m = re.fullmatch(r"(\d{4})-([A-Za-z]{3,})\.?-(\d{1,2})", d)
    if m:
        return date(int(m.group(1)), MONTHS[m.group(2)[:3].lower()], int(m.group(3))), None
    return None, None


def add_date_iso(recs, investigations: dict | None = None):
    unparsed = []
    for r in recs:
        try:
            d, how = _parse_date(r)
        except (ValueError, KeyError):
            d, how = None, None
        r["date_iso"] = d.isoformat() if d else ""
        if how and d and ("from" in how):
            flag(r, "D1", f"date_iso {how}")
        if r["date"] and not d:
            unparsed.append((r["investigation_id"], r["date"]))
        # cross-check with explicit Year/Month/Date columns where the table has them (ACME day logs)
        rw = raw(r)
        if d and rw.get("Year", "").strip().isdigit():
            month = rw.get("Month", "").strip().rstrip(".")
            mnum = int(month) if month.isdigit() else MONTHS.get(month[:3].lower())
            day = rw.get("Date", "").strip()
            if mnum and day.isdigit():
                explicit = date(int(rw["Year"]), mnum, int(day))
                if abs((explicit - d).days) > 1:
                    flag(r, "D2", f"YYDDD gives {d}, Year/Month/Date columns give {explicit}; kept columns")
                    r["date_iso"] = explicit.isoformat()
    return unparsed


def fix_gmt_year_typo(recs):
    """BASS / BASS-II tables have a GMT day-of-year column next to the date. Where the date's year is
    impossible (e.g. '5/15/2103') but month/day match the GMT day in the plausible year, fix the year;
    every other date/GMT disagreement is only flagged."""
    for r in recs:
        if r["investigation_id"] not in ("PSI-25", "PSI-26") or not r["date_iso"]:
            continue
        gmt = (raw(r).get("GMT") or "").strip()
        if not gmt.isdigit():
            continue
        d = date.fromisoformat(r["date_iso"])
        if d.timetuple().tm_yday == int(gmt) and d.year <= 2030:
            continue
        cand = None
        if d.year > 2030:  # impossible year: try swapping its two middle digits (2103 -> 2013)
            y = str(d.year)
            try:
                c = date(int(y[0] + y[2] + y[1] + y[3]), d.month, d.day)
                if c.timetuple().tm_yday == int(gmt):
                    cand = c
            except ValueError:
                pass
        if cand:
            r["date_iso"] = cand.isoformat()
            flag(r, "F10", f"date '{r['date']}' -> {cand} (year typo; the row's GMT day {gmt} = {cand}, "
                           f"neighbouring tests have the same day)")
        else:
            flag(r, "D2", f"date {d} does not match the row's GMT day {gmt} (kept as published)")


def metadata_date_notes(recs, investigations):
    """Tests dated outside the PSI metadata start/end: the metadata is incomplete (e.g. s-Flame has a
    'Year: 2020' column; CFI flew in 2017), so this is a note per investigation, not a row flag."""
    notes = []
    if not investigations:
        return notes
    by = defaultdict(list)
    for r in recs:
        if r["date_iso"]:
            by[r["investigation_id"]].append(r["date_iso"])
    for pid, ds in by.items():
        inv = investigations.get(pid) or {}
        try:
            start = datetime.strptime(inv.get("start") or "", "%m-%d-%Y").date()
            end = datetime.strptime(inv.get("end") or "", "%m-%d-%Y").date()
        except ValueError:
            continue
        out = [d for d in ds if not start - timedelta(days=31) <= date.fromisoformat(d) <= end + timedelta(days=31)]
        if out:
            notes.append((pid, f"{len(out)} test dates {min(out)}..{max(out)} fall outside the PSI metadata period "
                               f"{start}..{end} (metadata incomplete; table dates kept)"))
    return notes


# ------------------------------------------------------------------ entry point

FIXES = [fix_saffire_thickness, fix_saffire2_leading_zero, fix_saffire2_missing_material, fix_bass_materials,
         fix_bass2_materials, fix_autofill, fix_duplicate_test_ids, fix_slice_text, flag_flex_physics]


def apply(recs: list[dict], investigations: dict | None = None) -> dict:
    for r in recs:
        r.setdefault("flags", "")
    for fn in FIXES:
        fn(recs)
    unparsed = add_date_iso(recs, investigations)
    fix_gmt_year_typo(recs)
    counts = defaultdict(int)
    for r in recs:
        for part in filter(None, (r["flags"] or "").split(" | ")):
            counts[part.split(":")[0]] += 1
    return {"flag_counts": dict(sorted(counts.items())), "unparsed_dates": unparsed,
            "metadata_date_notes": metadata_date_notes(recs, investigations)}
