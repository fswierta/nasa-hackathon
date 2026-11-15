"""Test rows taken from PSI *reports* where the experimental table lacks the value (#63 item 2).

ACME BRE (PSI-20): the PSI table is a day log without pressures ("The chamber pressures were also varied but
are not identified in this summary table", Final Report p. 24). The Final Report (2022) states O2 AND pressure
for five individual tests - the only PSI evidence of a solid-emulating flame at exploration atmospheres.

Rules: every row lists the page and a verbatim quote that contains its numbers; etl/tests/test_report_rows.py
re-reads the PDF and fails if a quote is not on that page. Nothing is inferred beyond unit conversion.
Rows get row_kind = "report_test" (so they never change n_tests of the day-log table) and a flag R1.
"""
from __future__ import annotations

import json

PSI_TO_KPA, BAR_TO_KPA = 6.894757, 100.0
BRE_REPORT = "PSI-20_Reports_Final_20220201.pdf"
BRE_DIR = "project-data-1-pc1/data/documents/PSI-20"

# (test id, date, o2 %, pressure value, unit, outcome, mass flux g/m2s, hc kJ/g, L kJ/g, page, verbatim quote)
BRE_ROWS = [
    ("19056F1", "2019-Feb-25", 27.0, 10.2, "psia", "self-extinguished at 45 s", 4.55, 24, 0.66, 25,
     "Flame growth from ignition to self-extinction in test 19056F1. The conditions are ethylene/N2, hc = 24 kJ/g, "
     "4.55 g/m2s, 27% O2, 10.2 psia, L = 0.66 kJ/g, extinction at 45 s."),
    ("19070K1", "2019-March-11", 26.5, 10.2, "psia", "steady burning (quasi-steady; ended by shutting off the fuel)", 2.3, 24, 3.0, 25,
     "Flame growth from ignition to steady state in test 19070K1. The conditions are ethylene/N2, hc = 24 kJ/g, "
     "2.3 g/m2s, 26.5% O2, 10.2 psia, L = 3.0 kJ/g, steady."),
    ("19070D1", "2019-March-11", 34.0, 8.2, "psia", "steady burning (quasi-steady; ended by shutting off the fuel)", 4.95, 24, 2.42, 26,
     "Flame growth from ignition to steady state in test 19070D1. The conditions are ethylene/N2, hc = 24 kJ/g, "
     "4.95 g/m2s, 34% O2, 8.2 psia, L = 2.42 kJ/g, steady."),
    ("19066D5", "2019-March-7", 34.0, 0.56, "bar", None, 1.76, None, None, 122,
     "Mar. 7 19066D5 25 34% 0.56 50% 1.76"),
    ("19070L1", "2019-March-11", 27.0, 0.7, "bar", None, 1.74, None, None, 122,
     "Mar. 11 19070L1 27% 0.7 1.74"),
]
# Interpretation quoted with the rows (p. 26), shown by the app next to them - not our wording:
BRE_PMMA_QUOTE = ("This emulation corresponds to a material with a heat of combustion of about 24 kJ/g and a heat of "
                  "gasification of about 3 kJ/g. As such it could represent PMMA as a material solid. This is a profound "
                  "result indicating a common plastic could steadily burn under these NASA proposed atmospheric conditions.")
BRE_PMMA_PAGE = 26


def bre_rows(columns: list[str]) -> list[dict]:
    out = []
    for tid, date, o2, p, unit, outcome, flux, hc, L, page, quote in BRE_ROWS:
        kpa = round(p * (PSI_TO_KPA if unit == "psia" else BAR_TO_KPA), 3)
        rec = {c: "" for c in columns}
        notes = [f"mass flux {flux} g/m2s"]
        if hc:
            notes.append(f"emulated solid: heat of combustion {hc} kJ/g, heat of gasification {L} kJ/g")
        notes.append(f"ACME/BRE Final Report (2022) p. {page}")
        rec.update(
            investigation_id="PSI-20", test_id=tid, row_kind="report_test", date=date,
            fuel="50% C2H4", material="ethylene/N2 (50 % mixture), gas burner emulating a burning solid",
            geometry="porous burner 25 mm", o2_frac=round(o2 / 100, 4), pressure_kPa=kpa, diluent="N2",
            outcome=outcome or "", notes="; ".join(notes), source_file=BRE_REPORT,
            flags=f"R1: row taken from the report, not the PSI table (p. {page}; pressure {p:g} {unit} = {kpa:g} kPa)",
            raw_row=json.dumps({"report": BRE_REPORT, "page": page, "quote": quote}, ensure_ascii=False),
        )
        out.append(rec)
    return out


def all_rows(columns: list[str]) -> list[dict]:
    return bre_rows(columns)
