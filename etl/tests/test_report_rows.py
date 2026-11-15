"""Rows taken from PSI reports (#63): every value must be on the cited page, verbatim.
Run from repo root:  py -m pytest etl/tests -q"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl"))

import report_rows as rr  # noqa: E402


def _norm(t: str) -> str:
    # the PDF writes the Greek Delta of "Delta hc" with a symbol-font glyph (private-use code point): drop it
    t = "".join(ch for ch in t if not 0xE000 <= ord(ch) <= 0xF8FF)
    return re.sub(r"\s+", " ", t).strip()


def test_every_quote_is_on_its_page():
    from pypdf import PdfReader
    reader = PdfReader(str(ROOT / rr.BRE_DIR / rr.BRE_REPORT))
    for tid, *_, page, quote in rr.BRE_ROWS:
        text = _norm(reader.pages[page - 1].extract_text() or "")
        assert _norm(quote) in text, (tid, page)
    assert _norm(rr.BRE_PMMA_QUOTE) in _norm(reader.pages[rr.BRE_PMMA_PAGE - 1].extract_text())


def test_numbers_in_rows_match_their_quote():
    rows = {r["test_id"]: r for r in csv.DictReader((ROOT / "processed" / "tests.csv").open(encoding="utf-8"))
            if r["row_kind"] == "report_test"}
    assert set(rows) == {r[0] for r in rr.BRE_ROWS}
    for tid, _, o2, p, unit, *_ in rr.BRE_ROWS:
        row, quote = rows[tid], json.loads(rows[tid]["raw_row"])["quote"]
        assert f"{o2:g}%" in quote and f"{p:g}" in quote
        assert abs(float(row["o2_frac"]) - o2 / 100) < 1e-9 and "R1:" in row["flags"]
    assert abs(float(rows["19070D1"]["pressure_kPa"]) - 56.537) < 0.01     # 8.2 psia = the exploration atmosphere
