"""Run from repo root:  py -m pytest etl/tests -q"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "etl"))

import parse_tests as pt  # noqa: E402


def test_nums_ignores_digits_in_formulas():
    assert pt.nums("~21% O2 with N2") == [21.0]
    assert pt.nums("LOW O2") == []
    assert pt.nums("100% CH4") == [100.0]
    assert pt.nums("0.17; 0.21;0.25; 0.3") == [0.17, 0.21, 0.25, 0.3]
    assert pt.nums(".33 cm") == [0.33]


def test_one_and_span():
    assert pt.one("20 cm/s") == 20.0
    assert pt.one("~ 21.5") == 21.5
    assert pt.one("n/a") is None
    assert pt.one("–") is None
    assert pt.span("1.5 - 5") == (1.5, 5.0)
    assert pt.span("") == (None, None)


def test_percent_span_fixes_excel_artifact():
    assert pt.percent_span("3400%", "t", 1, "c") == (0.34, None)
    assert pt.percent_span("21; 26.5; 40", "t", 1, "c") == (0.21, 0.4)


def test_clean_fuel_rejects_autofill_formulas():
    assert pt.clean_fuel("100%C2H4", "t", 1) == "100% C2H4"
    assert pt.clean_fuel("75 % Propylene", "t", 1) == "75% Propylene"
    assert pt.clean_fuel("40% CH7", "t", 1) is None


def test_material_and_geometry():
    assert pt.material_of("2 cm pmma sphere") == "PMMA"
    assert pt.material_of("PMMA & Nomex") == "PMMA + Nomex"
    assert pt.material_of("2 cm sphere") is None
    assert pt.geometry_of("1/4\" black rod") == "rod"


def _tests():
    path = ROOT / "processed" / "tests.csv"
    if not path.exists():
        pt.main()
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_output_invariants():
    rows = _tests()
    assert len(rows) > 1500
    for r in rows:
        for c in ("o2_frac", "o2_frac_max"):
            if r[c]:
                assert 0 < float(r[c]) <= 1, (r["investigation_id"], r["test_id"], c, r[c])
        for c in ("pressure_kPa", "pressure_kPa_max"):
            if r[c]:
                assert 10 < float(r[c]) < 1000, (r["investigation_id"], r["test_id"], r[c])
        if r["flow_cm_s"]:
            assert 0 <= float(r["flow_cm_s"]) < 200
        json.loads(r["raw_row"])


def test_every_tier_a_investigation_has_rows():
    ids = {r["investigation_id"] for r in _tests()}
    for pid in ["PSI-26", "PSI-25", "PSI-98", "PSI-99", "PSI-100", "PSI-102", "PSI-101", "PSI-69"]:
        assert pid in ids


def test_known_values():
    rows = {(r["investigation_id"], r["test_id"]): r for r in _tests()}
    flex1 = rows[("PSI-69", "1")]  # FLEX test 1: 757.7 mmHg, 0.21 O2, methanol, D0 1.69 mm
    assert abs(float(flex1["pressure_kPa"]) - 101.018) < 0.01
    assert flex1["o2_frac"] == "0.21" and flex1["fuel"] == "Methanol" and flex1["d0_mm"] == "1.69"
    s1 = rows[("PSI-98", "S1")]  # SAFFIRE-I S1: 20 cm/s, 21.5-21.7 % O2, 0.37 cm thick
    assert s1["flow_cm_s"] == "20.0" and s1["o2_frac"] == "0.215" and s1["thickness_mm"] == "3.7"


def test_coverage_gaps_are_backed_by_data():
    import build_coverage
    build_coverage.main()
    cov = json.loads((ROOT / "processed" / "coverage.json").read_text(encoding="utf-8"))
    hab = {h["name"]: h for h in cov["habitats"]}
    assert set(hab) >= {"ISS", "Lunar base", "Spacesuit"}
    assert hab["ISS"]["n_tests_near"] > 0
    assert hab["Lunar base"]["by_fuel_phase"]["solid"] == 0
    assert hab["ISS"]["o2_match_pressure_unknown"]["by_fuel_phase"]["solid"] > 0  # BASS-II/SAFFIRE at ~21 %
    assert all(g["text"] and g["evidence"] for g in cov["gaps"])
