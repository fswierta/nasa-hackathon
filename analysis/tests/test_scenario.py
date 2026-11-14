"""Run from repo root:  py -m pytest analysis/tests -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import scenario as sc  # noqa: E402

AIR = {"o2_frac": 0.21, "pressure_kPa": 101.3}


def test_droplet_prediction_and_refusal():
    a = sc.assess({"fuel_class": "droplet", "fuel": "heptane", **AIR})
    assert a["prediction"] and 0 <= a["prediction"]["p_goes_out"] <= 1 and a["sources"][0]["psi_id"] == "PSI-69"
    out = sc.assess({"fuel_class": "droplet", "fuel": "heptane", "o2_frac": 0.34, "pressure_kPa": 56.5})
    assert out["prediction"] is None and "outside the FLEX range" in out["why_no_prediction"]


def test_no_number_for_moon_or_mars():
    for g in ("lunar", "mars"):
        for s in ({"fuel_class": "droplet", "fuel": "heptane", **AIR}, {"fuel_class": "gas", "fuel": "propane", "fuel_frac": 0.3, **AIR}):
            a = sc.assess({**s, "gravity": g})
            assert a["prediction"] is None and "No partial-gravity" in a["why_no_prediction"]
            assert any(src["psi_id"] == "PSI-62" for src in a["sources"])


def test_solids_get_evidence_never_a_prediction():
    a = sc.assess({"fuel_class": "solid", "fuel": "SIBAL", "o2_frac": 0.30, "pressure_kPa": 70.3})
    assert a["prediction"] is None and a["tests_of_this_material"]["n"] > 30
    assert a["tests_of_this_material"]["o2_range"][1] < 0.23 and "no test of this material" in a["gap"]
    assert len(a["gravity_comparison"]) == 2 and a["gravity_comparison"][0]["microgravity"]["spread_rate"] == "2.1 mm/s"


def test_gas_model_beats_baseline_and_respects_range():
    card = sc.gas_model_card()
    for m in card["models"].values():
        assert m["cv_accuracy"] > m["baseline_accuracy"]
    a = sc.assess({"fuel_class": "gas", "fuel": "propane", "o2_frac": 0.30, "pressure_kPa": 70.3, "fuel_frac": 0.3})
    assert 0 <= a["prediction"]["p_hot_ignition"] <= 1
    assert sc.assess({"fuel_class": "gas", "fuel": "propane", "o2_frac": 0.85, "pressure_kPa": 101.3})["prediction"] is None  # burner O2 level


def test_suppression_evidence_matches_the_data():
    ev = sc.suppression_evidence()
    rich = ev["table"][-1]
    assert rich["CO2"]["share_went_out"] > rich["N2 only"]["share_went_out"]      # CO2 helps in O2-rich air (FLEX)
    assert all(row[a]["n"] >= 0 for row in ev["table"] for a in ("N2 only", "CO2", "He"))
    scan = sc.suppressant_scan("heptane", 0.30, 101.3, "CO2")
    ok = [p["p_goes_out"] for p in scan["points"] if p["p_goes_out"] is not None]
    assert ok[-1] > ok[0] and scan["points"][0]["o2_frac"] == 0.30 and scan["points"][-1]["o2_frac"] < 0.30


def test_misconceptions_are_verbatim_from_the_guide():
    chunks = " ".join(json.loads(l)["text"] for l in (ROOT / "processed/ai/chunks.jsonl").open(encoding="utf-8")
                      if "Guide" in l)
    norm = lambda t: "".join(t.lower().split())
    for fact in sc.RESPONSE_FACTS:
        assert norm(fact["claim"])[:60] in norm(chunks), fact["claim"]
    assert "not a crew procedure" in sc.DISCLAIMER
