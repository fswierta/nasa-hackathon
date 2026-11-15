"""Run from repo root:  py -m pytest analysis/tests -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import flex_classifier as fc  # noqa: E402


def test_training_data_is_flex_only_and_complete():
    rows = fc.load_rows()
    assert 250 <= len(rows) <= 274
    assert {r["outcome"] for r in rows} == set(fc.CLASSES)
    assert all(r["fuel"] in fc.FUELS and r["d0_mm"] > 0 and r["pressure_kPa"] > 50 for r in rows)


def test_model_beats_majority_baseline():
    card = fc.model_card()
    m = card["cv"][card["model"]]
    assert m["cv_accuracy"] > card["baseline_accuracy"]
    assert m["cv_balanced_accuracy"] > 1 / 3 + 0.1  # clearly better than chance on 3 classes


def test_prediction_shape_and_probabilities():
    r = fc.predict_outcome("methanol", 0.18, 101.3, x_co2=0.1, d0_mm=2.5)
    assert r["reason"] is None and r["predicted"] in fc.CLASSES
    assert abs(sum(r["probs"].values()) - 1) < 1e-6
    assert r["source"]["psi_id"] == "PSI-69"


def test_refuses_outside_tested_ranges():
    assert fc.predict_outcome("heptane", 0.21, 56.5)["probs"] is None          # below FLEX pressures
    assert fc.predict_outcome("heptane", 0.60, 101.3)["probs"] is None         # O2 above tested
    assert "not tested" in fc.predict_outcome("PMMA", 0.21, 101.3)["reason"]   # solids not in FLEX


def test_low_oxygen_methanol_goes_out():
    # FLEX methanol at low O2 overwhelmingly ended in extinction; the model must reflect that
    assert fc.predict_outcome("methanol", 0.14, 101.3)["probs"]["extinction"] > 0.7


def test_boundary_file_matches_contract():
    fc.main()
    d = json.loads((ROOT / "processed/analysis/flex_boundary.json").read_text(encoding="utf-8"))
    assert d["model_card"]["n_train"] > 250 and d["grids"]
    for g in d["grids"]:
        assert len(g["p_extinction"]) == len(g["x_diluent"]) and len(g["p_extinction"][0]) == len(g["o2_frac"])
        vals = [v for row in g["p_extinction"] for v in row if v is not None]
        assert all(0 <= v <= 1 for v in vals)
        assert all(t["outcome"] in fc.CLASSES for t in g["tests"])


def test_goes_out_model_finds_more_completions_and_beats_baseline():
    # #67: 3-class completion recall was 0.30. The 2-class model (disruption left out) must beat its
    # majority baseline on accuracy, keep balanced accuracy above the 3-class model and find more completions.
    b, card = fc.burnout_card(), fc.model_card()
    op = b["operating_points"]["default 0.5"]
    three = card["cv"][card["model"]]
    assert op["cv_accuracy"] > b["baseline_accuracy"]
    assert op["cv_balanced_accuracy"] > three["cv_balanced_accuracy"]
    assert op["recall_completion"] > three["recall"]["completion"] + 0.1
    assert b["roc_auc_completion"] > 0.75 and b["pr_auc_completion"] > 2 * b["pr_auc_baseline"]
    cons = b["operating_points"]["tuned"]
    assert cons["recall_completion"] >= op["recall_completion"]          # conservative = more completions found
    study = {s["strategy"] for s in fc.imbalance_card()}
    assert any("no weighting" in s for s in study) and any("2 classes" in s for s in study)


def test_prediction_carries_the_goes_out_score():
    r = fc.predict_outcome("heptane", 0.21, 101.3)
    b = r["burnout"]
    assert abs(b["p_goes_out"] + b["p_completion"] - 1) < 1e-6
    assert b["predicted"] in fc.BURNOUT_CLASSES and isinstance(b["may_burn_completely"], bool)
    assert 0.1 <= b["conservative_threshold"] <= 0.9
