"""Run from repo root:  py -m pytest analysis/tests -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import registry as reg  # noqa: E402


def test_registry_describes_every_model_for_the_ui():
    models = {m["id"]: m for m in reg.describe()}
    assert set(models) == {"flex_outcome", "flex_burning_rate", "cfig_ignition", "cfig_self_extinguish", "similarity", "ranking"}
    assert models["ranking"]["kind"] == "formula" and "not machine learning" in models["ranking"]["algorithm"]
    for m in models.values():
        assert m["title"] and m["scope"] and m["limits"] and m["page"] and m["kind"] in {"ml", "unsupervised", "formula"}
    for mid in ("flex_outcome", "cfig_ignition", "cfig_self_extinguish"):
        met = models[mid]["metrics"]
        assert met["cv_accuracy"] > met["baseline_accuracy"]          # a model must beat its baseline to be listed as ML
        assert models[mid]["importance"] and models[mid]["reliability"]
        assert all(0 <= b["p_predicted"] <= 1 and 0 <= b["p_observed"] <= 1 for b in models[mid]["reliability"])
    json.dumps(reg.describe())                                          # serialisable for the UI


def test_predict_gives_value_nearest_tests_or_a_reason():
    ok = reg.predict("flex_outcome", fuel="Heptane", o2_frac=0.21, pressure_kPa=101.3)
    assert ok["ok"] and 0 <= ok["value"] <= 1 and len(ok["nearest_tests"]) == 5
    assert all(t["actual"] in ("extinction", "disruption", "completion") for t in ok["nearest_tests"])
    no = reg.predict("flex_outcome", fuel="Heptane", o2_frac=0.34, pressure_kPa=56.5)
    assert not no["ok"] and no["value"] is None and "outside the FLEX range" in no["reason"]
    rate = reg.predict("flex_burning_rate", fuel="Heptane", o2_frac=0.21, pressure_kPa=101.3)
    assert rate["ok"] and rate["prediction"]["k_low"] <= rate["value"] <= rate["prediction"]["k_high"]


def test_explain_is_a_what_if_against_the_median():
    ex = reg.explain("flex_outcome", fuel="Heptane", o2_frac=0.30, pressure_kPa=101.3, x_co2=0.25)
    assert {e["feature"] for e in ex} >= {"o2_frac", "x_co2", "d0_mm"}
    assert ex == sorted(ex, key=lambda e: -abs(e["effect"])) and all(-1 <= e["effect"] <= 1 for e in ex)
    assert reg.explain("flex_outcome", fuel="Heptane", o2_frac=0.34, pressure_kPa=56.5) == []   # no prediction, nothing to explain


def test_habitat_predictions_never_invent_numbers():
    hp = {h["habitat"]: h for h in reg.habitat_predictions()}
    assert set(hp) >= {"ISS", "Lunar base", "Mars base", "Spacesuit"}
    assert hp["ISS"]["at_habitat_gravity"] is None and hp["ISS"]["in_microgravity"]["droplet_goes_out"][0]["value"] is not None
    for name in ("Lunar base", "Mars base"):
        g = hp[name]["at_habitat_gravity"]
        assert g["prediction"] is None and "No partial-gravity" in g["statement"]
    for h in hp.values():
        assert h["solids"]["prediction"] is None and h["solids"]["reason"]
        for group in h["in_microgravity"].values():
            assert all((p["value"] is None) != (p["reason"] is None) for p in group)   # a number or a reason, never both / neither
    suit = hp["Spacesuit"]["in_microgravity"]
    assert all(p["value"] is None for group in suit.values() for p in group)          # 100 % O2 was never tested
