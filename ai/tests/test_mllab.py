"""ML Lab backend (issue #47). Run: python -m pytest ai/tests/test_mllab.py -q

Live training uses a small logistic-regression config (3-fold × 1) so these tests stay fast; the forest
reference numbers are read from the committed precomputed runs (processed/ai/mllab_runs.json).
"""
import pandas as pd
import pytest

from ai import mllab as M


def fast(dataset="flex", target=None, **over):
    cfg = M.default_config(dataset, target)
    cfg.update(model="logreg", params={"C": 1.0}, cv={"folds": 3, "repeats": 1, "seed": 0})
    cfg.update(over)
    return cfg


# ------------------------------------------------------------------ precomputed reference runs
def test_default_and_shipped_runs_are_precomputed_and_current():
    """Every default config + the shipped model opens instantly from the committed cache; the cache key
    includes a fingerprint of the dataset, so this fails if main's ETL changed the data (re-run --precompute)."""
    for cfg in M.precomputed_configs():
        r = M.run(cfg)
        assert r.get("from_cache"), f"{cfg['dataset']}/{cfg['target']}: run python -m ai.mllab --precompute"
        assert M.learning_curve(cfg)["valid_mean"]


def test_reference_numbers_reproduce():
    """Issue #47: cfig self_extinguished random forest ≈ 0.85 accuracy vs 0.67 baseline. FLEX outcome: the
    issue's '≈ 0.77' predates ETL round 2; the shipped model now scores 0.707 — the same as
    analysis.flex_classifier.evaluate() on today's data (checked by hand, 14 Nov)."""
    r = M.run(M.default_config("cfig", "self_extinguished"))
    assert 0.82 <= r["metrics"]["accuracy"]["mean"] <= 0.90
    assert 0.64 <= r["baseline"]["accuracy"]["mean"] <= 0.69
    s = M.run(M.shipped_config())
    assert 0.69 <= s["metrics"]["accuracy"]["mean"] <= 0.73 and s["n_rows"] == 261
    assert s["metrics"]["accuracy"]["mean"] > s["baseline"]["accuracy"]["mean"]


def test_shipped_config_mirrors_flex_classifier():
    from analysis import flex_classifier as F
    cfg = M.shipped_config()
    names = {"heptane": "is_heptane", "ln_p": "ln_pressure"}
    assert cfg["features"] == [names.get(f, f) for f in F.FEATURES]
    assert cfg["params"]["n_estimators"] == 300 and cfg["params"]["min_samples_leaf"] == 3
    assert cfg["class_weight"] == "balanced" and cfg["cv"] == {"folds": 5, "repeats": 5, "seed": F.SEED}


def test_cache_ignores_entries_for_other_data(monkeypatch):
    cfg = M.default_config("cfig")
    monkeypatch.setattr(M, "data_fingerprint", lambda ds: "different-data")
    assert M._disk().get(M._disk_key("run", cfg)) is None


# ------------------------------------------------------------------ leakage
def test_leaking_feature_is_refused():
    cfg = fast(features=["o2_frac", "burn_time_s"])
    with pytest.raises(M.FeatureRefused) as e:
        M.run(cfg)
    assert e.value.refused[0]["feature"] == "burn_time_s" and "leaks" in e.value.refused[0]["reason"]
    # catalog 'leak' lists also count (extinction diameter for the outcome target)
    assert M.check_features("flex", "outcome", ["extinction_diameter_mm"])


def test_leakage_demo_runs_and_is_flagged():
    r = M.run(fast(features=["o2_frac", "d0_mm", "burn_time_s"], leakage_demo=True))
    assert r["leakage_demo"] is True and r["refused"]


# ------------------------------------------------------------------ results + warnings
def test_classification_result_shape():
    r = M.run(fast())
    assert r["task"] == "classification" and r["n_rows"] > 200
    assert {"accuracy", "balanced_accuracy"} <= set(r["metrics"]) and "accuracy" in r["baseline"]
    cm = r["confusion_matrix"]
    assert sum(map(sum, cm["counts"])) == r["n_rows"]  # first CV repeat: every test predicted once
    assert set(r["recall"]) == set(r["classes"]) == {"extinction", "disruption", "completion"}
    assert {d["feature"] for d in r["permutation_importance"]} == set(r["config"]["features"])


def test_regression_result_and_readable_binary_labels():
    r = M.run(fast("flex", "burning_rate_k_mm2_s", model="ridge", params={"alpha": 1.0}))
    assert r["task"] == "regression" and {"r2", "mae"} <= set(r["metrics"])
    assert r["baseline"]["r2"]["mean"] <= 0.01  # mean predictor
    b = M.run(fast("cfig", "self_extinguished"))
    assert set(b["classes"]) == {"self-extinguished", "flow-terminated"}


def test_warnings():
    cfg = M.default_config("flex")  # default features include o2_frac and inert_frac = 1 - o2_frac
    X = pd.DataFrame({"o2_frac": [0.2, 0.3, 0.25, 0.21], "inert_frac": [0.8, 0.7, 0.75, 0.79]})
    w = " ".join(M.warnings_for(dict(cfg, features=["o2_frac", "inert_frac"]), X,
                                pd.Series(["extinction", "extinction", "completion", "extinction"])))
    assert "collinear pairs" in w and "o2_frac ~ inert_frac" in w and "samples per feature" in w and "only 1 examples" in w
    w = " ".join(M.run(fast("cfig", "burn_time_s", model="ridge", params={"alpha": 1.0}))["warnings"])
    assert "censored" in w and "burner" in w


def test_exclude_flagged_drops_corrected_rows():
    a, b = M.run(fast()), M.run(fast(exclude_flagged=True))
    assert b["n_rows"] < a["n_rows"] and any("ETL corrections excluded" in w for w in b["warnings"])


# ------------------------------------------------------------------ predict
def test_predict_in_range_out_of_range_and_neighbours():
    cfg = fast(features=["o2_frac", "pressure_kPa", "d0_mm", "is_heptane"])
    ok = M.predict(cfg, {"o2_frac": 0.21, "pressure_kPa": 101.3, "d0_mm": 3.0, "is_heptane": 1.0})
    assert ok["refused"] is None and abs(sum(ok["probs"].values()) - 1) < 1e-6
    assert len(ok["nearest"]) == 5 and all(n["test_id"] and n["actual"] for n in ok["nearest"])
    bad = M.predict(cfg, {"o2_frac": 0.21, "pressure_kPa": 56.5, "d0_mm": 3.0, "is_heptane": 1.0})
    assert bad["refused"] and "outside the tested range" in bad["refused"] and "probs" not in bad
    assert bad["nearest"]  # still shows what was actually tested nearby


def test_mutual_information_and_correlations():
    mi = M.mutual_information("cfig", "self_extinguished", ["is_inverse", "o2_frac", "tad_K"])
    assert len(mi) == 3 and all(d["mi"] >= 0 for d in mi)
    c = M.correlations("flex", ["o2_frac", "inert_frac", "fuel"])
    assert list(c.columns) == ["o2_frac", "inert_frac"] and c.loc["o2_frac", "inert_frac"] < -0.99


def test_nearest_tests_show_one_readable_outcome():
    """Regression: for FLEX the target 'outcome' is also a meta column; 'actual' must be one label."""
    cfg = fast(features=["o2_frac", "pressure_kPa", "d0_mm", "is_heptane"])
    near = M.predict(cfg, {"o2_frac": 0.21, "pressure_kPa": 101.3, "d0_mm": 3.0, "is_heptane": 1.0})["nearest"]
    assert all(n["actual"] in {"extinction", "disruption", "completion"} for n in near)
