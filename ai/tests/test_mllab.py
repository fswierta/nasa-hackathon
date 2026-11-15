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
    """Issue #47 reference numbers, under the pinned scikit-learn (analysis/requirements.txt):
    cfig self_extinguished ≈ 0.85 vs 0.67 baseline; shipped FLEX outcome model ≈ 0.77 vs 0.69."""
    r = M.run(M.default_config("cfig", "self_extinguished"))
    assert 0.82 <= r["metrics"]["accuracy"]["mean"] <= 0.90
    assert 0.64 <= r["baseline"]["accuracy"]["mean"] <= 0.69
    s = M.run(M.shipped_config("flex_outcome:three_class"))
    assert 0.74 <= s["metrics"]["accuracy"]["mean"] <= 0.79 and s["n_rows"] == 261
    assert s["metrics"]["accuracy"]["mean"] > s["baseline"]["accuracy"]["mean"]


def test_shipped_config_mirrors_flex_classifier():
    from analysis import flex_classifier as F
    cfg = M.shipped_config()
    names = {"heptane": "is_heptane", "ln_p": "ln_pressure"}
    assert cfg["features"] == [names.get(f, f) for f in F.FEATURES]
    assert cfg["params"]["n_estimators"] == 300 and cfg["params"]["min_samples_leaf"] == 3
    assert cfg["class_weight"] == "balanced" and cfg["cv"] == {"folds": 5, "repeats": 3, "seed": F.SEED}
    assert cfg["target"] == "went_out" and M.shipped_config("flex_outcome:three_class")["cv"]["repeats"] == 5


@pytest.mark.parametrize("model_id", ["flex_outcome", "flex_outcome:three_class", "cfig_ignition", "cfig_self_extinguish"])
def test_shipped_models_reproduce_the_registry(model_id):
    """#59: the Lab's shipped configuration gives the Models hub's numbers (analysis.registry.specs())."""
    cfg = M.shipped_config(model_id)
    r, spec = M.run(cfg), M.registry_spec(model_id)
    assert r.get("from_cache"), "run python -m ai.mllab --precompute"
    met = spec["metrics"]
    assert r["n_rows"] == met["n_train"]
    assert abs(r["metrics"]["accuracy"]["mean"] - met["cv_accuracy"]) < 0.0015
    assert abs(r["baseline"]["accuracy"]["mean"] - met["baseline_accuracy"]) < 0.0015
    assert abs(r["metrics"]["balanced_accuracy"]["mean"] - met["balanced_accuracy"]) < 0.0015
    assert M.is_shipped(cfg) == model_id and M.is_shipped(dict(cfg, model="logreg")) is None


def test_start_config_is_the_shipped_model_where_one_exists():
    assert M.start_config("flex") == M.shipped_config("flex_outcome")
    assert M.start_config("cfig") == M.shipped_config("cfig_self_extinguish")
    assert M.start_config("cfig")["rows"] == {"flame_type": "normal"}
    from analysis import scenario
    assert M.start_config("cfig")["features"] == scenario.GAS_FEATURES
    assert M.start_config("flex")["target"] == "went_out"  # the Models hub headline: P(goes out), 2 classes
    assert M.start_config("flex", "outcome") == M.shipped_config("flex_outcome:three_class")
    assert M.start_config("flex", "burning_rate_k_mm2_s") == M.default_config("flex", "burning_rate_k_mm2_s")


def test_feature_order_is_canonical():
    cfg = M.shipped_config()
    assert M._key(dict(cfg, features=list(reversed(cfg["features"])))) == M._key(cfg)
    assert M.canonical_features("flex", ["d0_mm", "o2_frac"]) == ["o2_frac", "d0_mm"]


def test_row_filter_keeps_normal_flames_and_drops_burner_warning():
    r = M.run(fast("cfig", "self_extinguished", features=["o2_frac", "pressure_kPa"], rows={"flame_type": "normal"}))
    assert r["n_rows"] < M.run(fast("cfig", "self_extinguished", features=["o2_frac", "pressure_kPa"]))["n_rows"]
    assert not any("burner" in w for w in r["warnings"])


def test_start_configs_have_leaderboard_ablation_and_inspection_precomputed():
    for cfg in M.start_configs():
        board = M.peek_leaderboard(cfg)
        assert board, f"{cfg['dataset']}/{cfg['target']}: run python -m ai.mllab --precompute"
        scores = [r["score"]["mean"] for r in board["rows"]]
        assert scores == sorted(scores, reverse=True) and board["pick"] in {r["model"] for r in board["rows"]}
        assert sum(r["is_current"] for r in board["rows"]) == 1
        abl = M.peek_ablation(cfg)
        assert abl and {b["set"] for b in abl["bars"]} == {"raw", "both", "engineered"}
        assert len({b["n_rows"] for b in abl["bars"]}) == 1  # same rows for all three sets
        assert M.peek_partial_dependence(cfg) and M.peek_diagnostics(cfg)


def test_shipped_flex_leaderboard_marks_the_shipped_forest():
    board = M.peek_leaderboard(M.shipped_config())
    rows = {r["model"]: r for r in board["rows"]}
    assert rows["random_forest"]["shipped"] == "flex_outcome" and rows["random_forest"]["is_current"]
    assert set(rows) == set(M.CLASSIFIERS) and board["baseline"]["score"]["mean"] <= 0.5


def test_partial_dependence_shapes():
    p = M.partial_dependence(M.shipped_config())
    assert p["of"] == "P(went out)" and 1 <= len(p["curves"]) <= 4
    for c in p["curves"]:
        assert len(c["x"]) == len(c["y"]) and all(0 <= y <= 1 for y in c["y"]) and len(c["rug"]) == p["n_rows"]
    assert {(m["x"], m["y"]) for m in p["maps"]} == {("o2_frac", "x_co2"), ("o2_frac", "ln_pressure")}
    m = p["maps"][0]
    assert len(m["z"]) == len(m["ys"]) and len(m["z"][0]) == len(m["xs"])
    gas = M.partial_dependence(M.shipped_config("cfig_self_extinguish"))
    assert [(m["x"], m["y"]) for m in gas["maps"]] == [("o2_frac", "ln_pressure")] and gas["pairs_missing"] == ["O₂ × CO₂"]


def test_diagnostics_calibration_and_errors():
    d = M.diagnostics(M.shipped_config())
    cal = d["calibration"]
    assert sum(b["n"] for b in cal["bins"]) == d["n_rows"] == 206
    assert 0 <= cal["brier"] <= 1 and cal["verdict"].split(" ")[0] in {"Yes", "Roughly", "No"}
    assert len(d["errors"]) == 10 and all(e["actual"] != e["predicted"] or e["p_actual"] < 0.5 for e in d["errors"])
    ps = [e["p_actual"] for e in d["errors"]]
    assert ps == sorted(ps)
    # same out-of-fold method as analysis.registry, so the shipped model's chart matches the Models hub
    hub = M.registry_spec("flex_outcome")["reliability"]
    assert [(b["p_predicted"], b["p_observed"], b["n"]) for b in cal["bins"]] == \
        [(b["p_predicted"], b["p_observed"], b["n"]) for b in hub]
    reg = M.diagnostics(fast("flex", "burning_rate_k_mm2_s", model="ridge", params={"alpha": 1.0}))
    assert reg["calibration"] is None and len(reg["errors"]) == 10


def test_config_links_round_trip_and_reject_bad_input():
    import base64
    import json
    for cfg in M.start_configs():
        assert M.decode_config(M.encode_config(cfg)) == cfg
    assert M.decode_config(M.share_query("cfig_ignition")["config"]) == M.shipped_config("cfig_ignition")

    def tok(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    good = M.shipped_config()
    for bad in ["", "%%%", "x" * 4000, tok([1, 2]), tok(dict(good, dataset="rockets")),
                tok(dict(good, features=["__import__('os')"])), tok(dict(good, model="svm")),
                tok(dict(good, params={"n_estimators": 10 ** 9})), tok(dict(good, cv={"folds": 1, "repeats": 1, "seed": 0})),
                tok(dict(good, rows={"flame_type": "normal"}))]:  # flex has no row filter
        with pytest.raises(ValueError):
            M.decode_config(bad)
    # unknown keys and parameters of other models are dropped, not trusted
    d = M.decode_config(tok(dict(good, evil=1, params={"n_estimators": 100, "C": 5})))
    assert "evil" not in d and d["params"] == {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2}


def test_run_record_is_reproducible_json():
    import json
    cfg = M.shipped_config()
    rec = M.run_record(cfg, M.run(cfg))
    assert rec["shipped_model"] == "flex_outcome" and rec["data"]["fingerprint_sha256_16"] == M.data_fingerprint("flex")
    assert M.decode_config(rec["config_token"]) == cfg and json.loads(json.dumps(rec, default=str))


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
