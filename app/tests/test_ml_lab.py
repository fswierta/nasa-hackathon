"""ML Lab page (issues #47, #59, owned by pc2). Run: python -m pytest app/tests/test_ml_lab.py -q

Uses the precomputed runs (processed/ai/mllab_runs.json), so no test trains a forest live.
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAGE = str(ROOT / "app" / "views" / "ml_lab.py")
TIMEOUT = 240


def page(query: dict | None = None):
    at = AppTest.from_file(PAGE, default_timeout=TIMEOUT)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    return at.run()


def features_on(at, ds="flex", target="outcome"):
    pre = f"ml_feat_{ds}_{target}_"
    return {c.key[len(pre):] for c in at.checkbox if c.key and c.key.startswith(pre) and c.value}


def test_opens_on_the_shipped_flex_model_with_registry_numbers():
    from ai.mllab import shipped_config
    at = page()
    assert not at.exception, at.exception
    assert any("ML Lab" in t.value for t in at.title)
    assert features_on(at) == set(shipped_config()["features"])
    assert at.slider(key="ml_rf_trees").value == 300 and at.slider(key="ml_repeats").value == 5
    acc = next(m for m in at.metric if m.label == "Accuracy (CV)")
    assert acc.value.startswith("0.765")
    assert any("This is the shipped model" in i.value and "0.765" in i.value for i in at.info)
    assert any("Liquid fuel droplets" in i.value for i in at.info)          # scope line
    assert not any("language model" in e.value for e in at.error)


def test_cfig_opens_on_normal_flame_self_extinguish_model():
    from analysis.scenario import GAS_FEATURES
    at = page()
    at.radio(key="ml_dataset").set_value("cfig").run()
    assert not at.exception, at.exception
    assert at.radio(key="ml_rows_cfig").value == "normal"
    assert features_on(at, "cfig", "self_extinguished") == set(GAS_FEATURES)
    assert any(m.label == "Accuracy (CV)" and m.value.startswith("0.869") for m in at.metric)
    # all flames: the burner-O2 caveat comes back
    at.radio(key="ml_rows_cfig").set_value("all").run()
    assert any("burner" in c.value for c in at.caption)


def test_leaderboard_ablation_and_inspection_shown_for_the_shipped_model():
    at = page()
    sub = [s.value for s in at.subheader]
    for title in ("Compare algorithms", "What does feature engineering add?", "What the model learned",
                  "Can the probabilities be read as percentages?", "Where the model is wrong"):
        assert title in sub, title
    board = next(d.value for d in at.dataframe if "Beats baseline" in d.value.columns)
    assert board.iloc[0]["Model"].startswith("Baseline") and len(board) == 5
    assert any("ships (Models hub)" in str(n) for n in board["Note"])
    errs = next(d.value for d in at.dataframe if "Data explorer" in d.value.columns)
    assert len(errs) == 10 and errs["Data explorer"].str.startswith("explorer?psi=PSI-69&test=").all()
    assert any(m.label.startswith("Brier score") for m in at.metric)


def test_leaking_feature_is_refused_and_blocks_training():
    at = page()
    at.checkbox(key="ml_feat_flex_outcome_burn_time_s").check().run()
    assert not at.exception, at.exception
    assert any("leak the answer" in e.value or "leaks the answer" in e.value for e in at.error)
    assert at.button(key="ml_train").disabled
    # the demo switch allows it, labelled
    at.checkbox(key="ml_leak_demo").check().run()
    assert any("LEAKAGE DEMO" in e.value for e in at.error)
    assert not at.button(key="ml_train").disabled


def test_training_adds_a_run_to_the_history():
    at = page()
    at.button(key="ml_train").click().run()
    assert not at.exception, at.exception
    hist = [d.value for d in at.dataframe if "feature list" in d.value.columns]
    assert len(hist) == 1 and len(hist[0]) == 1
    assert hist[0].iloc[0]["dataset"] == "flex" and hist[0].iloc[0]["accuracy"] > 0
    assert hist[0].iloc[0]["shipped"] == "yes"


def test_reset_restores_the_shipped_model():
    from ai.mllab import shipped_config
    at = page()
    at.checkbox(key="ml_feat_flex_outcome_x_he").uncheck().run()
    assert any("Not the shipped configuration" in c.value for c in at.caption)
    at.button(key="ml_reset").click().run()
    assert not at.exception, at.exception
    assert features_on(at) == set(shipped_config()["features"])


def test_config_link_opens_that_configuration():
    from ai.mllab import encode_config, shipped_config
    cfg = shipped_config("cfig_ignition")
    at = page({"config": encode_config(cfg)})
    assert not at.exception, at.exception
    assert at.radio(key="ml_dataset").value == "cfig" and at.selectbox(key="ml_target_cfig").value == "hot_ignition"
    assert any("Opened from a shared link" in i.value for i in at.info)
    assert any(m.label == "Accuracy (CV)" and m.value.startswith("0.808") for m in at.metric)
    assert any("?config=" in c.value for c in at.code)


def test_bad_config_link_falls_back_to_the_shipped_model():
    at = page({"config": "not-a-config"})
    assert not at.exception, at.exception
    assert any("could not be used" in w.value for w in at.warning)
    assert at.radio(key="ml_dataset").value == "flex"


def test_predict_tab_predicts_and_shows_nearest_tests():
    at = page()
    assert any(m.label == "Most likely" for m in at.metric)
    assert any("most similar real tests" in s.value for s in at.subheader)
