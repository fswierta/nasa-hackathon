"""ML Lab page (issue #47, owned by pc2). Run: python -m pytest app/tests/test_ml_lab.py -q

Uses the precomputed default runs (processed/ai/mllab_runs.json), so no test trains a forest live.
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAGE = str(ROOT / "app" / "views" / "ml_lab.py")
TIMEOUT = 240


def page():
    return AppTest.from_file(PAGE, default_timeout=TIMEOUT).run()


def test_renders_flex_with_precomputed_metrics_next_to_baseline():
    at = page()
    assert not at.exception, at.exception
    assert any("ML Lab" in t.value for t in at.title)
    labels = [m.label for m in at.metric]
    assert "Accuracy (CV)" in labels and any(lab.startswith("Baseline") for lab in labels)
    assert any("Liquid fuel droplets" in i.value for i in at.info)          # scope line
    assert not any("language model" in e.value for e in at.error)


def test_renders_cfig_and_warns_about_burner_o2():
    at = page()
    at.radio(key="ml_dataset").set_value("cfig").run()
    assert not at.exception, at.exception
    assert any("burner" in c.value for c in at.caption)
    assert any("Accuracy (CV)" == m.label for m in at.metric)


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


def test_reset_to_shipped_model_loads_its_features():
    at = page()
    at.button(key="ml_reset").click().run()
    assert not at.exception, at.exception
    from ai.mllab import shipped_config
    on = {c.key.rsplit("_outcome_", 1)[1] for c in at.checkbox if c.key and c.key.startswith("ml_feat_flex_outcome_") and c.value}
    assert on == set(shipped_config()["features"])
    assert at.slider(key="ml_rf_trees").value == 300 and at.slider(key="ml_repeats").value == 5


def test_predict_tab_predicts_and_shows_nearest_tests():
    at = page()
    assert any(m.label == "Most likely" for m in at.metric)
    assert any("most similar real tests" in s.value for s in at.subheader)
