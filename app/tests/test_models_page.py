"""Models hub + ML components (#58, parts 1-2 by main).   python -m pytest -q app/tests/test_models_page.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import registry  # noqa: E402
from app.nav import APP_DIR, SECTIONS, available  # noqa: E402

TIMEOUT = 300


def _page():
    return AppTest.from_file(str(APP_DIR / "views/models.py"), default_timeout=TIMEOUT).run()


def test_models_page_has_one_card_per_registry_entry():
    at = _page()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    for model in registry.specs():
        assert model["title"] in md, model["id"]
    assert "Transparent formula — not ML" in md                      # the ranking is never labelled as ML
    labels = {m.label: m.value for m in at.metric}
    assert labels["ML models"] == "4" and labels["Predictions for solids / Moon / Mars"] == "0"


def test_playground_predicts_explains_and_shows_real_tests():
    at = _page()
    assert any(m.label == "Flame goes out on its own" and m.value.endswith("%") for m in at.metric)
    assert len(at.dataframe) >= 1                                    # closest real tests
    assert len(at.get("plotly_chart")) >= 6                          # score bars, probabilities, gauges, explanation, importance, reliability
    at.selectbox[0].select("flex_burning_rate").run()
    assert not at.exception and any("mm²/s" in m.value for m in at.metric)


def test_models_page_is_first_in_the_ai_ml_group():
    assert "views/models.py" in [p for p, *_ in available()]
    assert SECTIONS["AI & ML"][0] == "views/models.py"


def test_prediction_card_refusal_is_a_designed_state():
    script = '''
import sys; sys.path.insert(0, r"%s")
from analysis import registry
from app import ml_ui
model = next(m for m in registry.specs() if m["id"] == "flex_outcome")
ml_ui.prediction_card(registry.predict("flex_outcome", fuel="Heptane", o2_frac=0.34, pressure_kPa=56.5), model, key="t")
''' % str(ROOT)
    at = AppTest.from_string(script, default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    info = " ".join(i.value for i in at.info)
    assert "No prediction" in info and "outside the FLEX range" in info
    assert not any(m.label == "Flame goes out on its own" for m in at.metric)   # no number when the model refuses
