"""Smoke tests: every page renders against the real processed/ data without an exception.

    python -m pytest -q app/tests
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.nav import APP_DIR, available  # noqa: E402

TIMEOUT = 120


def test_main_renders_home():
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("Flame in Freefall" in t.value for t in at.title)
    labels = [m.label for m in at.metric]
    assert {"Investigations", "Test records", "Documents indexed"} <= set(labels)
    assert at.metric[0].value == "24"


@pytest.mark.parametrize("path", [p for p, *_ in available() if p != "views/home.py"])
def test_page_renders(path):
    at = AppTest.from_file(str(APP_DIR / path), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception


def test_overview_filters():
    at = AppTest.from_file(str(APP_DIR / "views/overview.py"), default_timeout=TIMEOUT).run()
    assert at.dataframe[0].value.shape[0] == 24
    at.multiselect[1].select("A").run()            # relevance tier A
    assert at.dataframe[0].value.shape[0] == 8
    at.text_input[0].input("PMMA").run()
    assert set(at.dataframe[0].value["Investigation"]) == {"BASS", "BASS-II", "SAFFIRE-II"}


@pytest.mark.parametrize("pid", ["PSI-26", "PSI-69", "PSI-60", "PSI-142"])
def test_detail_every_kind(pid):
    """Flight with table, droplets, ground study without table, table with no tests."""
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = pid
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox[0].value == pid


def test_detail_all_24_render():
    import json
    ids = [i["id"] for i in json.loads((ROOT / "processed/investigations.json").read_text(encoding="utf-8"))]
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT).run()
    for pid in ids:
        at.selectbox[0].select(pid).run()
        assert not at.exception, (pid, at.exception)


def test_detail_with_ai_profile(tmp_path, monkeypatch):
    """Profile branch (contract 3) renders, using a throwaway profile in a temp dir."""
    import json

    import ai.profiles
    prof = {"id": "PSI-26", "summary_plain": "PLAIN", "summary_expert": "EXPERT",
            "key_findings": [{"text": "FINDING", "source": {"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80",
                                                           "file": "documents/PSI-26/x.pdf", "page": 3}}],
            "fire_safety_pillar": ["prevention"], "habitat_relevance": {"iss": 0.9},
            "earth_applications": [], "measurements": ["video", {"name": "radiometer"}]}
    (tmp_path / "PSI-26.json").write_text(json.dumps(prof), encoding="utf-8")
    monkeypatch.setattr(ai.profiles, "PROFILES_DIR", tmp_path)
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "FINDING" in text and "radiometer" in text


def test_ranker_presets_and_no_data():
    at = AppTest.from_file(str(APP_DIR / "views/ranker.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    from ai.habitats import HABITATS
    for preset in HABITATS:
        at.selectbox[0].select(preset).run()
        assert not at.exception, (preset, at.exception)
        assert at.slider[0].value == HABITATS[preset]["o2_frac"] * 100   # preset values loaded
    captions = " ".join(c.value for c in at.caption)
    assert "no data" in captions          # missing components are labelled, not shown as 0
    at.selectbox[1].select("detection").run()
    assert not at.exception, at.exception


def test_condition_map_focus():
    at = AppTest.from_file(str(APP_DIR / "views/condition_map.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    at.selectbox[0].select(at.selectbox[0].options[1]).run()
    assert not at.exception, at.exception


def test_ask_example_question_cites_sources():
    at = AppTest.from_file(str(APP_DIR / "views/ask.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    at.button[1].click().run()                     # "Which materials were tested in Saffire?"
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "**Sources**" in md
    assert "doi.org" in md                         # [n] markers became links
    at.chat_input[0].set_value("smoke particle size").run()
    assert not at.exception, at.exception


def test_compare_marks_differences():
    at = AppTest.from_file(str(APP_DIR / "views/compare.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    df = at.dataframe[0].value
    assert df.shape[1] == 2 + 3                    # label, ≠, three investigations
    assert (df["≠"] == "≠").any()
    at.toggle[0].set_value(True).run()
    assert (at.dataframe[0].value["≠"] == "≠").all()


@pytest.mark.parametrize("payload", [
    {"insights": [{"pillar": "prevention", "title": "T1", "text": "X",
                   "citations": [{"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "page": 2}],
                   "habitats": ["ISS"]}],
     "earth_applications": [{"title": "Mist", "text": "Y", "citations": ["PSI-69"]}]},
    [{"pillar": "detection", "title": "T2", "text": "Z", "citations": ["PSI-102"]}],
])
def test_insights_shapes(tmp_path, monkeypatch, payload):
    """Insights page accepts both likely insights.json shapes (pc2 has not shipped it yet)."""
    import json

    from app import data as app_data
    f = tmp_path / "insights.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_data, "PROCESSED", tmp_path.parent)
    (tmp_path.parent / "ai").mkdir(exist_ok=True)
    (tmp_path.parent / "ai" / "insights.json").write_text(json.dumps(payload), encoding="utf-8")
    at = AppTest.from_file(str(APP_DIR / "views/insights.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "T1" in text or "T2" in text


def test_streamlit_secret_reaches_ai_layer(monkeypatch):
    """On Streamlit Cloud the key arrives via st.secrets; main.py must put it in os.environ."""
    import os
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT)
    at.secrets["ANTHROPIC_API_KEY"] = "sk-test-not-real"
    at.run()
    assert not at.exception, at.exception
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-not-real"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_ask_question_from_url():
    at = AppTest.from_file(str(APP_DIR / "views/ask.py"), default_timeout=TIMEOUT)
    at.query_params["q"] = "Which materials were tested in Saffire?"
    at.run()
    assert not at.exception, at.exception
    assert "**Sources**" in " ".join(m.value for m in at.markdown)


def test_loaders_reread_changed_files(tmp_path):
    """#20: st.cache_data ignores parameters starting with '_', so the mtime must not be '_m'."""
    import json
    import os

    from app import data as app_data
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"v": 1}), encoding="utf-8")
    assert app_data._json_or_none(str(f), app_data._mtime(f)) == {"v": 1}
    f.write_text(json.dumps({"v": 2}), encoding="utf-8")
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))   # make sure the mtime moves
    assert app_data._json_or_none(str(f), app_data._mtime(f)) == {"v": 2}


@pytest.mark.parametrize("mode,must,must_not", [
    ("no-llm", "local AI search", "generated"),
    ("extractive", "local QA model", "generated by"),
    ("gemini", "Google Gemini", "Claude"),
    ("llm", "Anthropic Claude", "Gemini"),
])
def test_ask_label_follows_mode(monkeypatch, mode, must, must_not):
    """#16: the provenance label depends on ask()'s mode; no-llm never claims generated text."""
    import ai.qa
    monkeypatch.setattr(ai.qa, "ask", lambda q, **kw: {
        "answer": "Answer [1].", "mode": mode,
        "citations": [{"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "documents/PSI-26/x.pdf", "page": 1}]})
    at = AppTest.from_file(str(APP_DIR / "views/ask.py"), default_timeout=TIMEOUT)
    at.query_params["q"] = "test question"
    at.run()
    assert not at.exception, at.exception
    answer_captions = " ".join(c.value for c in at.caption[:-1])   # last caption is the page footer
    assert must in answer_captions
    assert must_not not in answer_captions


def test_rule_based_profiles_not_labelled_ai():
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    caps = " ".join(c.value for c in at.caption)
    assert "AI-generated" not in caps and "Written by AI" not in caps
    assert "Summary from PSI metadata" in caps


def test_predict_page_predicts_and_refuses():
    """#25: in-range inputs give probabilities; Moon-base pressure (56.5 kPa) is refused, not extrapolated."""
    at = AppTest.from_file(str(APP_DIR / "views/predict.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("Most likely" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 2            # probability bars + boundary map
    at.number_input[0].set_value(56.5).run()
    assert not at.exception, at.exception
    assert any("outside the FLEX range" in w.value for w in at.warning)
    at.radio[0].set_value("Methanol").run()
    at.radio[2].set_value("HE").run()                  # helium boundary grid
    assert not at.exception, at.exception


SIM = {"method": "test", "points": [{"id": "PSI-26", "acronym": "BASS", "x": 0, "y": 0, "cluster": 0},
                                    {"id": "PSI-25", "acronym": "BASS-II", "x": 1, "y": 0, "cluster": 0},
                                    {"id": "PSI-69", "acronym": "FLEX", "x": 5, "y": 5, "cluster": 1}],
       "clusters": [{"id": 0, "label": "solids", "members": ["PSI-26", "PSI-25"]},
                    {"id": 1, "label": "droplets", "members": ["PSI-69"]}],
       "neighbors": {"PSI-26": [{"id": "PSI-25", "score": 0.93, "shared": ["solid", "MSG flow duct"]}]}}


def test_similarity_map_and_neighbours(tmp_path, monkeypatch):
    import json

    from app import data as app_data
    (tmp_path / "similarity.json").write_text(json.dumps(SIM), encoding="utf-8")
    monkeypatch.setattr(app_data, "ANALYSIS", tmp_path)
    at = AppTest.from_file(str(APP_DIR / "views/compare.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 1
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    assert not at.exception, at.exception
    assert any("Most similar" in s.value for s in at.subheader)


def test_similarity_not_available_yet(tmp_path, monkeypatch):
    from app import data as app_data
    monkeypatch.setattr(app_data, "ANALYSIS", tmp_path)          # empty dir: no similarity.json
    at = AppTest.from_file(str(APP_DIR / "views/compare.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("not available yet" in i.value for i in at.info)


def test_ask_shows_answer_span(monkeypatch):
    import ai.qa
    snippet = "Saffire-I burned a large sample of SIBAL cotton-fiberglass fabric in a Cygnus vehicle."
    monkeypatch.setattr(ai.qa, "ask", lambda q, **kw: {
        "answer": "SIBAL fabric [1]", "mode": "extractive",
        "answer_span": {"text": "SIBAL cotton-fiberglass fabric", "score": 0.81, "passage_index": 0, "start": 0, "end": 0},
        "passages": [{"n": 1, "snippet": snippet, "psi_id": "PSI-98", "file": "documents/PSI-98/a.pdf", "page": 3}],
        "citations": [{"psi_id": "PSI-98", "doi": "10.60555/0t15-1z43", "file": "documents/PSI-98/a.pdf", "page": 3}]})
    at = AppTest.from_file(str(APP_DIR / "views/ask.py"), default_timeout=TIMEOUT)
    at.query_params["q"] = "what burned in saffire"
    at.run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "<mark>SIBAL cotton-fiberglass fabric</mark>" in md


def test_polish_labels_and_size():
    """#18: readable variable names; raw-data size computed from investigations.json."""
    from app import data as app_data
    assert app_data.pretty_vars(["o2", "sample_thickness", "flow_direction"]) == "O₂, sample thickness, flow direction"
    assert app_data.pretty_vars(None) == "—"
    at = AppTest.from_file(str(APP_DIR / "views/about.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert "~13.5 TB" in at.markdown[0].value
    at = AppTest.from_file(str(APP_DIR / "views/compare.py"), default_timeout=TIMEOUT).run()
    assert "sample_thickness" not in at.dataframe[0].value.to_string()
