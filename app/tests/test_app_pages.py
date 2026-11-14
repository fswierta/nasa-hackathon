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
    assert any("IGNIS" in t.value for t in at.title)
    assert any("Flame in Freefall" in c.value for c in at.caption)
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
    ("gemini", "Google's free API", "Claude"),
    ("anthropic", "Anthropic's API", "Gemini"),
    ("llm", "Anthropic's API", "Gemini"),
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


def test_rule_based_profiles_not_labelled_ai(monkeypatch, tmp_path):
    """#45/#44: with no LLM profile the page falls back to processed/ai/profiles_rules and says so."""
    import ai.profiles
    monkeypatch.setattr(ai.profiles, "PROFILES_DIR", tmp_path)      # empty: no Gemini profile
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    assert not at.exception, at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "AI-generated" not in caps and "Written by AI" not in caps
    assert "Summary from PSI metadata" in caps


def test_llm_profile_labelled_ai_with_model():
    """Mirror of the test above: a committed Gemini profile is labelled as AI with its model name."""
    import json
    from ai.profiles import PROFILES_DIR
    f = PROFILES_DIR / "PSI-26.json"
    if not f.exists():
        pytest.skip("no LLM profile for PSI-26")
    gen = json.loads(f.read_text(encoding="utf-8")).get("generated") or {}
    if "rule" in (gen.get("by") or "").lower():
        pytest.skip("PSI-26 profile is rule-based")
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    caps = " ".join(c.value for c in at.caption)
    assert "Written by AI" in caps and (gen.get("model") or gen.get("by")) in caps


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



def test_generated_label_names_model_and_cache():
    """#28: generated answers name the model; cached ones say no new request was sent."""
    from app import data as app_data
    icon, text = app_data.answer_label({"mode": "gemini", "model": "gemini-3.5-flash-lite"})
    assert "`gemini-3.5-flash-lite`" in text and "sent to Google's free API" in text and "improve its products" in text
    _, cached = app_data.answer_label({"mode": "gemini", "model": "gemini-3.8-flash", "cached": True})
    assert "earlier, not now" in cached and "Your question is sent" not in cached
    _, offline = app_data.answer_label({"mode": "no-llm"})
    assert "Generated" not in offline and "Google" not in offline
    assert app_data.answer_label({"mode": "something-new"}) is None


def test_secrets_include_gemini(monkeypatch):
    """#28: GEMINI_API_KEY and FIF_LLM from Streamlit secrets reach the environment."""
    import os
    for k in ("GEMINI_API_KEY", "FIF_LLM"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT)
    at.secrets["GEMINI_API_KEY"] = "test-not-real"
    at.secrets["FIF_LLM"] = "none"          # keep the test offline even if ai/ would use the key
    at.run()
    assert not at.exception, at.exception
    assert os.environ.get("GEMINI_API_KEY") == "test-not-real" and os.environ.get("FIF_LLM") == "none"
    for k in ("GEMINI_API_KEY", "FIF_LLM"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------- review fixes (gravity, O2 role, myths, spinoffs)
def _all_text(at) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning") for e in getattr(at, kind)]
    return " ".join(str(p) for p in parts)


def test_no_page_says_0g_and_1g_bound_partial_gravity():
    """NASA (Ferkul et al., LPSC 2026): lunar gravity may be the MOST flammable case, so the app must
    never claim microgravity and 1 g results bound it — even while older processed files still do."""
    import re
    for path in ("views/about.py", "views/condition_map.py", "views/insights.py", "views/ranker.py"):
        at = AppTest.from_file(str(APP_DIR / path), default_timeout=TIMEOUT).run()
        assert not at.exception, (path, at.exception)
        if path == "views/ranker.py":
            at.radio[0].set_value("lunar").run()
        text = _all_text(at)
        assert not re.search(r"results (only )?bound", text, re.I), path
    assert "most flammable" in text and "20260001966" in text      # ranker shows the source


def test_partial_g_fix_rewrites_old_sentences():
    from app import data
    for old in ("All 19 flight investigations ran in microgravity. Microgravity and 1 g results bound partial-g behaviour.",
                "PSI has no partial-gravity combustion data; microgravity and 1 g results only bound this case."):
        new = data.partial_g_fix(old)
        assert "bound partial" not in new and "only bound" not in new and "not** safely bound" in new


def test_ranker_gravity_choice():
    at = AppTest.from_file(str(APP_DIR / "views/ranker.py"), default_timeout=TIMEOUT).run()
    assert at.radio[0].label == "Gravity" and at.radio[0].value == "lunar"       # Lunar base preset
    at.selectbox[0].select("ISS").run()
    assert at.radio[0].value == "micro"
    for g in ("micro", "lunar", "mars", "earth"):
        at.radio[0].set_value(g).run()
        assert not at.exception, (g, at.exception)


def test_inverse_flame_o2_is_not_atmosphere():
    from app import data
    t = data.tests()
    role = data.o2_role(t)
    burner = t[role == "burner"]
    assert set(burner["investigation_id"]) == {"PSI-10", "PSI-159"}
    assert (t.loc[role == "ambient", "o2_frac"] <= 0.41).all()          # no 85 % "atmosphere" left
    import base64
    import json

    import numpy as np

    def values(v):   # plotly 6 may ship arrays as {"dtype", "bdata"}
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]) if isinstance(v, dict) else v

    at = AppTest.from_file(str(APP_DIR / "views/condition_map.py"), default_timeout=TIMEOUT).run()
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    test_traces = [tr for tr in spec["data"] if tr.get("name") in ("solid", "droplet", "gas")]
    assert test_traces and max(max(values(tr["y"])) for tr in test_traces) <= 41
    at.toggle(key="burner").set_value(True).run()
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert any("Burner O₂" in (tr.get("name") or "") for tr in spec["data"])


def test_insights_no_fan_myth_and_spinoffs():
    at = AppTest.from_file(str(APP_DIR / "views/insights.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    text = _all_text(at)
    assert "Shutting off fans is a real" not in text
    assert "self-extinguish if ventilation is shut off" in text            # quoted as a myth, p. 35
    assert "****" not in " ".join(m.value for m in at.markdown)
    for name in ("water-mist", "Smoke detectors", "air tanks", "protective clothing"):
        assert name in text, name
    raw = (ROOT / "processed/ai/insights.json").read_text(encoding="utf-8")
    assert "Shutting off fans is a real fire-control lever" not in raw


# ---------------------------------------------------------------- #34 data explorer + investigators' words
def _rows_caption(at) -> str:
    return next(c.value for c in at.caption if "rows** match" in c.value)


def test_explorer_filters_download_and_flags():
    at = AppTest.from_file(str(APP_DIR / "views/explorer.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert "1,947 of 1,947" in _rows_caption(at)
    assert at.get("download_button") and "1,947 rows" in at.get("download_button")[0].proto.label

    at.multiselect[0].select("PSI-26").run()                     # BASS
    assert "121 of 1,947" in _rows_caption(at)
    at.checkbox(key="f_o2_none").uncheck().run()                  # BASS has no O2 column
    assert "**0 of" in _rows_caption(at)
    at.checkbox(key="f_o2_none").check().run()

    at.toggle(key="only_flagged").set_value(True).run()
    n_flag = int(_rows_caption(at).split("**")[1].split(" of")[0].replace(",", ""))
    assert 0 < n_flag < 121
    text = " ".join(m.value for m in at.markdown)
    import re
    assert re.search(r"⚠ \*\*[FD]\d+\*\* \(.+\): \S", text)       # inspector shows the flag id + its text
    assert any(df.value.columns.tolist() == ["PSI column", "value"] for df in at.dataframe)   # original row


def test_explorer_raw_table_and_file_index():
    at = AppTest.from_file(str(APP_DIR / "views/explorer.py"), default_timeout=TIMEOUT).run()
    at.selectbox(key="raw_inv").select("PSI-10").run()           # two tables -> radio
    assert not at.exception, at.exception
    assert at.radio and len(at.radio[0].options) == 2
    at.selectbox(key="files_inv").select("PSI-26").run()
    assert not at.exception, at.exception
    assert any("files ·" in c.value and "links only" in c.value for c in at.caption)


def test_detail_investigators_words_and_documents():
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    assert not at.exception, at.exception
    assert "In the investigators' words" in [h.value for h in at.subheader]
    cap = " ".join(c.value for c in at.caption)
    assert "Written by the investigation team: Sandra L. Olson (NASA GRC)" in cap and "unedited" in cap
    import json
    inv = next(i for i in json.loads((ROOT / "processed/investigations.json").read_text(encoding="utf-8"))
               if i["id"] == "PSI-26")
    assert any(inv["objective"][:80] in m.value for m in at.markdown)          # verbatim
    assert any(s.value.startswith("Science documents (") for s in at.subheader)
    assert at.get("download_button")


# ---------------------------------------------------------------- #42 demo tour
@pytest.mark.parametrize("step,title,check", [
    ("1", "Scenario ranker", lambda at: at.selectbox(key="preset").value == "Lunar base"),
    ("2", "Condition map", lambda at: at.toggle(key="gap_hl").value and any("The gap" in e.value for e in at.error)),
])
def test_home_tour_steps(step, title, check):
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert len([b for b in at.button if b.key and b.key.startswith("tour_")]) == 3
    at.button(key=f"tour_{step}").click().run()
    assert not at.exception, at.exception
    assert any(title in t.value for t in at.title)
    assert check(at)


def test_tour_ask_step_queues_the_demo_question(monkeypatch):
    import ai.qa
    seen = []
    monkeypatch.setattr(ai.qa, "ask", lambda q, **kw: seen.append(q) or {"answer": "stub", "citations": []})
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()
    at.button(key="tour_3").click().run()
    assert not at.exception, at.exception
    assert seen and "Moon-base" in seen[0]


def test_url_deep_links_for_ranker_and_map():
    at = AppTest.from_file(str(APP_DIR / "views/ranker.py"), default_timeout=TIMEOUT)
    at.query_params["habitat"] = "Mars base"
    at.run()
    assert at.selectbox(key="preset").value == "Mars base"
    at = AppTest.from_file(str(APP_DIR / "views/condition_map.py"), default_timeout=TIMEOUT)
    at.query_params["gap"] = "solid"
    at.run()
    assert at.toggle(key="gap_hl").value and any("Lunar base" in e.value for e in at.error)


def test_detail_documents_table_one_dtype():
    """#42: 'Pages searchable in Ask' mixed int and '—' -> Arrow warning; now always text."""
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    df = next(d.value for d in at.dataframe if "Pages searchable in Ask" in d.value.columns)
    assert df["Pages searchable in Ask"].map(type).eq(str).all()


# ---------------------------------------------------------------- #48 UI kit, sections, theme
def test_pages_use_the_ui_kit():
    """Titles go through ui.page_header so a redesign changes one file (pc2's ML Lab is its own file)."""
    for p in (APP_DIR / "views").glob("*.py"):
        if p.name == "ml_lab.py":
            continue
        src = p.read_text(encoding="utf-8")
        assert "st.title(" not in src, p.name
        assert "ui.page_header(" in src, p.name


def test_provenance_badges():
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    caps = [c.value for c in at.caption]
    assert any(c.startswith(":blue-badge[PSI verbatim]") and "investigation team" in c for c in caps)
    assert any(c.startswith((":violet-badge[AI-written]", ":gray-badge[Rule-based]")) for c in caps)


def test_ui_helpers():
    from app import ui
    assert ui.source_link("10.60555/58h4-dy80") == "[10.60555/58h4-dy80](https://doi.org/10.60555/58h4-dy80)"
    assert ui.source_link(None) == "" and ui.source_link("x", "DOI x") == "[DOI x](https://doi.org/x)"
    assert set(ui.PROVENANCE) == {"psi", "computed", "rules", "ai", "ml"}


def test_nav_sections_cover_every_page():
    from app.nav import SECTIONS, grouped
    g = grouped()
    assert list(g)[:1] == [""] and "More" not in g          # every page has a section
    assert [s for s in SECTIONS if s] == ["Explore", "Decide", "AI & ML", "Learn"]
    assert sum(len(v) for v in g.values()) == len(available())


def test_theme_has_light_and_dark():
    import tomllib
    cfg = tomllib.loads((APP_DIR.parent / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["theme"]["dark"]["backgroundColor"] == "#0B1426"
    assert cfg["theme"]["dark"]["primaryColor"] == "#F26419"
    assert cfg["theme"]["light"]["backgroundColor"] == "#FFFFFF" and "base" not in cfg["theme"]["light"]


# ---------------------------------------------------------------- #49 Fire scenario + Response advisor
def _fire_page(page, **widgets):
    """Run a #49 page, then set widgets one by one (key -> value), re-running after each."""
    at = AppTest.from_file(str(APP_DIR / "views" / f"{page}.py"), default_timeout=300).run()
    for key, value in widgets.items():
        kind = "slider" if key.startswith(("fs_o2", "fs_p_")) else "selectbox" if "fuel" in key or key == "fs_preset" \
            else "radio"
        getattr(at, kind)(key=key).set_value(value).run()
    assert not at.exception, at.exception
    return at


def _page_text(at) -> str:
    return " ".join(e.value for kind in ("markdown", "caption", "info", "warning", "error") for e in getattr(at, kind))


@pytest.mark.parametrize("page", ["scenario", "response"])
def test_fire_pages_show_the_disclaimer(page):
    from analysis.scenario import DISCLAIMER
    at = _fire_page(page)
    assert any(DISCLAIMER in w.value for w in at.warning)            # banner at the top
    assert any(DISCLAIMER in w.value for w in at.sidebar.warning)    # and in the sidebar, visible while scrolling


def test_scenario_droplet_in_iss_air_has_a_prediction():
    at = _fire_page("scenario")                                      # defaults: heptane droplet, ISS, microgravity
    labels = {m.label: m.value for m in at.metric}
    assert labels["Goes out on its own"].endswith("%")
    assert labels["Model accuracy"].endswith("%") and labels["Baseline"].endswith("%")   # accuracy next to baseline
    assert len(at.get("plotly_chart")) == 1


def test_scenario_moon_shows_the_statement_and_no_number():
    at = _fire_page("scenario", fs_g_ISS="lunar")
    assert any("No partial-gravity combustion data exists in PSI" in i.value for i in at.info)
    assert not at.metric and not at.get("plotly_chart")              # no percentage anywhere
    at = _fire_page("scenario", fs_class="gas", fs_g_ISS="mars")
    assert not at.metric and any("Mars" in i.value for i in at.info)


def test_scenario_solid_says_no_prediction_and_shows_saffire_rows():
    at = _fire_page("scenario", fs_class="solid", fs_fuel_solid="silicone")
    assert any("No prediction" in i.value and "no outcome data for solid materials" in i.value for i in at.info)
    saffire = next(d.value for d in at.dataframe if "Burn length, 1 g" in d.value.columns)
    assert len(saffire) == 4 and set(saffire["Material"]) == {"silicone"}
    assert "Complete" in set(saffire["Burn length, 1 g"])            # burned at 1 g, '~ 0' in microgravity
    assert set(saffire["Burn length, microgravity"]) == {"~ 0"}
    assert not any(str(m.value).endswith("%") for m in at.metric)


def test_scenario_out_of_range_names_the_input():
    at = _fire_page("scenario", fs_o2_ISS=60.0)
    msg = next(i.value for i in at.info if "No prediction" in i.value)
    assert "o2_frac=0.6" in msg and "0.13" in msg and "0.34" in msg  # which input + the tested range
    ranges = next(d.value for d in at.dataframe if "Tested range" in d.value.columns)
    assert "⚠ outside" in set(ranges[""])


def test_scenario_earth_shows_the_saffire_comparison():
    at = _fire_page("scenario", fs_g_ISS="earth")
    table = next(d.value for d in at.dataframe if "Burn length, 1 g" in d.value.columns)
    assert len(table) == 9 and {"silicone", "SIBAL"} <= set(table["Material"])
    assert any("microgravity only" in w.value for w in at.warning)


def test_response_advisor_sections():
    at = _fire_page("response")
    assert len(at.get("plotly_chart")) == 3                          # outcome bars + CO2 scan + He scan
    table = next(d.value for d in at.dataframe if "O₂ in the atmosphere" in d.value.columns)
    assert len(table) == 4 and table["CO₂"].str.contains(r"\(n=\d+\)").all()
    text = _page_text(at)
    assert "~~A fire will self-extinguish if ventilation is shut off.~~" in text      # verbatim, struck through
    assert ":blue-badge[Finding] Since many fire detectors" in text                   # the finding is not struck
    assert "FLEX set the suppressant fraction before ignition" in text                # caveats shown
    at = _fire_page("response", fs_class="solid")
    assert len(at.get("plotly_chart")) == 0 and any("no estimate" in i.value for i in at.info)


def test_response_advisor_related_evidence_without_a_key(monkeypatch):
    import ai.qa
    asked = []
    monkeypatch.setattr(ai.qa, "ask", lambda q, **kw: asked.append(q) or {
        "mode": "no-llm", "answer": "stub passage [1]",
        "citations": [{"psi_id": "PSI-69", "doi": "10.60555/mbq8-0451", "file": "documents/PSI-69/x.pdf", "page": 3}]})
    at = _fire_page("response")
    at.button(key="fire_search").click().run()
    assert not at.exception, at.exception
    assert asked == ["How was a heptane droplet fire suppressed or extinguished in microgravity?"]
    assert any("FLEX" in c.value and "p. 3" in c.value for c in at.caption)


# ---------------------------------------------------------------- #55 redesign: layout A + habitat picker, checklist
def test_home_habitat_picker_drives_gap_map_and_top5():
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()   # Home needs the navigation
    assert not at.exception, at.exception
    assert [m.label for m in at.metric] == ["Investigations", "Test records", "Documents indexed", "Data gaps found"]
    assert any("Data gap for Lunar base" in e.value for e in at.error)            # the gap is red everywhere
    assert any("Top evidence for Lunar base" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 1
    at.button_group[0].set_value("ISS").run()
    assert not at.exception, at.exception
    assert any("Top evidence for ISS" in m.value for m in at.markdown)
    assert not any("Data gap for" in e.value for e in at.error)                   # ISS has no solid-material gap
    assert any(i.value.startswith("**ISS:**") and "solid-material tests match" in i.value for i in at.info)
    at.button_group[0].set_value("Spacesuit").run()                               # 100 % O2: off the chart, no crash
    assert not at.exception, at.exception


def test_ranker_layout_a():
    at = AppTest.from_file(str(APP_DIR / "views/ranker.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert labels[0] == "Best match" and "…with solid materials" in labels
    table = next(d.value for d in at.dataframe if "Why" in d.value.columns)
    assert len(table) == 10 and list(table["#"]) == list(range(1, 11))
    assert table["Score"].is_monotonic_decreasing
    assert len(at.number_input) == 0                                              # no locale-formatted '70,30'
    at.button_group[0].set_value(5).run()
    assert len(next(d.value for d in at.dataframe if "Why" in d.value.columns)) == 5


def test_table_style_dash_and_four_significant_figures():
    import numpy as np
    import pandas as pd
    from app import ui
    df = pd.DataFrame({"a": [0.123456, np.nan], "b": [1.0, 2.5], "n": [3, 4], "t": ["x", None],
                       "big": [12345.678, np.nan], "url": [None, "https://doi.org/x"]})
    out = ui.tidy(df, raw=("url",))
    assert list(out["a"]) == ["0.1235", "—"] and list(out["t"]) == ["x", "—"]      # missing -> em dash
    assert list(out["b"]) == [1.0, 2.5] and list(out["n"]) == [3, 4]              # complete numbers stay numeric
    assert out["big"][0] == "12,346" and pd.isna(out["url"][0])                   # raw columns untouched
    assert out["url"][1] == "https://doi.org/x"
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-99"
    at.run()
    tests_table = next(d.value for d in at.dataframe if "test_id" in d.value.columns)
    assert not tests_table.isna().any().any()                                     # no 'None' cells on screen


def test_no_emoji_in_navigation_and_no_orange_data_colour():
    import re
    from app import ui
    from app.nav import PAGE_SPECS
    assert all(icon.startswith(":material/") for _, _, icon, _ in PAGE_SPECS)
    # pc2 owns app/views/ml_lab.py (#47): it is told about the kit in the issue, not edited or tested here.
    assert "#eb6834" not in " ".join(p.read_text(encoding="utf-8") for p in APP_DIR.rglob("*.py")
                                     if "tests" not in p.parts and p.name != "ml_lab.py")   # orange = accent only
    assert ui.PHASE_COLOURS["droplet"] != ui.PALETTE["flame"]
    for p in (APP_DIR / "views").glob("*.py"):
        if p.name == "ml_lab.py":
            continue
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"st\.(subheader|dataframe)\(", src), p.name       # sections and tables use the kit
        assert "container(border=True)" not in src, p.name


def test_chart_theme_has_both_modes(monkeypatch):
    from app import ui
    dark = ui.chart_theme()
    monkeypatch.setattr(ui, "is_dark", lambda: False)
    light = ui.chart_theme()
    assert dark["label_fg"] != light["label_fg"] and set(dark) == set(light)


def test_detail_guide_citations_do_not_show_a_psi_doi(monkeypatch):
    """A finding quoted from NASA's Researcher's Guide (psi_id REF) links to the guide, not to a PSI DOI."""
    from app import data
    prof = {"id": "PSI-26", "summary_plain": "x", "summary_expert": "y", "generated": {"by": "Google Gemini"},
            "key_findings": [{"text": "A guide finding.", "source": {"psi_id": "REF", "doi": "10.60555/58h4-dy80",
                              "file": "reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf", "page": 31}}]}
    monkeypatch.setattr(data, "profile", lambda pid: prof)
    at = AppTest.from_file(str(APP_DIR / "views/detail.py"), default_timeout=TIMEOUT)
    at.query_params["id"] = "PSI-26"
    at.run()
    line = next(m.value for m in at.markdown if "A guide finding." in m.value)
    assert "NASA Researcher's Guide" in line and "nasa.gov" in line and "doi.org" not in line
