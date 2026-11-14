"""Issue #37: misconception guard on generated text, gravity in rank(), partial-g wording.
Run: python -m pytest ai/tests/test_guards_gravity.py -q"""
import json

import pytest

from ai.guards import myth_in, remove_myths


@pytest.mark.parametrize("sentence", [
    "Shutting off fans is a real fire-control lever.",                      # the old LLM insight
    "Turning off the cabin ventilation will put the fire out.",
    "A fire will self-extinguish if ventilation is shut off.",
    "A steady flame cannot exist in a quiescent microgravity environment.",
    "Because microgravity is inherently safer, terrestrial fire standards are very conservative.",
])
def test_guard_catches_the_researchers_guide_misconceptions(sentence):
    assert myth_in(sentence)


@pytest.mark.parametrize("sentence", [
    # framed as a misconception (how the Guide itself says it)
    "NASA lists the belief that a fire self-extinguishes when ventilation is shut off as a misconception.",
    # a real BASS finding that says the opposite — must never be dropped
    "Local application of a suppressant is not adequate for flame extinguishment because the local jet entrains "
    "air and sustains the flame even when the ambient airflow is turned off.",
    "Extinction was observed at a low but finite flow speed of less than 1 cm/s.",
    "The flame was extinguished by nitrogen from the suppressant jet.",
])
def test_guard_keeps_real_findings(sentence):
    assert myth_in(sentence) is None


def test_remove_myths_keeps_other_sentences():
    text = "BASS burned fabric in a flow duct. Shutting off fans is a real fire-control lever. Spread rose with flow."
    clean, removed = remove_myths(text)
    assert clean == "BASS burned fabric in a flow duct. Spread rose with flow."
    assert removed[0]["myth"] == "ventilation-off extinguishes" and removed[0]["source"]["page"] == 35


def test_misconception_source_is_in_the_knowledge_base():
    """The guard cites Researcher's Guide p. 35 — that passage must exist in our extracted text."""
    from ai.config import CHUNKS_FILE
    found = any("self-extinguish if ventilation is shut off" in " ".join(json.loads(l)["text"].split())
                for l in open(CHUNKS_FILE, encoding="utf-8"))
    assert found


def test_shipped_generated_text_has_no_misconception():
    import glob

    from ai.config import AI_OUT
    from ai.textutil import sentences
    texts = []
    for f in glob.glob(str(AI_OUT / "profiles" / "PSI-*.json")):
        p = json.loads(open(f, encoding="utf-8").read())
        texts += [p["summary_plain"], p["summary_expert"], *[x["text"] for x in p["key_findings"]]]
    texts += [v["answer"] for v in json.loads((AI_OUT / "qa_cache.json").read_text(encoding="utf-8")).values()]
    bad = [s for t in texts for s in sentences(t or "") if myth_in(s)]
    assert not bad, bad


def test_ask_drops_a_myth_sentence_and_its_citation(monkeypatch):
    from ai import llm, qa
    monkeypatch.setattr(qa, "QA_CACHE", qa.QA_CACHE.with_name("qa_cache_test_tmp.json"))
    monkeypatch.setattr(qa, "_CACHE", {})
    monkeypatch.setattr(qa, "_save_cache", lambda: None)
    hits = [{"psi_id": "PSI-26", "doi": "d1", "file": "a.pdf", "page": 1, "text": "x"},
            {"psi_id": "PSI-25", "doi": "d2", "file": "b.pdf", "page": 2, "text": "y"}]
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)
    monkeypatch.setattr(qa, "provider", lambda: "gemini")
    monkeypatch.setattr(qa, "json_call", lambda *a, **k: {
        "answer": "Flames spread faster with flow [1]. Shutting off the fans will put the fire out [2].",
        "used_excerpts": [1, 2], "_model": "m"})
    out = qa.ask("does ventilation matter?", use_cache=False)
    assert out["answer"] == "Flames spread faster with flow [1]."
    assert [c["psi_id"] for c in out["citations"]] == ["PSI-26"] and out["guard_removed"]


# ------------------------------------------------------------------ gravity in rank()
def test_gravity_component():
    from ai.habitats import HABITATS
    from ai.ranking import DEFAULT_WEIGHTS, rank
    assert "gravity" in DEFAULT_WEIGHTS
    # #54: 1.0 only with test data AT that gravity, else None ("no data") — never partial credit
    iss = {r["id"]: r["breakdown"]["gravity"] for r in rank(HABITATS["ISS"])}
    assert iss["PSI-69"] == 1.0 and iss["PSI-62"] is None          # flight tests vs a ground model study
    lunar = rank(HABITATS["Lunar base"])
    assert all(r["breakdown"]["gravity"] is None for r in lunar)    # no partial-g data anywhere in PSI
    earth = {r["id"]: r["breakdown"]["gravity"] for r in rank(dict(HABITATS["ISS"], gravity="earth"))}
    from analysis.scenario import gravity_evidence
    assert {k for k, v in earth.items() if v} == {g["source"]["psi_id"] for g in gravity_evidence()} == {"PSI-99"}


def test_partial_gravity_gap_is_the_engines_statement_and_e_field_is_a_proxy_note():
    from analysis.scenario import GRAVITY
    from ai.habitats import HABITATS
    from ai.ranking import rank
    res = {r["id"]: r for r in rank(HABITATS["Mars base"])}
    assert GRAVITY["mars"]["statement"] in res["PSI-69"]["gaps"]
    assert any("proxy" in g for g in res["PSI-22"]["gaps"]) and res["PSI-22"]["breakdown"]["gravity"] is None


def test_partial_gravity_wording_never_claims_bounds():
    """#36/#37: 0 g / 1 g results must not be said to bound partial gravity."""
    from ai.habitats import HABITATS
    from ai.ranking import rank
    gaps = " ".join(g for r in rank(HABITATS["Lunar base"]) for g in r["gaps"]).lower()
    assert "no partial-gravity" in gaps and "may not bound" in gaps and "only bound" not in gaps
    from ai import profiles
    assert "bound partial-gravity" in profiles.SYSTEM and "Never state or imply" in profiles.SYSTEM
