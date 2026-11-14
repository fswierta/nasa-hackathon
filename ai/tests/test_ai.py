"""Run from repo root: python -m pytest ai/tests -q   (no API key or index needed)."""
from ai import qa
from ai.catalog import load_investigations
from ai.habitats import HABITATS
from ai.ranking import DEFAULT_WEIGHTS, _range_score, rank


def test_catalog_has_24_with_doi():
    invs = load_investigations()
    assert len(invs) == 24
    assert all(i["doi"] and i["doi"].startswith("10.60555/") for i in invs)


def test_range_score():
    assert _range_score(0.21, [0.12, 0.34], 0.1) == 1.0
    assert _range_score(0.50, [0.12, 0.34], 0.1) == 0.0
    assert 0 < _range_score(0.39, [0.12, 0.34], 0.1) < 1
    assert _range_score(0.21, None, 0.1) is None
    # unknown lower bound: only the stated value counts as covered (no guessing)
    assert _range_score(0.85, [None, 0.85], 0.1) == 1.0
    assert _range_score(0.50, [None, 0.85], 0.1) == 0.0


def test_rank_contract_shape():
    res = rank(HABITATS["ISS"])
    assert len(res) == 24
    assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)
    for r in res:
        assert {"id", "score", "breakdown", "why"} <= r.keys()
        assert 0.0 <= r["score"] <= 1.0
        assert set(r["breakdown"]) == set(DEFAULT_WEIGHTS)


def test_pillar_focus():
    top = [r["acronym"] for r in rank(dict(HABITATS["ISS"], pillar="detection"))[:2]]
    assert set(top) <= {"SAME", "SAME-R", "DAFT / DAFT-2"}


def test_partial_gravity_gap_reported():
    assert any("partial-gravity" in g for g in rank(HABITATS["Lunar base"])[0]["gaps"])


def test_weights_override():
    only_tier = {k: 0.0 for k in DEFAULT_WEIGHTS} | {"tier": 1.0}
    res = rank(HABITATS["ISS"], only_tier)
    assert all(r["score"] == 1.0 for r in res[:5])


import pytest


@pytest.fixture(autouse=True)
def _isolated_qa_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(qa, "QA_CACHE", tmp_path / "qa_cache.json")
    monkeypatch.setattr(qa, "_CACHE", None)


def test_ask_caches_llm_answers(monkeypatch):
    hits = [{"psi_id": "PSI-26", "doi": "d", "file": "a.pdf", "page": 1, "text": "x"}]
    calls = []
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)
    monkeypatch.setattr(qa, "json_call", lambda *a, **k: calls.append(1) or {"answer": "A [1].", "used_excerpts": [1]})
    first = qa.ask("Same  question?")
    second = qa.ask("same question?")
    assert len(calls) == 1 and second["cached"] and second["answer"] == first["answer"]
    assert qa.QA_CACHE.exists()


def test_rank_is_fast():
    import time
    rank(HABITATS["ISS"])  # warm caches
    t0 = time.perf_counter()
    for _ in range(20):
        rank(HABITATS["Lunar base"])
    assert (time.perf_counter() - t0) / 20 < 0.05


def test_flex_model_quality_and_refusal():
    from ai.model import model_card, predict_flex
    card = model_card()
    assert card["k"]["n_train"] > 200 and card["k"]["cv_r2"] > 0.5
    p = predict_flex(0.21, 101.3, "N2", 3.0, "heptane")
    assert p["k_low"] < p["k"] < p["k_high"] and p["reason"] is None
    # Exploration atmosphere pressure (56.5 kPa) is below anything FLEX tested -> refuse
    assert predict_flex(0.34, 56.5, "N2", 3.0, "heptane")["k"] is None
    assert predict_flex(0.21, 101.3, "N2", 3.0, "decane")["k"] is None
    assert predict_flex(0.21, 101.3, "N2+CO2", 3.0, "heptane")["k"] is None  # missing co2_frac


@pytest.mark.parametrize("with_key", [False, True])
def test_ask_without_anthropic_package(monkeypatch, with_key):
    """#11: no SDK installed (with or without a key) -> no-llm fallback, never ModuleNotFoundError."""
    import sys
    from ai import llm

    hits = [{"psi_id": "PSI-98", "doi": "d", "file": "a.docx", "page": None, "text": "SIBAL fabric"}]
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError
    monkeypatch.setattr(llm, "has_api_key", lambda: with_key)
    llm._client.cache_clear()
    out = qa.ask("Which materials were tested in Saffire?")
    llm._client.cache_clear()
    assert out["mode"] == "no-llm" and out["citations"][0]["psi_id"] == "PSI-98"


def test_ask_drops_invented_citations(monkeypatch):
    hits = [{"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "a.pdf", "page": 3, "text": "x"},
            {"psi_id": "PSI-25", "doi": "10.60555/4qc4-de67", "file": "b.pdf", "page": 1, "text": "y"}]
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)
    monkeypatch.setattr(qa, "json_call", lambda *a, **k: {"answer": "Fact [2]. Other [9].", "used_excerpts": [2, 9]})
    out = qa.ask("q")
    assert out["answer"] == "Fact [1]. Other ."
    assert out["citations"] == [{"psi_id": "PSI-25", "doi": "10.60555/4qc4-de67", "file": "b.pdf", "page": 1}]


def test_ask_without_llm_returns_excerpts(monkeypatch):
    hits = [{"psi_id": "PSI-26", "doi": "d", "file": "a.pdf", "page": 3, "text": "flame spread"}]
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)

    def boom(*a, **k):
        raise qa.LLMUnavailable("no key")

    monkeypatch.setattr(qa, "json_call", boom)
    out = qa.ask("q")
    assert out["mode"] == "no-llm" and out["citations"][0]["psi_id"] == "PSI-26"


@pytest.mark.parametrize("text,expected", [
    ("ACME BRE (PSI-20) ran at 21-40 %", ["PSI-20"]),
    ("BASS nitrogen", ["PSI-26"]),
    ("What did SAFFIRE-II show?", ["PSI-99"]),
    ("Which materials were tested in Saffire?", ["PSI-98", "PSI-99", "PSI-100"]),
    ("FLEX vs FLEX-2", ["PSI-68", "PSI-69"]),
    ("SAME smoke", ["PSI-102"]),
    ("SAME-R smoke", ["PSI-101"]),
    ("cool flames", []),
])
def test_mentioned_investigations(text, expected):
    from ai.retrieval import mentioned_investigations
    assert sorted(mentioned_investigations(text)) == sorted(expected)


def test_insights_build_verified():
    """#14 (no LLM): every insight has >=1 verbatim backing quote (re-verified against chunks.jsonl),
    no quote contains our synthetic metadata header, and templates were filled from computed numbers."""
    from ai.insights import build
    out = build()  # raises ValueError if any quote is not in its cited file/page, or an insight has none
    ins = out["insights"]
    assert out["generated"]["by"] == "rules (no LLM)"
    for i in ins:
        quotes = [c["quote"] for c in i["citations"] if "quote" in c]
        assert 1 <= len(quotes) <= 3, i["id"]
        assert not any(q.startswith("PSI-") or ": approach:" in q for q in quotes), i["id"]
        assert "{" not in i["title"] + i["text"], i["id"]  # every template placeholder filled
    for e in out["earth_applications"]:
        assert e["citations"][0]["quote"] in e["text"]  # earth applications are quotes, not paraphrase
    for pillar in ("prevention", "detection", "suppression"):
        assert len([i for i in ins if i["pillar"] == pillar and i["kind"] != "data_gap"]) >= 3
    # every citation has a DOI, except the Researcher's Guide (psi_id "REF"), which is not a PSI dataset
    assert all(i["citations"] and all(c["psi_id"] and (c["doi"] or c["psi_id"] == "REF") for c in i["citations"])
               for i in ins)
    gaps = " ".join(i["title"] for i in ins if i["kind"] == "data_gap")
    assert "solid" in gaps.lower() and "22.2" in gaps and "pressure" in gaps
    assert any(i["id"] == "gap-flex-pressure" for i in ins)
    assert out["earth_applications"] and all(e["citations"] for e in out["earth_applications"])


def test_insights_bad_quote_fails():
    from ai.insights import verify_quote
    with pytest.raises(ValueError):
        verify_quote({"file": "metadata/PSI-26.json#researchImpacts", "page": None, "quote": "made-up sentence not in PSI"})
    verify_quote({"file": "metadata/PSI-26.json#researchImpacts", "page": None,
                  "quote": "Using nitrogen as a flame suppressant in microgravity provides a direct link to current "
                           "and planned extinguishment techniques."})


def test_profile_drops_unsourced_items(monkeypatch):
    """#12: findings / measurements citing a non-existent excerpt are dropped; sources resolve to file+page."""
    from ai import profiles
    hits = [{"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "a.pdf", "page": 4, "text": "camera"}]
    monkeypatch.setattr(profiles, "gather", lambda pid: hits)
    fake = {"summary_plain": "p", "summary_expert": "e", "fire_safety_pillar": ["prevention"],
            "habitat_relevance": {k: 1.5 for k in profiles.HABITAT_KEYS}, "earth_applications": [],
            "key_findings": [{"text": "ok", "excerpt": 1}, {"text": "invented", "excerpt": 7}],
            "measurements": [{"name": "color video camera", "type": "camera", "measures": "flame", "excerpt": 1},
                             {"name": "ghost sensor", "type": "other", "measures": "?", "excerpt": 0}]}
    monkeypatch.setattr(profiles, "json_call", lambda *a, **k: fake)
    prof = profiles.generate("PSI-26")
    assert [f["text"] for f in prof["key_findings"]] == ["ok"]
    assert [m["name"] for m in prof["measurements"]] == ["color video camera"]
    assert prof["measurements"][0]["source"] == {"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "a.pdf", "page": 4}
    assert all(v == 1.0 for v in prof["habitat_relevance"].values())


def test_rule_profiles_are_verbatim_and_sourced():
    """#12 (no LLM): all 24 profiles exist; findings are verbatim from the cited chunk; summaries quote metadata."""
    import json
    from ai.config import CHUNKS_FILE, DATA_DIR, PROFILES_DIR
    from ai.textutil import norm

    by_src = {}
    for line in open(CHUNKS_FILE, encoding="utf-8"):
        c = json.loads(line)
        by_src.setdefault((c["file"], c["page"]), []).append(norm(c["text"]))
    files = sorted(PROFILES_DIR.glob("PSI-*.json"))
    assert len(files) == 24
    for f in files:
        p = json.loads(f.read_text(encoding="utf-8"))
        assert p["generated"]["by"] == "rules (no LLM)"
        assert len(p["key_findings"]) >= 3 or p["key_findings_note"]
        for item in p["key_findings"] + p["measurements"]:
            src = item["source"]
            assert src["psi_id"] and src["file"] and (src["doi"] or src["psi_id"] == "REF")
        for fnd in p["key_findings"]:
            texts = by_src[(fnd["source"]["file"], fnd["source"]["page"])]
            assert any(norm(fnd["text"]) in t for t in texts), (p["id"], fnd["text"][:60])
        meta = json.loads((DATA_DIR / "metadata" / f"{p['id']}.json").read_text(encoding="utf-8-sig"))
        if p["summary_plain"]:
            assert norm(p["summary_plain"]) in norm(meta["objective"])
        assert set(p["habitat_relevance"]) == {"spacecraft", "iss", "lunar_base", "mars_base", "rover"}
        assert all(0 <= v <= 1 for v in p["habitat_relevance"].values())


def test_finding_score_rejects_non_results():
    from ai.textutil import finding_score
    assert finding_score("The FLEX tests showed that the cool flame can burn for a prolonged period of time.") >= 3
    assert finding_score("On the right, the view of the flame from the front window video camera is shown.") < 2
    assert finding_score("Their results showed that a CO2 volume fraction greater than 19.1% is needed.") < 3
    assert finding_score("The experiment records are found on the Data tab in the SAFFIRE-II record.") < 2


def test_no_llm_answer_is_verbatim_and_ranked(monkeypatch):
    """#13: no-llm answer leads with the best sentence; every shown sentence is verbatim from its passage."""
    from ai import llm
    from ai.textutil import norm
    monkeypatch.setattr(llm, "has_api_key", lambda: False)
    llm._client.cache_clear()
    r = qa.ask("Which materials were tested in Saffire?", use_cache=False)
    llm._client.cache_clear()
    assert r["mode"] == "no-llm" and r["answer"].startswith("**Best match [1]**")
    assert len(r["passages"]) == len(r["citations"]) >= 2
    assert len({p["sentence"] for p in r["passages"]}) == len(r["passages"])  # no duplicate sentences
    # verbatim check against the full source chunk, not the 600-char display snippet (#21: the best
    # sentence can sit past the cut, which made this fail on some machines)
    import json
    from ai.config import CHUNKS_FILE
    texts = {}
    for line in open(CHUNKS_FILE, encoding="utf-8"):
        c = json.loads(line)
        texts.setdefault((c["file"], c["page"]), []).append(norm(c["text"]))
    for p in r["passages"]:
        assert any(norm(p["sentence"]) in t for t in texts[(p["file"], p["page"])]), p["sentence"][:60]
    assert {p["psi_id"] for p in r["passages"]} <= {"PSI-98", "PSI-99", "PSI-100", "REF"}
    assert any("PMMA" in p["sentence"] for p in r["passages"])


def test_best_sentence_skips_citations_and_aims():
    from ai.textutil import best_sentence
    text = ("Opposed flow spread has been shown to depend on oxygen (Ross, 2001). "
            "It is hypothesized that flames spread faster in microgravity than in normal gravity. "
            "The measured spread rate was fastest at an intermediate value of flow speed in microgravity.")
    assert best_sentence(text, "Does flame spread faster in microgravity?").startswith("The measured spread rate")


def test_shipped_vectors_match_chunks():
    """#21: embeddings in the repo cover every chunk and match chunks.jsonl (fresh clone has vectors)."""
    import numpy as np
    from ai import vectors
    from ai.ingest import load_chunks
    meta = json_load(vectors.META_FILE)
    chunks = load_chunks()
    m = np.load(vectors.EMB_FILE)
    assert m.dtype == np.float16 and m.shape == (meta["n_vectors"], meta["dim"])
    assert meta["chunks_fingerprint"] == vectors.chunks_fingerprint(chunks)
    assert len(meta["chunk_rows"]) == len(chunks) and max(meta["chunk_rows"]) == len(m) - 1
    assert vectors.status()["ok"]


def test_vector_search_collapses_copies():
    """#15/#21: a text copied into several investigations is one hit that lists all its PSI-IDs."""
    from ai.retrieval import search
    hits = search("Which materials were tested in Saffire?", k=6)
    keys = [" ".join(h["text"].split())[:200].lower() for h in hits]
    assert len(keys) == len(set(keys))
    readme = [h for h in hits if "SAFFIRE-I and SAFFIRE-III Readme" in h["file"]]
    assert readme and set([readme[0]["psi_id"], *readme[0]["also_in"]]) >= {"PSI-98", "PSI-100"}


def test_stale_vectors_warn_not_crash(monkeypatch):
    from ai import vectors
    monkeypatch.setattr(vectors, "chunks_fingerprint", lambda chunks: "different")
    vectors._load.cache_clear()
    try:
        st = vectors.status()
        assert not st["ok"] and "stale" in st["warning"]
        with pytest.warns(UserWarning):
            assert vectors.search("flame", 3) == []
    finally:
        monkeypatch.undo()
        vectors._load.cache_clear()


def json_load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))
