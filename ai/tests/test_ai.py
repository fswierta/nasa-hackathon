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
        assert set(r["breakdown"]) == (set(DEFAULT_WEIGHTS) - {"o2", "pressure"}) | {"atmosphere"}


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


def test_atmosphere_is_scored_jointly_and_matches_coverage():
    """#64 item 1: O2 and pressure are scored together from the test rows, with coverage.json's definition of
    'near', so the ranker, Home and the gap banner never disagree."""
    import json
    from ai.config import PROCESSED_DIR
    from ai.ranking import atmosphere_near
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    for h in cov["habitats"]:
        near = atmosphere_near(round(h["o2_frac"], 4), round(h["pressure_kPa"], 2))
        assert {k: v["n"] for k, v in near.items() if v["n"]} == h["by_investigation"], h["name"]
    for name in HABITATS:
        for r in rank(HABITATS[name]):
            atm, near = r["breakdown"]["atmosphere"], r["near"]
            if near is None:
                assert atm is None                       # no O2 + pressure pair in its table: 'no data', not 0
            elif near["n"] == 0:
                assert atm < 0.5 and "tests within" not in r["why"]
                assert any("No test within ±6 pts O2" in g for g in r["gaps"])
            else:
                assert atm >= 0.5 and f"{near['n']} tests within ±6 pts O₂" in r["why"] or atm < 0.5
    lunar = rank(HABITATS["Lunar base"])
    assert lunar[0]["id"] == "PSI-69"                     # FLEX: 33 tests near 30 % O2 / 70.3 kPa
    cfig = next(r for r in lunar if r["id"] == "PSI-159")
    assert cfig["near"]["n"] == 0 and cfig["breakdown"]["atmosphere"] < 0.5
    assert all(r["id"] != "PSI-159" for r in lunar[:3])   # was #3 with O2 and pressure 'overlap 1.00'
    assert all((r["near"] or {}).get("n", 1) > 0 for r in lunar[:3] if r["near"] is not None)
    # o2 + pressure weights act together on the joint component; an explicit "atmosphere" weight wins
    only_atm = rank(HABITATS["Lunar base"], {k: 0.0 for k in DEFAULT_WEIGHTS} | {"o2": 0.5, "pressure": 0.5})
    assert only_atm[0]["score"] == only_atm[0]["breakdown"]["atmosphere"] == 1.0  # FLEX-2: habitat inside its matrix
    assert rank(HABITATS["Lunar base"], {k: 0.0 for k in DEFAULT_WEIGHTS} | {"atmosphere": 1.0})[0]["score"] == 1.0


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
    monkeypatch.setenv("FIF_LLM", "anthropic")  # never route this test to Gemini
    monkeypatch.setattr(llm, "has_api_key", lambda *a: with_key)
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
    """#12/#30: findings citing a non-existent excerpt are dropped; sources resolve to file+page;
    habitat scores are clipped to 0-1; measurements always come from the rule-based detector."""
    from ai import profiles
    hits = [{"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "a.pdf", "page": 4, "text": "camera"}]
    monkeypatch.setattr(profiles, "gather", lambda pid: hits)
    monkeypatch.setattr(profiles, "provider", lambda: "gemini")
    fake = {"summary_plain": "p", "summary_expert": "e", "fire_safety_pillar": ["prevention"],
            "habitat_relevance": {k: 1.5 for k in profiles.HABITAT_KEYS}, "earth_applications": [],
            "key_findings": [{"text": "ok", "excerpt": 1}, {"text": "invented", "excerpt": 7}]}
    monkeypatch.setattr(profiles, "json_call", lambda *a, **k: fake)
    prof = profiles.generate("PSI-26")
    llm_found = [f for f in prof["key_findings"] if f["provenance"] == "Google Gemini (free tier)"]
    assert [f["text"] for f in llm_found] == ["ok"]
    # topped up to 3 with verbatim rule-based quotes, each labelled as such
    assert len(prof["key_findings"]) == 3 and prof["generated"]["validation"]["findings_added_from_rules"] == 2
    assert all(f["provenance"].endswith("(rules, no LLM)") for f in prof["key_findings"][1:])
    assert prof["key_findings"][0]["source"] == {"psi_id": "PSI-26", "doi": "10.60555/58h4-dy80", "file": "a.pdf", "page": 4}
    assert all(v == 1.0 for v in prof["habitat_relevance"].values())
    assert prof["measurements"] and all(m["source"]["file"] for m in prof["measurements"])  # from rules, real sources


def test_rule_profiles_are_verbatim_and_sourced():
    """#12 (no LLM): all 24 profiles exist; findings are verbatim from the cited chunk; summaries quote metadata."""
    import json
    from ai.config import CHUNKS_FILE, DATA_DIR, PROFILES_DIR
    from ai.textutil import norm

    by_src = {}
    for line in open(CHUNKS_FILE, encoding="utf-8"):
        c = json.loads(line)
        by_src.setdefault((c["file"], c["page"]), []).append(norm(c["text"]))
    from ai.profiles import PROFILES_RULES_DIR
    files = sorted(PROFILES_RULES_DIR.glob("PSI-*.json"))
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
    monkeypatch.setattr(llm, "has_api_key", lambda *a: False)
    llm._client.cache_clear()
    r = qa.ask("Which materials were tested in Saffire?", use_cache=False)
    llm._client.cache_clear()
    assert r["mode"] == "no-llm" and r["answer"].startswith("**Best match [1]**")
    assert len(r["passages"]) == len(r["citations"]) >= 2
    assert len({p["sentence"] for p in r["passages"]}) == len(r["passages"])  # no duplicate sentences
    # verbatim check against the full source chunk, not the 600-char display snippet (#21: the best
    # sentence can sit past the cut, which made this fail on some machines)
    from ai.ingest import load_chunks
    texts = {}
    for c in load_chunks():  # PDF chunks + data cards: anything search() can return
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
    # (a query aimed at the readme's wording: since #19 the data cards rightly outrank it for "which materials…")
    hits = search("Saffire sample material composite fabric cotton fiberglass substrate", k=6)
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


# ----------------------------------------------------------------------------- #26 Gemini backend
@pytest.mark.parametrize("fif_llm,keys,expected", [
    ("", set(), "none"),
    ("", {"gemini"}, "gemini"),
    ("", {"anthropic"}, "anthropic"),
    ("", {"gemini", "anthropic"}, "gemini"),      # free tier preferred when both exist
    ("anthropic", {"gemini", "anthropic"}, "anthropic"),
    ("gemini", {"anthropic"}, "none"),            # forced provider without its key -> none, no crash
    ("none", {"gemini"}, "none"),
])
def test_provider_switch(monkeypatch, fif_llm, keys, expected):
    from ai import llm
    monkeypatch.setenv("FIF_LLM", fif_llm)
    monkeypatch.setattr(llm, "has_api_key", lambda p=None: (p in keys) if p else bool(keys))
    assert llm.provider() == expected


def _fake_gemini(monkeypatch, behaviours):
    """behaviours: model -> exception instance or JSON text."""
    from ai import llm
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(model)
            b = behaviours[model]
            if isinstance(b, Exception):
                raise b
            return type("R", (), {"text": b, "prompt_feedback": None})()

    monkeypatch.setattr(llm, "_gemini_client", lambda: type("C", (), {"models": Models()})())
    monkeypatch.setenv("FIF_LLM", "gemini")
    monkeypatch.setattr(llm, "has_api_key", lambda *a: True)
    monkeypatch.delenv("FIF_MODEL", raising=False)
    return calls


def _client_error(code, msg):
    from google.genai import errors
    e = errors.ClientError.__new__(errors.ClientError)
    Exception.__init__(e, msg)
    e.code, e.message, e.status = code, msg, msg
    return e


def test_gemini_quota_falls_back_to_lighter_model(monkeypatch):
    from ai import llm
    calls = _fake_gemini(monkeypatch, {
        llm.GEMINI_MODELS[0]: _client_error(429, "429 RESOURCE_EXHAUSTED: quota exceeded"),
        llm.GEMINI_MODELS[1]: '{"answer": "A [1].", "used_excerpts": [1]}',
    })
    out = llm.json_call("s", "u", {"type": "object"})
    assert out["answer"] == "A [1]." and out["_model"] == llm.GEMINI_MODELS[1]
    assert calls == llm.GEMINI_MODELS


def test_gemini_errors_become_unavailable(monkeypatch):
    from ai import llm
    _fake_gemini(monkeypatch, {m: _client_error(429, "RESOURCE_EXHAUSTED") for m in llm.GEMINI_MODELS})
    with pytest.raises(llm.LLMUnavailable, match="quota"):
        llm.json_call("s", "u", {"type": "object"})
    _fake_gemini(monkeypatch, {llm.GEMINI_MODELS[0]: _client_error(400, "INVALID_ARGUMENT")})
    with pytest.raises(llm.LLMUnavailable, match="rejected"):
        llm.json_call("s", "u", {"type": "object"})
    _fake_gemini(monkeypatch, {llm.GEMINI_MODELS[0]: "not json"})
    with pytest.raises(llm.LLMUnavailable, match="invalid JSON"):
        llm.json_call("s", "u", {"type": "object"})


def test_ask_with_gemini_labels_mode_and_keeps_guards(monkeypatch):
    """Answer comes from the mocked provider; invented citation [9] is dropped; mode/model are labelled."""
    from ai import llm
    monkeypatch.setattr(qa, "_save_cache", lambda: None)   # a mocked answer must never reach the committed cache
    _fake_gemini(monkeypatch, {llm.GEMINI_MODELS[0]: '{"answer": "Fact [2]. Other [9].", "used_excerpts": [2, 9]}'})
    hits = [{"psi_id": "PSI-26", "doi": "d1", "file": "a.pdf", "page": 3, "text": "x"},
            {"psi_id": "PSI-25", "doi": "d2", "file": "b.pdf", "page": 1, "text": "y"}]
    monkeypatch.setattr(qa, "search", lambda *a, **k: hits)
    out = qa.ask("q", use_cache=False)
    assert out["mode"] == "gemini" and out["model"] == llm.GEMINI_MODELS[0]
    assert out["answer"] == "Fact [1]. Other ." and [c["psi_id"] for c in out["citations"]] == ["PSI-25"]
    assert out["passages"][0]["psi_id"] == "PSI-25"


def test_no_key_means_no_llm_everywhere(monkeypatch):
    """Without any key everything behaves exactly as before #26 (extractive answers, rule profiles)."""
    from ai import llm
    from ai import config
    for v in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "FIF_LLM"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_env", lambda: None)  # has_api_key() would otherwise re-read .env
    assert llm.provider() == "none" and llm.model_name() == "none"
    with pytest.raises(llm.LLMUnavailable):
        llm.json_call("s", "u", {"type": "object"})


@pytest.mark.skipif(not __import__("ai.config", fromlist=["has_api_key"]).has_api_key("gemini"),
                    reason="no GEMINI_API_KEY in .env")
def test_gemini_live_smoke():
    """One tiny real call (free tier). Skipped when there is no key."""
    from ai import llm
    if llm.provider() != "gemini":
        pytest.skip("FIF_LLM routes away from gemini")
    try:
        out = llm.json_call("Reply with JSON only.", "Set ok to true.",
                            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, max_tokens=64)
    except llm.LLMUnavailable as e:
        if any(w in str(e).lower() for w in ("quota", "high demand", "server error", "cannot reach")):
            pytest.skip(f"Gemini transiently unavailable: {str(e)[:80]}")  # Google-side, not our bug
        raise
    assert out["ok"] is True and out["_model"] in llm.GEMINI_MODELS


# ----------------------------------------------------------------------------- #30 LLM profiles
def _chunk_locations():
    from ai.ingest import load_chunks
    from ai.textutil import norm
    locs = {}
    for c in load_chunks():  # documents + data cards
        locs.setdefault((c["file"], c["page"]), []).append(norm(c["text"]))
    return locs


def test_all_profiles_have_contract_keys_and_real_citations():
    """#30: every profile (LLM and rule-based) has the contract keys, a provenance label, and cites
    only file+page locations that exist in chunks.jsonl; LLM findings use only numbers from the cited chunk."""
    import json
    from ai.profiles import CONTRACT_KEYS, PROFILES_DIR, PROFILES_RULES_DIR, numbers_in, unsupported_numbers
    locs = _chunk_locations()
    seen = 0
    for d in (PROFILES_DIR, PROFILES_RULES_DIR):
        for f in sorted(d.glob("PSI-*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            seen += 1
            assert set(CONTRACT_KEYS) <= set(p), f.name
            assert p["generated"]["by"] in ("Google Gemini (free tier)", "Anthropic Claude", "rules (no LLM)")
            for item in p["key_findings"] + p["measurements"]:
                src = item["source"]
                assert (src["file"], src["page"]) in locs, (f.name, src)
            if d == PROFILES_DIR:  # LLM finding: no number that the cited chunk doesn't contain
                for fnd in p["key_findings"]:
                    src = fnd["source"]
                    assert not unsupported_numbers(fnd["text"], locs[(src["file"], src["page"])]), (f.name, fnd["text"])
    assert seen >= 24


def test_load_profile_prefers_llm(tmp_path, monkeypatch):
    import json
    from ai import profiles
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path / "llm")
    monkeypatch.setattr(profiles, "PROFILES_RULES_DIR", tmp_path / "rules")
    (tmp_path / "rules").mkdir(); (tmp_path / "llm").mkdir()
    (tmp_path / "rules" / "PSI-1.json").write_text(json.dumps({"id": "PSI-1", "generated": {"by": "rules (no LLM)"}}))
    assert profiles.load_profile("PSI-1")["generated"]["by"] == "rules (no LLM)"
    (tmp_path / "llm" / "PSI-1.json").write_text(json.dumps({"id": "PSI-1", "generated": {"by": "Google Gemini (free tier)"}}))
    assert profiles.load_profile("PSI-1")["generated"]["by"] == "Google Gemini (free tier)"
    assert profiles.load_profile("PSI-2") is None


def test_unsupported_numbers():
    from ai.profiles import unsupported_numbers
    src = ["Samples were 75% cotton and 25% fiberglass, flow 20 cm/s, 0.21 mole fraction."]
    assert unsupported_numbers("The fabric is 75 % cotton at 20 cm/s and 0.21 O2.", src) == set()
    assert unsupported_numbers("Spread rate was 2.3 mm/s at 30 cm/s.", src) == {2.3, 30.0}
    assert unsupported_numbers("See PSI-26 for 20 cm/s.", src) == set()  # PSI-IDs are not values


def test_generate_drops_unsupported_numbers(monkeypatch):
    """A finding citing excerpt 1 but quoting a number absent from it is dropped; expert sentences likewise."""
    from ai import profiles
    hits = [{"psi_id": "PSI-26", "doi": "d", "file": "a.pdf", "page": 2, "text": "Flames went out below 1 cm/s flow."}]
    monkeypatch.setattr(profiles, "gather", lambda pid: hits)
    monkeypatch.setattr(profiles, "provider", lambda: "gemini")
    monkeypatch.setattr(profiles, "json_call", lambda *a, **k: {
        "summary_plain": "Plain.", "summary_expert": "Extinction occurred below 1 cm/s. The LOI was 14 %.",
        "key_findings": [{"text": "Extinction below 1 cm/s.", "excerpt": 1}, {"text": "LOI of 14 % found.", "excerpt": 1},
                         {"text": "Ghost.", "excerpt": 4}],
        "fire_safety_pillar": ["prevention"], "habitat_relevance": {k: 0.5 for k in profiles.HABITAT_KEYS},
        "earth_applications": [], "_model": "m"})
    p = profiles.generate("PSI-26")
    assert [f["text"] for f in p["key_findings"] if f["provenance"] == "Google Gemini (free tier)"] ==         ["Extinction below 1 cm/s."]
    assert p["summary_expert"] == "Extinction occurred below 1 cm/s."
    v = p["generated"]["validation"]
    assert len(v["findings_dropped"]) == 2 and v["expert_sentences_dropped"] == ["The LOI was 14 %."]
    assert p["generated"]["by"] == "Google Gemini (free tier)" and p["measurements_source"].startswith("rules")


# ----------------------------------------------------------------------------- #19 data cards
def test_datacards_are_current_and_match_etl():
    """Committed processed/ai/datacards.jsonl equals a fresh build from investigations.json + tests.csv
    (re-run `python -m ai.datacards` after ETL changes), and its numbers are the ETL numbers."""
    import json
    import pandas as pd
    from ai import datacards
    from ai.config import PROCESSED_DIR
    committed = datacards.load_cards()
    invs = json.load(open(PROCESSED_DIR / "investigations.json", encoding="utf-8"))
    tests = pd.read_csv(PROCESSED_DIR / "tests.csv")
    assert len(committed) == len(invs) == 24
    for card, inv in zip(committed, invs):
        assert card["psi_id"] == inv["id"] and card["doi"] == inv["doi"] and card["file"] == datacards.SOURCE_FILE
        fresh = datacards.card_text(inv, tests[tests["investigation_id"] == inv["id"]])
        assert card["text"] == fresh, f"{inv['id']}: datacards.jsonl is stale — run python -m ai.datacards"
        if inv.get("n_tests"):
            assert f"ran {inv['n_tests']} tests" in card["text"]
        if inv.get("o2_range") and inv["o2_range"][0] is not None:
            assert f"{inv['o2_range'][0] * 100:.0f}" in card["text"]


def _ask_no_llm(monkeypatch, q):
    from ai import llm
    monkeypatch.setenv("FIF_LLM", "none")
    llm._client.cache_clear()
    return qa.ask(q, use_cache=False)


@pytest.mark.parametrize("q,pid,must", [
    ("Which materials were tested in BASS?", "PSI-26", "sample materials tested"),
    ("What O2 levels did FLEX use?", "PSI-69", "oxygen (O2) levels in"),
    ("How many tests did BASS-II run?", "PSI-25", "tests according to its PSI experimental table"),
    ("What O2 level stopped droplet burning in FLEX?", "PSI-69", "the flame was extinguished in"),
])
def test_factual_questions_lead_with_table_values(monkeypatch, q, pid, must):
    from ai.datacards import SOURCE_FILE
    r = _ask_no_llm(monkeypatch, q)
    lead = r["passages"][0]
    assert lead["psi_id"] == pid and lead["file"] == SOURCE_FILE and must in lead["sentence"], lead
    assert r["citations"][0]["doi"]  # the data card carries the investigation's DOI


def test_open_questions_are_not_treated_as_lookups(monkeypatch):
    from ai.retrieval import FACT_QUESTION
    assert FACT_QUESTION.match("Which materials were tested in Saffire?")
    assert FACT_QUESTION.match("How many tests did BASS-II run?")
    assert FACT_QUESTION.match("What O2 levels did FLEX use?")
    assert not FACT_QUESTION.match("What is known about fires at Moon-base atmospheres (30–34 % O2)?")
    assert not FACT_QUESTION.match("Does flame spread faster in microgravity than in 1 g?")
    r = _ask_no_llm(monkeypatch, "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?")
    assert "common plastic could steadily burn" in r["passages"][0]["sentence"]


def test_query_terms_keep_chemical_formulas():
    from ai.textutil import query_terms
    assert "o2" in query_terms("What O2 levels did FLEX use?")
    assert "use" not in query_terms("What O2 levels did FLEX use?")
