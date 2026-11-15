"""Issue #52: live answers inside a 19 s budget, number check on answers, dated cached answers.
All Gemini calls are mocked (429 / timeout / network error); no key or network needed.
Run: python -m pytest ai/tests/test_answers.py -q"""
import json

import pytest

from ai import llm, qa


@pytest.fixture(autouse=True)
def _private_cache(monkeypatch, tmp_path):
    """Mocked answers go to a throwaway cache, never to the committed processed/ai/qa_cache.json."""
    monkeypatch.setattr(qa, "QA_CACHE", tmp_path / "qa_cache.json")
    monkeypatch.setattr(qa, "_CACHE", None)

HITS = [{"psi_id": "PSI-26", "doi": "d1", "file": "a.pdf", "page": 3,
         "text": "The flame went out at 0.5 m/s in 21 % O2 (PSI-26)."},
        {"psi_id": "PSI-25", "doi": "d2", "file": "b.pdf", "page": 1, "text": "Flames spread more slowly."}]


def _fake(monkeypatch, behaviours, clock=None):
    """behaviours: model -> exception / JSON text / callable(config). Records (model, timeout_ms, retry attempts)."""
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            ho = config.http_options
            calls.append((model, ho.timeout, ho.retry_options.attempts if ho.retry_options else None))
            b = behaviours[model]
            if callable(b) and not isinstance(b, Exception):
                b = b(config)
            if isinstance(b, Exception):
                raise b
            return type("R", (), {"text": b, "prompt_feedback": None})()

    monkeypatch.setattr(llm, "_gemini_client", lambda: type("C", (), {"models": Models()})())
    monkeypatch.setenv("FIF_LLM", "gemini")
    monkeypatch.setattr(llm, "has_api_key", lambda *a: True)
    monkeypatch.delenv("FIF_MODEL", raising=False)
    return calls


def _quota():
    from google.genai import errors
    e = errors.ClientError.__new__(errors.ClientError)
    Exception.__init__(e, "429 RESOURCE_EXHAUSTED: quota exceeded")
    e.code, e.message, e.status = 429, "quota", "RESOURCE_EXHAUSTED"
    return e


OK = '{"answer": "A [1].", "used_excerpts": [1]}'


def test_budget_splits_time_and_disables_sdk_retries(monkeypatch):
    m0, m1 = llm.GEMINI_MODELS
    calls = _fake(monkeypatch, {m0: TimeoutError("read timed out"), m1: OK})
    out = llm.json_call("s", "u", {"type": "object"}, budget_s=19)
    assert out["_model"] == m1
    assert [c[0] for c in calls] == [m0, m1]
    assert all(c[1] <= 9_500 and c[2] == 1 for c in calls)          # ~9.5 s each, no SDK retries inside


def test_batch_jobs_keep_the_long_timeout(monkeypatch):
    calls = _fake(monkeypatch, {llm.GEMINI_MODELS[0]: OK})
    llm.json_call("s", "u", {"type": "object"})
    assert calls[0][1] == llm.GEMINI_TIMEOUT_MS and calls[0][2] is None  # profiles/briefings: unchanged


def test_budget_used_up_skips_the_second_model(monkeypatch):
    import time
    m0, m1 = llm.GEMINI_MODELS
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def slow(config):
        now[0] += 18.0                                               # the primary burned 18 of 19 s
        return TimeoutError("timed out")
    calls = _fake(monkeypatch, {m0: slow, m1: OK})
    with pytest.raises(llm.LLMUnavailable, match="time budget"):
        llm.json_call("s", "u", {"type": "object"}, budget_s=19)
    assert [c[0] for c in calls] == [m0]


@pytest.mark.parametrize("err", [_quota(), TimeoutError("timed out"), ConnectionError("network unreachable")])
def test_ask_falls_back_to_passages_on_429_timeout_network(monkeypatch, err):
    _fake(monkeypatch, {m: err for m in llm.GEMINI_MODELS})
    monkeypatch.setattr(qa, "search", lambda *a, **k: HITS)
    out = qa.ask("When did the flame go out?", use_cache=False)
    assert out["mode"] == "no-llm" and out["citations"][0]["psi_id"] == "PSI-26"  # verbatim passages view
    assert out.get("llm_error")


def test_answer_numbers_must_be_in_cited_passages(monkeypatch):
    ans = ("The flame went out at 0.5 m/s [1]. It burned for 42 s [2]. Flames spread more slowly [2]. "
           "Ignition took 7 s.")
    _fake(monkeypatch, {llm.GEMINI_MODELS[0]: json.dumps({"answer": ans, "used_excerpts": [1, 2]})})
    monkeypatch.setattr(qa, "search", lambda *a, **k: HITS)
    out = qa.ask("q", use_cache=False)
    assert out["answer"] == "The flame went out at 0.5 m/s [1]. Flames spread more slowly [2]."
    assert [d["numbers"] for d in out["numbers_dropped"]] == [[42.0], [7.0]]   # wrong source / no citation
    assert [c["psi_id"] for c in out["citations"]] == ["PSI-26", "PSI-25"]


def test_number_check_keeps_trailing_citations_lists_and_ids():
    d = qa._drop_unsupported_numbers
    assert d("Out at 0.5 m/s. [1] In PSI-26 too [1].", HITS)[1] == []        # "[n]" after the full stop
    assert d("1. Out at 0.5 m/s [1].\n2. Slower [2].", HITS)[1] == []        # list markers are not numbers
    text, dropped = d("Facts:\n- 0.5 m/s [1].\n- 99 kPa [2].", HITS)
    assert text == "Facts:\n- 0.5 m/s [1]." and dropped[0]["numbers"] == [99.0]


def test_cached_answers_are_dated(monkeypatch, tmp_path):
    _fake(monkeypatch, {llm.GEMINI_MODELS[0]: OK})
    monkeypatch.setattr(qa, "search", lambda *a, **k: HITS)
    monkeypatch.setattr(qa, "QA_CACHE", tmp_path / "qa_cache.json")
    monkeypatch.setattr(qa, "_CACHE", None)
    out = qa.ask("dated?", use_cache=True)
    assert len(out["date"]) == 10 and out["date"][4] == "-"
    cached = json.loads((tmp_path / "qa_cache.json").read_text(encoding="utf-8"))
    assert all(v["date"] == out["date"] for v in cached.values())
