"""Issue #51: query encoder without chromadb, offline fallback, slim runtime requirements.
Run: python -m pytest ai/tests/test_encoder.py -q"""
import sys

import numpy as np
import pytest

QUERIES = ["Which materials were tested in Saffire?", "What O2 level stopped droplet burning in FLEX?",
           "How does smoke differ in microgravity (SAME)?", "Does flame spread faster in microgravity than in 1 g?",
           "fires at Moon-base atmospheres (30-34 % O2, ~56-70 kPa)", "cool flames after radiative extinction",
           "PMMA rod sizes BASS-II", "smoke detector particle size", "CO2 helium suppressant", "Nomex flammability",
           "How thick is the cotton-fiberglass fabric?", "burning rate constant K heptane",
           "electric field partial gravity proxy", "soot volume fraction", "spacesuit 100 % oxygen",
           "ventilation flow speed extinction limit", "DAFT dust P-Trak", "ACME BRE burner emulator",
           "LOI limiting oxygen index", "word " * 400]


def test_encoder_matches_chromadb():
    """Same model, tokenizer, pooling and normalisation as chromadb's encoder (cosine >= 0.999 on 20 queries)."""
    pytest.importorskip("chromadb", reason="chromadb is optional now (ai/requirements-ingest.txt)")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    from ai.encoder import embed
    ours = embed(QUERIES)
    ref = np.asarray(DefaultEmbeddingFunction()(QUERIES), dtype=np.float32)
    cos = (ours * ref).sum(1) / np.linalg.norm(ours, axis=1) / np.linalg.norm(ref, axis=1)
    assert ours.shape == (20, 384) and cos.min() >= 0.999


def test_vectors_work_without_chromadb(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)        # `import chromadb` now fails
    from ai import vectors
    v = vectors.embed(["flame spread in microgravity"])
    assert v.shape == (1, 384) and abs(float(np.linalg.norm(v)) - 1) < 1e-5
    assert vectors.status()["ok"]


def test_offline_falls_back_to_keyword_search_with_a_clear_message(monkeypatch):
    from ai import encoder, retrieval, vectors

    def offline():
        raise encoder.EncoderUnavailable("could not download the all-MiniLM-L6-v2 model (URLError: offline); "
                                         "search falls back to keywords")
    monkeypatch.setattr(encoder, "_session", offline)
    monkeypatch.setattr(vectors, "_ENCODER_ERROR", None)
    try:
        st = vectors.status()
        assert not st["ok"] and "could not download" in st["warning"] and "Keyword search still works" in st["warning"]
        calls = []
        monkeypatch.setattr(encoder, "_session", lambda: calls.append(1) or offline())
        hits = retrieval.search("Which materials were tested in Saffire?", k=4)
        assert hits and not calls                               # remembered failure: no retry per query
    finally:
        monkeypatch.setattr(vectors, "_ENCODER_ERROR", None)


def test_runtime_requirements_are_slim():
    from ai.config import ROOT
    runtime = {l.split("=")[0].split("<")[0].split(">")[0].strip().lower()
               for l in (ROOT / "ai" / "requirements.txt").read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith(("#", "-r"))}
    assert {"onnxruntime", "tokenizers"} <= runtime
    assert not runtime & {"chromadb", "pypdf", "python-docx"}   # ingest-only -> ai/requirements-ingest.txt
