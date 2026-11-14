"""Chunk embeddings shipped in the repo + numpy cosine search (issue #21).

    python -m ai.vectors          # (re)build processed/ai/embeddings.npy + embeddings_meta.json
    python -m ai.vectors --check  # print status

Why: the Chroma index is local-only (gitignored), so a fresh clone (judges, Streamlit Cloud) had no
vectors and search silently fell back to BM25. The embeddings now live in git (~3 MB, float16) and
search is plain numpy. Identical texts (one PDF copied into several investigations' folders) are
embedded once; every chunk maps to its vector row, and hits list all PSI-IDs that hold the text.

Query encoder: ai/encoder.py — the all-MiniLM-L6-v2 ONNX model via onnxruntime + tokenizers (no chromadb;
downloads ~80 MB once, CPU only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from functools import lru_cache

import numpy as np

from .config import AI_OUT, CHUNKS_FILE

EMB_FILE = AI_OUT / "embeddings.npy"
META_FILE = AI_OUT / "embeddings_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2 (ONNX via onnxruntime; identical to chromadb DefaultEmbeddingFunction)"
BATCH = 64


def _norm_text(t: str) -> str:
    return " ".join((t or "").split())


def chunks_fingerprint(chunks: list[dict]) -> str:
    """Hash of chunk ids + texts — stable across CRLF/LF checkouts, unlike hashing the file bytes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["id"].encode())
        h.update(_norm_text(c["text"]).encode("utf-8", "replace"))
    return h.hexdigest()


_ENCODER_ERROR: str | None = None  # remembered for the process: offline -> no download retry per query


def encoder():
    """Load the query encoder (ai.encoder: onnxruntime + tokenizers, no chromadb, #51). Raises
    ai.encoder.EncoderUnavailable with a readable reason; a failure is remembered for this process."""
    global _ENCODER_ERROR
    from .encoder import EncoderUnavailable, _session

    if _ENCODER_ERROR:
        raise EncoderUnavailable(_ENCODER_ERROR)
    try:
        return _session()
    except EncoderUnavailable as e:
        _ENCODER_ERROR = str(e)
        raise


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised all-MiniLM-L6-v2 embeddings (same numbers as chromadb's encoder: cosine 1.0, tested)."""
    from .encoder import embed as _embed

    encoder()  # raises EncoderUnavailable (remembered) instead of retrying a download
    return _embed(list(texts))


def build(chunks: list[dict] | None = None) -> dict:
    from .ingest import load_chunks

    chunks = chunks if chunks is not None else load_chunks()
    row_of_text: dict[str, int] = {}
    unique: list[str] = []
    rows = []
    for c in chunks:
        key = _norm_text(c["text"])
        if key not in row_of_text:
            row_of_text[key] = len(unique)
            unique.append(c["text"])
        rows.append(row_of_text[key])
    print(f"embedding {len(unique)} unique texts ({len(chunks) - len(unique)} duplicate chunks share a vector) …")
    m = embed(unique)
    AI_OUT.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, m.astype(np.float16))
    meta = {"model": MODEL_NAME, "dim": int(m.shape[1]), "n_chunks": len(chunks), "n_vectors": len(unique),
            "chunks_fingerprint": chunks_fingerprint(chunks), "chunk_ids": [c["id"] for c in chunks], "chunk_rows": rows}
    META_FILE.write_text(json.dumps(meta), encoding="utf-8")
    _load.cache_clear()
    return {k: v for k, v in meta.items() if k not in ("chunk_ids", "chunk_rows")}


@lru_cache(maxsize=1)
def _load():
    """(matrix float32 L2-normalised, meta, chunks) or (None, reason, None)."""
    from .ingest import load_chunks

    if not (EMB_FILE.exists() and META_FILE.exists() and CHUNKS_FILE.exists()):
        return None, f"missing {EMB_FILE.name} / {META_FILE.name} — run `python -m ai.vectors`", None
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()
    if meta.get("chunks_fingerprint") != chunks_fingerprint(chunks) or meta.get("chunk_ids") != [c["id"] for c in chunks]:
        return None, "embeddings are stale (chunks.jsonl changed) — run `python -m ai.vectors`", None
    m = np.load(EMB_FILE).astype(np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return m, meta, chunks


def status() -> dict:
    """{"ok", "n_vectors", "n_chunks", "model", "warning"} — shown by the app instead of failing silently."""
    m, meta, _ = _load()
    if m is None:
        return {"ok": False, "n_vectors": 0, "n_chunks": 0, "model": MODEL_NAME, "warning": meta}
    try:
        encoder()
    except Exception as e:  # offline download / onnxruntime missing -> can't encode queries
        return {"ok": False, "n_vectors": len(m), "n_chunks": meta["n_chunks"], "model": MODEL_NAME,
                "warning": f"Semantic search is off — {e} Keyword search still works."}
    return {"ok": True, "n_vectors": len(m), "n_chunks": meta["n_chunks"], "model": MODEL_NAME, "warning": None}


@lru_cache(maxsize=1)
def _rows_index():
    m, meta, chunks = _load()
    members: dict[int, list[int]] = {}
    for ci, r in enumerate(meta["chunk_rows"]):
        members.setdefault(r, []).append(ci)
    return members


def search(query: str, k: int, psi_ids: list[str] | None = None) -> list[dict]:
    """Cosine top-k over unique texts. Each hit = one representative chunk (inside `psi_ids` if given)
    plus `also_in`: the other PSI-IDs whose folders hold the identical text."""
    m, meta, chunks = _load()
    if m is None:
        warnings.warn(f"[ai.vectors] {meta}", stacklevel=2)
        return []
    q = embed([query])[0]
    sims = m @ q
    members = _rows_index()
    scope = set(psi_ids) if psi_ids else None
    out = []
    for r in np.argsort(-sims):
        idx = members.get(int(r), [])
        in_scope = [i for i in idx if scope is None or chunks[i]["psi_id"] in scope]
        if not in_scope:
            continue
        rep = chunks[in_scope[0]]
        others = sorted({chunks[i]["psi_id"] for i in idx} - {rep["psi_id"]})
        out.append({"psi_id": rep["psi_id"], "doi": rep.get("doi"), "file": rep["file"], "page": rep.get("page"),
                    "text": rep["text"], "score": round(float(sims[r]), 3), "also_in": others})
        if len(out) == k:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if not args.check:
        print(build())
    print(status())


if __name__ == "__main__":
    main()
