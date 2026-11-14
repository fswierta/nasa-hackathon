"""Retrieve document chunks for a question: Chroma vector search, keyword fallback."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from .config import CHUNKS_FILE, INDEX_DIR

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*")
STOP = set("the a an of and or in on for to is are was were be by with at from as that this it its what which how why do does".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOP and len(t) > 1]


@lru_cache(maxsize=1)
def _chunks():
    from .ingest import load_chunks
    return load_chunks() if CHUNKS_FILE.exists() else []


@lru_cache(maxsize=1)
def _bm25_index():
    docs = [_tokens(c["text"]) for c in _chunks()]
    df = Counter(t for d in docs for t in set(d))
    avg = sum(map(len, docs)) / max(1, len(docs))
    return docs, df, avg


def keyword_search(query: str, k: int, psi_ids: list[str] | None = None) -> list[dict]:
    docs, df, avg = _bm25_index()
    n = len(docs)
    q = _tokens(query)
    scored = []
    for i, d in enumerate(docs):
        c = _chunks()[i]
        if psi_ids and c["psi_id"] not in psi_ids:
            continue
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return [dict(c, score=round(s, 3)) for s, c in scored[:k]]


def vector_search(query: str, k: int, psi_ids: list[str] | None = None) -> list[dict]:
    from .ingest import get_collection

    col = get_collection()
    if col.count() == 0:
        return []
    where = {"psi_id": {"$in": psi_ids}} if psi_ids else None
    res = col.query(query_texts=[query], n_results=k, where=where)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"psi_id": meta["psi_id"], "doi": meta["doi"] or None, "file": meta["file"],
                    "page": None if meta["page"] == -1 else meta["page"], "text": doc,
                    "score": round(1 - dist, 3)})
    return out


@lru_cache(maxsize=1)
def _acronym_patterns():
    """Map each investigation to regexes for its acronym parts (e.g. 'SAFFIRE-II', 'BASS', 'FLEX-2')."""
    from .catalog import load_investigations

    pats = {}
    for inv in load_investigations():
        names = {inv["id"]}
        for part in re.split(r"\s*/\s*|\s+-\s+", inv.get("acronym") or ""):
            part = part.strip()
            if len(part) >= 3 and not part.startswith("("):
                names.add(part)
        pats[inv["id"]] = [re.compile(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", re.I) for n in names]
    return pats


def mentioned_investigations(query: str) -> list[str]:
    """Investigations named in the question. 'SAFFIRE' alone matches SAFFIRE-I/II/III."""
    hits = [pid for pid, pats in _acronym_patterns().items() if any(p.search(query) for p in pats)]
    if not hits:  # family names without suffix, e.g. "SAFFIRE", "ACME", "SAME"
        from .catalog import load_investigations
        words = {w.upper() for w in re.findall(r"[A-Za-z]{3,}", query)}
        hits = [inv["id"] for inv in load_investigations()
                if (inv.get("acronym") or "").split("-")[0].split()[0].upper() in words]
    return hits


def search(query: str, k: int = 6, psi_ids: list[str] | None = None) -> list[dict]:
    """Hybrid retrieval: reciprocal-rank fusion of vector and BM25 results.

    If the question names investigations (and no psi_ids filter is given), results are
    restricted to those plus the general Researcher's Guide.
    """
    if psi_ids is None:
        named = mentioned_investigations(query)
        if named:
            psi_ids = named + ["REF"]
    pool = 3 * k
    lists = []
    if (INDEX_DIR / "chroma.sqlite3").exists():
        try:
            lists.append(vector_search(query, pool, psi_ids))
        except Exception as e:
            print(f"[retrieval] vector search unavailable ({e}); using keyword search")
    lists.append(keyword_search(query, pool, psi_ids))

    fused: dict[tuple, dict] = {}
    for results in lists:
        for rank_, h in enumerate(results):
            key = (h["file"], h["page"], h["text"][:80])
            entry = fused.setdefault(key, dict(h, score=0.0))
            entry["score"] += 1.0 / (60 + rank_)
    out = sorted(fused.values(), key=lambda h: -h["score"])
    return [dict(h, score=round(h["score"], 4)) for h in out[:k]]
