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
    """Numpy cosine search over the embeddings shipped in the repo (ai/vectors.py, #21).
    Falls back to a local Chroma index only if the shipped embeddings are unusable."""
    from . import vectors

    if vectors.status()["ok"]:
        return vectors.search(query, k, psi_ids)
    return _chroma_search(query, k, psi_ids)


def status() -> dict:
    """Search health for the app: {"vector": vectors.status(), "keyword_chunks": n}."""
    from . import vectors

    return {"vector": vectors.status(), "keyword_chunks": len(_chunks())}


def _chroma_search(query: str, k: int, psi_ids: list[str] | None = None) -> list[dict]:
    if not (INDEX_DIR / "chroma.sqlite3").exists():
        return []
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


def warmup() -> None:
    """Load chunks, BM25 stats, shipped embeddings and the query encoder (~7 s once; first run
    downloads the ~80 MB ONNX model). Call at app start-up, e.g. in Streamlit:
    `@st.cache_resource def _w(): ai.retrieval.warmup()`. Show `ai.retrieval.status()` if not ok."""
    _bm25_index()
    _acronym_patterns()
    search("flame spread oxygen", k=1)


@lru_cache(maxsize=1)
def _acronym_patterns():
    """Map each investigation to regexes for its acronym parts (e.g. 'SAFFIRE-II', 'BASS', 'FLEX-2')."""
    from .catalog import load_investigations

    invs = load_investigations()
    acr_of = {inv["id"]: inv.get("acronym") for inv in invs}
    parts_of = {}
    for inv in invs:
        parts = set()
        for part in re.split(r"\s*/\s*|\s+-\s+", inv.get("acronym") or ""):
            part = part.strip()
            if len(part) >= 3 and not part.startswith("("):
                parts.add(part.upper())
        parts_of[inv["id"]] = parts
    # a part shared by several investigations ("ACME") is a family name, not an acronym
    counts = {}
    for parts in parts_of.values():
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
    for inv in invs:  # family prefixes ('ACME' in 'ACME BRE', 'ACME CFI-G', ...) count as shared too
        fam = _family(inv.get("acronym"))
        counts[fam] = counts.get(fam, 0) + 1
    pats = {}
    for pid, parts in parts_of.items():
        full = (acr_of[pid] or "").strip().upper()
        # keep a part if unique, or if it IS this investigation's whole acronym ('BASS', 'FLEX', 'SAME')
        names = {pid} | {p for p in parts if counts[p] == 1 or p == full}
        pats[pid] = [re.compile(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", re.I) for n in names]
    return pats


def _family(acronym: str) -> str:
    """'SAFFIRE-II' -> 'SAFFIRE', 'ACME CFI-G' -> 'ACME', 'DAFT / DAFT-2' -> 'DAFT'."""
    return re.split(r"[\s/\-]", (acronym or "").strip())[0].upper()


def mentioned_investigations(query: str) -> list[str]:
    """Investigations named in the text. A bare family name ('SAFFIRE', 'ACME') matches every
    member of the family, unless a specific member (e.g. 'SAFFIRE-II', 'ACME BRE') is named."""
    from .catalog import load_investigations

    invs = load_investigations()
    hits = [pid for pid, pats in _acronym_patterns().items() if any(p.search(query) for p in pats)]
    hit_families = {_family(by["acronym"]) for by in invs if by["id"] in hits}
    words = {w.upper() for w in re.findall(r"(?<![\w-])([A-Za-z]{3,})(?![\w-])", query)}
    for inv in invs:
        fam = _family(inv.get("acronym"))
        if fam and fam in words and fam not in hit_families and inv["id"] not in hits:
            hits.append(inv["id"])
    return hits


def search(query: str, k: int = 6, psi_ids: list[str] | None = None, max_per_file: int | None = None) -> list[dict]:
    """Hybrid retrieval: reciprocal-rank fusion of vector and BM25 results.

    If the question names investigations (and no psi_ids filter is given), results are
    restricted to those plus the general Researcher's Guide.
    """
    if psi_ids is None:
        named = mentioned_investigations(query)
        if named:
            psi_ids = named + ["REF"]
    pool = 4 * k
    lists = []
    try:
        lists.append(vector_search(query, pool, psi_ids))
    except Exception as e:  # e.g. encoder download blocked — keyword search still works
        print(f"[retrieval] vector search unavailable ({e}); using keyword search")
    lists.append(keyword_search(expand_query(query), pool * 2, psi_ids))

    # fuse by text, not by file: the same PDF is copied into several investigations' folders
    # (e.g. the Researcher's Guide under PSI-20, PSI-22 and REF) — show it once, listing every PSI-ID
    fused: dict[str, dict] = {}
    for results in lists:
        seen_here: set[str] = set()
        rank_ = 0
        for h in results:
            key = re.sub(r"\s+", " ", h["text"])[:200].lower()
            entry = fused.setdefault(key, dict(h, score=0.0, also_in=list(h.get("also_in", []))))
            for pid in [h["psi_id"], *h.get("also_in", [])]:
                if pid != entry["psi_id"] and pid not in entry["also_in"]:
                    entry["also_in"].append(pid)
            if key in seen_here:  # a copy of a text already ranked in this list adds no score
                continue
            seen_here.add(key)
            entry["score"] += 1.0 / (60 + rank_)
            rank_ += 1
    out, per_file = [], Counter()
    for h in sorted(fused.values(), key=lambda h: -h["score"]):
        if per_file[h["file"]] >= (max_per_file or MAX_PER_FILE):  # keep the list varied across documents
            continue
        per_file[h["file"]] += 1
        out.append(dict(h, score=round(h["score"], 4)))
        if len(out) == k:
            break
    return out


MAX_PER_FILE = 2


# Domain query expansion for the keyword (BM25) search only; the vector search gets the raw question.
EXPANSIONS = [
    (r"\bmaterials?\b|\bsamples?\b|\bfuels? tested\b",
     "sample material samples fabric PMMA Nomex silicone SIBAL cotton fiberglass Kapton Teflon"),
    (r"\b(?:o2|oxygen)\b.*\b(?:stop|stopped|extinguish|limit|minimum|lowest|level)\b|\bLOI\b",
     "limiting oxygen index LOI extinction oxygen mole fraction flammability limit lower"),
    (r"\bsmoke\b", "smoke particle size distribution larger particles aerosol detector"),
    (r"\bspread\b", "flame spread rate opposed concurrent flow normal gravity microgravity"),
    (r"moon|lunar|mars|exploration atmosphere|habitat atmosphere|56|70 kpa|8\.2 psi|10\.2 psi",
     "exploration atmosphere elevated oxygen reduced pressure 34% 8.2 psi NASA proposed atmospheric conditions"),
    (r"suppress|extinguish", "suppressant CO2 helium nitrogen extinction diluent"),
]


def term_weights(text: str) -> dict[str, float]:
    """IDF of each term in `text` over all chunks, scaled to 0.2..1 (rare terms weigh most)."""
    from .textutil import query_terms

    docs, df, _ = _bm25_index()
    n = max(1, len(docs))
    terms = set(query_terms(text))
    idf = {t: math.log(1 + n / (1 + df.get(t, 0))) for t in terms}
    top = max(idf.values(), default=1.0) or 1.0
    return {t: max(0.2, v / top) for t, v in idf.items()}


def investigation_name_terms(query: str) -> set[str]:
    """Lower-case words of the investigation names mentioned in `query` (e.g. {'same'}, {'saffire'})."""
    from .catalog import by_id

    out = set()
    for pid in mentioned_investigations(query):
        out |= {w.lower() for w in re.findall(r"[A-Za-z0-9]+", by_id()[pid].get("acronym") or "")}
    return out


def expand_query(query: str) -> str:
    extra = [terms for pat, terms in EXPANSIONS if re.search(pat, query, re.I)]
    return query + (" " + " ".join(extra) if extra else "")
