"""Build the RAG knowledge base: extract text page by page, chunk, embed (numpy, shipped in git).

    python -m ai.ingest            # extract + embeddings.npy (re-uses chunks.jsonl if present)
    python -m ai.ingest --chroma   # also build the optional local Chroma index
    python -m ai.ingest --rebuild  # re-extract everything

Sources: data/documents/PSI-XX/* (PDF, DOCX), the Researcher's Guide PDF, and the text fields
of data/metadata/PSI-XX.json. Every chunk keeps psi_id, doi, file and page so answers can cite it.
Identical files copied into several investigations are indexed once per investigation id
(the text is shared, but the citation must point at each PSI record that contains it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from .catalog import by_id
from .config import CHUNKS_FILE, COLLECTION, DATA_DIR, INDEX_DIR

CHUNK_CHARS = 3000
OVERLAP_CHARS = 400
REF_GUIDE = DATA_DIR / "reference" / "NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf"
META_FIELDS = ("objective", "hypothesis", "approach", "researchImpacts", "description")


def _clean(text: str) -> str:
    # PDFs can yield lone surrogates (math italic glyphs) that UTF-8 cannot encode
    text = text.encode("utf-8", "replace").decode("utf-8").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _long_path(p: Path) -> str:
    """Windows paths > 260 chars need the \\\\?\\ prefix (brief §5)."""
    s = str(p.resolve())
    if sys.platform == "win32" and len(s) > 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(_long_path(path))
    except Exception as e:  # corrupt / not really a PDF
        print(f"  ! cannot open {path.name}: {e}")
        return []
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = _clean(page.extract_text() or "")
        except Exception:
            txt = ""
        if txt:
            pages.append((i, txt))
    return pages


def extract_docx(path: Path) -> list[tuple[int | None, str]]:
    import docx

    d = docx.Document(_long_path(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    txt = _clean("\n".join(parts))
    return [(None, txt)] if txt else []


def _split(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):  # try to break on a paragraph / sentence boundary
            cut = max(text.rfind("\n", start + CHUNK_CHARS // 2, end), text.rfind(". ", start + CHUNK_CHARS // 2, end))
            if cut > start:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [c for c in out if c]


def _chunk_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def iter_sources():
    """Yield (psi_id, doi, file_label, page, text) records."""
    invs = by_id()
    # 1) metadata text
    for pid, inv in invs.items():
        meta_path = DATA_DIR / "metadata" / f"{pid}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        header = f"{pid} {inv.get('acronym')}: {inv.get('title')}"
        for field in META_FIELDS:
            val = (meta.get(field) or "").strip()
            if val:
                yield pid, inv.get("doi"), f"metadata/{pid}.json#{field}", None, f"{header}\n{field}:\n{_clean(val)}"
    # 2) documents
    doc_root = DATA_DIR / "documents"
    for folder in sorted(doc_root.iterdir()):
        pid = folder.name
        doi = invs.get(pid, {}).get("doi")
        for f in sorted(folder.iterdir()):
            ext = f.suffix.lower()
            if ext == ".pdf":
                pages = extract_pdf(f)
            elif ext == ".docx":
                pages = extract_docx(f)
            else:
                print(f"  - skip {pid}/{f.name} ({ext} not supported)")
                continue
            for page, text in pages:
                yield pid, doi, f"documents/{pid}/{f.name}", page, text
    # 3) Researcher's Guide (general background, not tied to one investigation)
    if REF_GUIDE.exists():
        for page, text in extract_pdf(REF_GUIDE):
            yield "REF", None, f"reference/{REF_GUIDE.name}", page, text


def build_chunks() -> list[dict]:
    chunks = []
    for pid, doi, file_label, page, text in iter_sources():
        for j, piece in enumerate(_split(text)):
            chunks.append({
                "id": _chunk_id(pid, file_label, page, j),
                "psi_id": pid, "doi": doi, "file": file_label, "page": page, "text": piece,
            })
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks


def load_chunks(include_datacards: bool = True) -> list[dict]:
    """Document chunks (chunks.jsonl) plus, by default, the per-investigation data cards
    (processed/ai/datacards.jsonl, built by `python -m ai.datacards` from the ETL tables)."""
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    if include_datacards:
        from .datacards import load_cards
        chunks += load_cards()
    return chunks


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index(chunks: list[dict]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[{"psi_id": c["psi_id"], "doi": c["doi"] or "", "file": c["file"],
                        "page": c["page"] if c["page"] is not None else -1} for c in part],
        )
        print(f"  indexed {min(i + batch, len(chunks))}/{len(chunks)}", end="\r")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="re-extract text even if chunks.jsonl exists")
    ap.add_argument("--extract-only", action="store_true", help="write chunks.jsonl but skip embeddings")
    ap.add_argument("--chroma", action="store_true", help="also build the optional local Chroma index")
    args = ap.parse_args(argv)

    if CHUNKS_FILE.exists() and not args.rebuild:
        chunks = load_chunks()
        print(f"Re-using {len(chunks)} chunks from {CHUNKS_FILE}")
    else:
        print("Extracting text ...")
        chunks = build_chunks()
        print(f"Wrote {len(chunks)} chunks to {CHUNKS_FILE}")
    if not args.extract_only:
        from . import vectors

        print("Embedding chunks -> processed/ai/embeddings.npy (committed; first run downloads the ONNX model) ...")
        print(vectors.build(chunks))
    if args.chroma:
        print("Building optional local Chroma index ...")
        build_index(chunks)
        print(f"Index ready in {INDEX_DIR}")


if __name__ == "__main__":
    main()
