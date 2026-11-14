"""Write ai/examples.md: the 5 demo questions answered by ask() in no-llm mode (issue #13).

    python -m ai.examples

For each question: the answer exactly as ask() returns it, a citation table (PSI-ID, DOI link,
file, page) and an automatic check that every cited file exists locally and the cited page is
within the PDF's page count. `REVIEW` holds the human/agent verdict on whether the passages
answer the question; `DATA_CHECK` adds numbers computed from processed/tests.csv where the
documents only answer qualitatively.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .catalog import by_id
from .config import DATA_DIR, PROCESSED_DIR, ROOT
from .qa import ask

OUT = ROOT / "ai" / "examples.md"

QUESTIONS = [
    "Does flame spread faster in microgravity than in 1 g?",
    "Which materials were tested in Saffire?",
    "What O2 level stopped droplet burning in FLEX?",
    "How does smoke differ in microgravity (SAME)?",
    "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?",
]

REVIEW = {
    QUESTIONS[0]: "Partly answered. BASS passages show spread rate depends on flow speed (fastest at an intermediate "
                  "speed; opposed-flow flames may spread faster at low flow) — i.e. 'it depends on ventilation', not a "
                  "simple yes/no. The E-FIELD passage is only loosely related.",
    QUESTIONS[1]: "Answered. Saffire-I/III: SIBAL cotton-fiberglass fabric; Saffire-II: SIBAL, silicone, structural "
                  "PMMA, flat PMMA, PMMA with Nomex.",
    QUESTIONS[2]: "Qualitatively answered: passage [2] says FLEX found a Limiting Oxygen Index significantly lower than "
                  "in 1 g. The lead sentence is about cool-flame vaporisation stopping — related but not the O2 level. "
                  "No single number appears in the documents; see the data check.",
    QUESTIONS[3]: "Answered with results: most SAME smoke particles were below 500 nm, and the ISS forward-scattering "
                  "detectors are most sensitive above ~1 µm — a detector mismatch. Earlier CSD flight: solid smoke "
                  "particles formed longer chains/larger aggregates in microgravity.",
    QUESTIONS[4]: "Answered via the closest proxy: ACME BRE (gas burner emulating solids) reports a common plastic could "
                  "steadily burn under NASA's proposed exploration atmospheres. No solid-material test exists at these "
                  "conditions (see processed/coverage.json gaps).",
}


def _data_check_flex() -> str:
    import pandas as pd

    t = pd.read_csv(PROCESSED_DIR / "tests.csv")
    f = t[t["investigation_id"] == "PSI-69"]
    rows = []
    for fuel, g in f.groupby("fuel"):
        done = g[g["outcome"] == "completion"]["o2_frac"]
        ext = g[g["outcome"] == "extinction"]["o2_frac"]
        rows.append(f"| {fuel} | {len(g)} | {done.min():.2f} | {ext.min():.2f}–{ext.max():.2f} |")
    return ("**Data check** (computed from `processed/tests.csv`, FLEX PSI-69, all pressures and diluents):\n\n"
            "| Fuel | Tests | Lowest O2 that still burned to completion | O2 range where flames extinguished |\n"
            "|---|---|---|---|\n" + "\n".join(rows) +
            "\n\nExtinction depends on droplet size, pressure and diluent as well as O2 — see `ai.model.predict_flex`.")


DATA_CHECK = {QUESTIONS[2]: _data_check_flex}


def _pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader

        from .ingest import _long_path
        return len(PdfReader(_long_path(path)).pages)
    except Exception:
        return None


def check_citation(c: dict) -> str:
    file = c["file"].split("#")[0]
    if file.startswith("processed/"):  # ETL output (e.g. the data cards' source): lives under the repo root
        return "ok (ETL table)" if (ROOT / file.split(" (")[0]).exists() else "missing ETL file"
    path = DATA_DIR / file
    if not path.exists():
        return "missing file"
    if c.get("page") and path.suffix.lower() == ".pdf":
        n = _pdf_pages(path)
        if n is None:
            return "file ok, page not checkable"
        return f"ok (p. {c['page']} of {n})" if 1 <= c["page"] <= n else f"BAD page {c['page']} > {n}"
    return "ok"


def render() -> str:
    invs = by_id()
    out = [
        "# Ask-the-data examples (no-LLM mode)",
        "",
        f"Generated {date.today().isoformat()} by `python -m ai.examples` from `ai.qa.ask()` **without an API key**: "
        "answers are verbatim sentences from the PSI documents, ranked by keyword relevance (BM25 + embeddings, "
        "domain query expansion, scoping by investigation name). Bold = question terms. "
        "Each question has a reviewer verdict and an automatic citation check.",
        "",
    ]
    for i, q in enumerate(QUESTIONS, start=1):
        r = ask(q, use_cache=False)
        out += [f"## {i}. {q}", "", r["answer"], "", "| # | PSI-ID | DOI | File | Page | Check |", "|---|---|---|---|---|---|"]
        for n, c in enumerate(r["citations"], start=1):
            doi = c.get("doi")
            doi_md = f"[{doi}](https://doi.org/{doi})" if doi else "— (Researcher's Guide)"
            pid = c["psi_id"]
            pid_md = f"[{pid}]({invs[pid]['landing_page']})" if pid in invs else pid
            out.append(f"| {n} | {pid_md} | {doi_md} | `{c['file']}` | {c.get('page') or '—'} | {check_citation(c)} |")
        out += ["", f"**Reviewer verdict:** {REVIEW[q]}", ""]
        if q in DATA_CHECK:
            out += [DATA_CHECK[q](), ""]
    return "\n".join(out)


def main():
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
