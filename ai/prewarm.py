"""Pre-warm the answer cache with the demo questions and write ai/examples_llm.md (issue #26).

    python -m ai.prewarm            # asks each demo question once (cached answers are reused)
    python -m ai.prewarm --refresh  # re-ask even if cached

Cached answers live in processed/ai/qa_cache.json (committed), so the demo and Streamlit Cloud
answer these questions instantly and spend no free-tier quota on them.
"""
from __future__ import annotations

import argparse
import re
from datetime import date

from .config import ROOT
from .examples import QUESTIONS
from .llm import PROVIDER_LABEL, provider
from .qa import ask

OUT = ROOT / "ai" / "examples_llm.md"


def _check(r: dict) -> str:
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", r["answer"])})
    bad = [n for n in nums if n > len(r["citations"])]
    return "all [n] markers resolve" if not bad else f"unresolved markers: {bad}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    lines = [f"# Ask-the-data examples — generated answers ({PROVIDER_LABEL[provider()]})", "",
             f"Generated {date.today().isoformat()} by `python -m ai.prewarm`. Each answer was produced by the LLM "
             "from retrieved PSI passages only; `[n]` cites the n-th passage below. Compare with the extractive "
             "answers in [examples.md](examples.md).", ""]
    for i, q in enumerate(QUESTIONS, start=1):
        r = ask(q, use_cache=not args.refresh)
        tag = f"{r['mode']} · {r.get('model', '-')}" + (" · from cache" if r.get("cached") else "")
        print(f"[{i}] {tag}: {q}")
        lines += [f"## {i}. {q}", "", f"*{tag}* — {_check(r)}", "", r["answer"], "", "| # | PSI-ID | DOI | File | Page |",
                  "|---|---|---|---|---|"]
        for n, c in enumerate(r["citations"], start=1):
            doi = f"[{c['doi']}](https://doi.org/{c['doi']})" if c.get("doi") else "—"
            lines.append(f"| {n} | {c['psi_id']} | {doi} | `{c['file']}` | {c.get('page') or '—'} |")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
