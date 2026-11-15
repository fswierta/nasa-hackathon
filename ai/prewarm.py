"""Pre-warm the answer cache with the demo questions and write ai/examples_llm.md (issue #26).

    python -m ai.prewarm            # asks each demo question once (cached answers are reused)
    python -m ai.prewarm --refresh  # re-ask even if cached
    python -m ai.prewarm --fixed    # also every fixed question the app asks (#52): the 15 benchmark questions,
                                    # the Home tour question and the Response advisor's question per fuel

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

# Fixed questions the app sends to ask() (#52). Kept in sync with the pages by ai/tests/test_answers.py.
TOUR_QUESTION = "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?"   # app/views/home.py
RESPONSE_TEMPLATE = "How was a {what} fire suppressed or extinguished in microgravity?"          # app/views/response.py
DROPLET_FUELS = ["heptane", "methanol"]                                                          # app/fire.py
GAS_FUELS = ["ethane", "propane", "butane"]
SOLID_IDS = ("PSI-25", "PSI-26", "PSI-98", "PSI-99", "PSI-100", "PSI-101", "PSI-102")


def solid_materials() -> list[str]:
    """The Response advisor's material list (app/fire.py _solid_materials): single-material test rows."""
    import pandas as pd
    from .config import PROCESSED_DIR
    t = pd.read_csv(PROCESSED_DIR / "tests.csv", low_memory=False)
    t = t[(t["row_kind"] == "test") & t["investigation_id"].isin(SOLID_IDS) & t["fuel"].notna()]
    return list(t[~t["fuel"].str.contains(r"\+")]["fuel"].value_counts().index)


def response_questions() -> list[str]:
    whats = ([f"{f} droplet" for f in DROPLET_FUELS] + [f"{f} gas flame" for f in GAS_FUELS] + solid_materials())
    return [RESPONSE_TEMPLATE.format(what=w) for w in whats]


def fixed_questions() -> list[tuple[str, str]]:
    """(group, question) for every question the app asks by itself, plus the round-1 benchmark."""
    from .benchmark import QUESTIONS as BENCH
    out = [("demo", q) for q in QUESTIONS] + [("benchmark", q["q"]) for q in BENCH[:15]] + [("tour", TOUR_QUESTION)]
    out += [("response", q) for q in response_questions()]
    seen, uniq = set(), []
    for g, q in out:
        if q not in seen:
            seen.add(q)
            uniq.append((g, q))
    return uniq


def prewarm_fixed(refresh: bool = False, pause_s: float = 4.0) -> dict:
    """Ask every fixed question once; generated answers land in the committed cache. Returns counts per mode."""
    import time
    counts = {}
    for i, (group, q) in enumerate(fixed_questions(), start=1):
        r = ask(q, use_cache=not refresh)
        mode = "cached" if r.get("cached") else r.get("mode")
        counts[mode] = counts.get(mode, 0) + 1
        print(f"[{i:2}] {group:9s} {mode}/{r.get('model', '-')}: {q}", flush=True)
        if not r.get("cached") and r.get("mode") not in (None, "no-llm"):
            time.sleep(pause_s)  # free-tier requests-per-minute
    return counts


def _check(r: dict) -> str:
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", r["answer"])})
    bad = [n for n in nums if n > len(r["citations"])]
    return "all [n] markers resolve" if not bad else f"unresolved markers: {bad}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="also pre-warm every fixed question the app asks (#52)")
    args = ap.parse_args(argv)
    if args.fixed:
        print(prewarm_fixed(args.refresh))
        args.refresh = False  # the demo questions were just asked; the report below reads them from the cache
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
