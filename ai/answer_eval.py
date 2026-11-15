"""Answer-quality evaluation (issue #52): the 15 retrieval-benchmark questions through ask().

    python -m ai.answer_eval            # cached answers are reused (no quota), new ones are generated + cached
    python -m ai.answer_eval --refresh  # re-ask everything
    python -m ai.answer_eval --all      # all 40 benchmark questions (default: the 15 of round 1, as in #52)

For each question it records whether the answer
  - cites an expected source       (a citation from one of the question's expected PSI-IDs / documents),
  - has resolving citations         (every [n] marker points at a listed citation, and there is at least one),
  - contains the answer term        (the benchmark's answer regex, now matched against the generated answer),
  - passes the claims guard         (etl/tests/test_claims.py: no overclaiming wording),
plus mode/model, how many sentences the number check and the misconception guard removed, and the date.
Writes processed/ai/answer_eval.json; `--markdown` prints the README table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import date

from .benchmark import QUESTIONS
from .config import AI_OUT, ROOT
from .qa import ask

OUT = AI_OUT / "answer_eval.json"
CHECKS = ("source_cited", "citations_resolve", "answer_term", "claims_ok")


def _claims_violations():
    """The team's claims guard, loaded from its test file (etl/tests is not a package)."""
    spec = importlib.util.spec_from_file_location("_claims", ROOT / "etl" / "tests" / "test_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.violations


def evaluate(q: dict, r: dict, violations) -> dict:
    cites = r.get("citations") or []
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", r.get("answer") or "")}
    source = any(c.get("psi_id") in q.get("psi", []) for c in cites) or any(
        s.lower() in (c.get("file") or "").lower() for c in cites for s in q.get("file", []))
    return {
        "q": q["q"], "mode": r.get("mode"), "model": r.get("model"), "date": r.get("date"),
        "cached": bool(r.get("cached")),
        "source_cited": source,
        "citations_resolve": bool(markers) and all(1 <= n <= len(cites) for n in markers),
        "answer_term": re.search(q["answer"], r.get("answer") or "", re.I) is not None,
        "claims_ok": not violations([(q["q"], r.get("answer") or "")]),
        "numbers_dropped": len(r.get("numbers_dropped") or []),
        "guard_removed": len(r.get("guard_removed") or []),
        "cited": [c.get("psi_id") for c in cites],
    }


ROUND1 = 15  # the benchmark's first 15 questions (#52); #68 added 25 more


def run(refresh: bool = False, all_questions: bool = False) -> dict:
    violations = _claims_violations()
    rows = []
    questions = QUESTIONS if all_questions else QUESTIONS[:ROUND1]
    for i, q in enumerate(questions, start=1):
        r = ask(q["q"], use_cache=not refresh)
        row = evaluate(q, r, violations)
        rows.append(row)
        flags = " ".join(("✓" if row[c] else "✗") + c for c in CHECKS)
        print(f"[{i:2}] {row['mode']}/{row['model']}: {flags} — {q['q']}")
    generated = [r for r in rows if r["mode"] not in (None, "no-llm")]
    summary = {c: round(sum(r[c] for r in generated) / max(1, len(generated)), 3) for c in CHECKS}
    out = {"generated": {"by": "python -m ai.answer_eval", "date": date.today().isoformat()},
           "n_questions": len(rows), "n_generated": len(generated),
           "models": sorted({r["model"] for r in generated if r["model"]}),
           "rates": summary, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {summary} over {len(generated)} generated answers")
    return out


def markdown(res: dict) -> str:
    r = res["rates"]
    lines = ["| answers | expected source cited | citations resolve | answer term present | claims guard passes |",
             "|---|---|---|---|---|",
             f"| {res['n_generated']}/{res['n_questions']} generated ({', '.join(res['models'])}) | "
             f"{r['source_cited']:.0%} | {r['citations_resolve']:.0%} | {r['answer_term']:.0%} | {r['claims_ok']:.0%} |"]
    misses = [f"- {x['q']} — " + ", ".join(c for c in CHECKS if not x[c]) for x in res["rows"]
              if x["mode"] != "no-llm" and not all(x[c] for c in CHECKS)]
    return "\n".join(lines + (["", "Misses:"] + misses if misses else []))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--all", action="store_true", help="all benchmark questions, not only round 1")
    ap.add_argument("--markdown", action="store_true", help="print the README table from the saved results")
    a = ap.parse_args(argv)
    if a.markdown:
        print(markdown(json.loads(OUT.read_text(encoding="utf-8"))))
        return
    print(markdown(run(a.refresh, a.all)))


if __name__ == "__main__":
    main()
