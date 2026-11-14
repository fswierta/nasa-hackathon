"""Retrieval benchmark (#32): does the search return the right PSI source?

15 questions with the investigation(s) or document that holds the answer. 8 name the investigation
(the easy case: the query can be scoped), 7 do not (the search has to find it by meaning). For each
method we report hit@1 and hit@3 = the expected source is in the top 1 / top 3 results.

    py -m ai.benchmark        -> processed/ai/benchmark.json + a table on stdout

Written by main (taken over from pc2). The questions and expected sources were fixed before any
result was looked at; results are reported as they come out.
"""
from __future__ import annotations

import json
import re
import time

from .config import AI_OUT

OUT_FILE = AI_OUT / "benchmark.json"

# expected: PSI ids that hold the answer and/or a substring of the file name ("Guide" = Researcher's Guide)
QUESTIONS = [
    # --- investigation named in the question
    {"q": "Which materials were tested in Saffire-II?", "psi": ["PSI-99", "PSI-98", "PSI-100"], "answer": r"silicone|PMMA|SIBAL|Nomex", "named": True},
    {"q": "Which suppressant gases were used in the FLEX experiment?", "psi": ["PSI-69"], "answer": r"carbon dioxide|CO2|CO 2|helium", "named": True},
    {"q": "When were the SAME-R flight tests conducted?", "psi": ["PSI-101"], "answer": r"July 16|September 9", "named": True},
    {"q": "What sizes of PMMA rods were burned in BASS-II?", "psi": ["PSI-25"], "answer": r"pmma rods|sizes of", "named": True},
    {"q": "In which year was SPICE operated on the ISS?", "psi": ["PSI-107", "PSI-60"], "answer": r"2009", "named": True},
    {"q": "What does the ACME Burning Rate Emulator emulate?", "psi": ["PSI-20"], "answer": r"emulat", "named": True},
    {"q": "What did the DAFT experiment measure?", "psi": ["PSI-47"], "answer": r"dust|P-Trak|aerosol", "named": True},
    {"q": "What was the goal of the ACME E-FIELD Flames experiment?", "psi": ["PSI-22"], "answer": r"electric field", "named": True},
    # --- no investigation named: the search must find the source by meaning
    {"q": "What misconceptions about fires in spacecraft does NASA list?", "file": ["Guide"], "answer": r"misconception", "named": False},
    {"q": "How thick is the cotton-fiberglass fabric used in the large-scale fire tests?", "psi": ["PSI-98", "PSI-99", "PSI-100"], "answer": r"0\.33 ?mm|0\.37", "named": False},
    {"q": "How do smoke particles from overheated materials differ in microgravity?", "psi": ["PSI-102", "PSI-101", "PSI-115"], "file": ["Guide"], "answer": r"particle|particulate", "named": False},
    {"q": "What is the Goldilocks flammability zone for flame spread?", "psi": ["PSI-62", "PSI-25", "PSI-26"], "answer": r"Goldilocks", "named": False},
    {"q": "Can a cool flame keep burning after the visible hot flame of a droplet goes out?", "psi": ["PSI-39", "PSI-69", "PSI-68", "PSI-117", "PSI-142"], "file": ["Guide"], "answer": r"cool flame", "named": False},
    {"q": "What is the smoke point of a gas jet flame and how is it measured?", "psi": ["PSI-107", "PSI-60", "PSI-106"], "answer": r"smoke point", "named": False},
    {"q": "How does an electric field change the shape of a flame?", "psi": ["PSI-22"], "answer": r"electric field", "named": False},
]


def _hit(result: dict, question: dict, strict: bool = False) -> bool:
    """Source-level: the result comes from an expected investigation/document.
    Strict (answer-level): ...and the passage itself contains the answer term."""
    ids = {result.get("psi_id"), *(result.get("also_in") or [])}
    source = bool(ids & set(question.get("psi", []))) or any(
        s.lower() in (result.get("file") or "").lower() for s in question.get("file", []))
    if not strict:
        return source
    return source and re.search(question["answer"], result.get("text") or "", re.I) is not None


def _methods() -> dict:
    from . import retrieval, vectors

    return {
        "keyword (BM25)": lambda q, k: retrieval.keyword_search(q, k),
        "vector (MiniLM embeddings)": lambda q, k: vectors.search(q, k),
        "hybrid (as used by Ask)": lambda q, k: retrieval.search(q, k=k),
    }


def run(k: int = 3) -> dict:
    methods, rows, summary = _methods(), [], {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        ranks, strict = [], []
        for question in QUESTIONS:
            results = fn(question["q"], k)
            rank = next((i for i, r in enumerate(results[:k], 1) if _hit(r, question)), None)
            srank = next((i for i, r in enumerate(results[:k], 1) if _hit(r, question, strict=True)), None)
            ranks.append(rank); strict.append(srank)
            rows.append({"method": name, "question": question["q"], "named": question["named"], "rank_of_first_hit": rank,
                         "rank_of_first_answer_passage": srank,
                         "top1": {"psi_id": results[0].get("psi_id"), "file": (results[0].get("file") or "").split("/")[-1][:70]} if results else None})
        def share(sub, top, src=None):
            sel = [r for r, q in zip(src if src is not None else ranks, QUESTIONS) if sub is None or q["named"] == sub]
            return round(sum(1 for r in sel if r is not None and r <= top) / len(sel), 2)
        summary[name] = {"hit@1": share(None, 1), "hit@3": share(None, 3),
                         "hit@3_named": share(True, 3), "hit@3_unnamed": share(False, 3),
                         "answer@1": share(None, 1, strict), "answer@3": share(None, 3, strict),
                         "answer@3_unnamed": share(False, 3, strict),
                         "seconds_per_question": round((time.perf_counter() - t0) / len(QUESTIONS), 2)}
    return {"n_questions": len(QUESTIONS), "n_named": sum(q["named"] for q in QUESTIONS), "k": k,
            "definition": "hit@k = a passage from the investigation/document that holds the answer is in the top k; "
                          "answer@k = stricter: that passage also contains the answer term itself",
            "summary": summary, "rows": rows, "questions": QUESTIONS}


def main() -> None:
    from . import retrieval
    retrieval.warmup()
    data = run()
    OUT_FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_FILE.name}: {data['n_questions']} questions ({data['n_named']} name the investigation)")
    print(f"{'method':30s} hit@1  hit@3  answer@1  answer@3  answer@3 unnamed  s/question")
    for name, s in data["summary"].items():
        print(f"{name:30s} {s['hit@1']:<6} {s['hit@3']:<6} {s['answer@1']:<9} {s['answer@3']:<9} {s['answer@3_unnamed']:<17} {s['seconds_per_question']}")
    for r in data["rows"]:
        if r["method"].startswith("hybrid") and r["rank_of_first_answer_passage"] is None:
            print("  hybrid: no answer passage in top 3 for:", r["question"], "-> top1", r["top1"])


if __name__ == "__main__":
    main()
