"""Retrieval benchmark (#32): does the search return the right PSI source?

40 questions with the investigation(s) or document that holds the answer (15 in round 1, 25 added in #68 to
cover every investigation with documents). 20 name the investigation (the easy case: the query can be scoped),
20 do not (the search has to find it by meaning). For each
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
    {"q": "How thick is the cotton-fiberglass fabric used in the large-scale fire tests?", "psi": ["PSI-98", "PSI-99", "PSI-100"], "answer": r"0\.33\s?mm|0\.37", "named": False},  # \s: the PDF has a no-break space (#68)
    {"q": "How do smoke particles from overheated materials differ in microgravity?", "psi": ["PSI-102", "PSI-101", "PSI-115"], "file": ["Guide"], "answer": r"particle|particulate", "named": False},
    {"q": "What is the Goldilocks flammability zone for flame spread?", "psi": ["PSI-62", "PSI-25", "PSI-26"], "answer": r"Goldilocks", "named": False},
    {"q": "Can a cool flame keep burning after the visible hot flame of a droplet goes out?", "psi": ["PSI-39", "PSI-69", "PSI-68", "PSI-117", "PSI-142"], "file": ["Guide"], "answer": r"cool flame", "named": False},
    {"q": "What is the smoke point of a gas jet flame and how is it measured?", "psi": ["PSI-107", "PSI-60", "PSI-106"], "answer": r"smoke point", "named": False},
    {"q": "How does an electric field change the shape of a flame?", "psi": ["PSI-22"], "answer": r"electric field", "named": False},
    # --- round 2 (#68): 25 more, covering the investigations the first 15 did not ask about.
    # Fixed before the run; each answer term was checked to occur in a passage of the expected source.
    # named
    {"q": "What are the objectives of ACME s-Flame?", "psi": ["PSI-23"], "answer": r"soot-free|sooty|spherical", "named": True},
    {"q": "Which fuels were tested in SPICE?", "psi": ["PSI-107", "PSI-60"], "answer": r"methane|ethylene|propane|propylene", "named": True},
    {"q": "Which fuels does ACME CFI-G use for cool diffusion flames?", "psi": ["PSI-159"], "answer": r"propane|butane|pentane", "named": True},
    {"q": "What is the goal of FLEX-2 compared with FLEX?", "psi": ["PSI-68"], "answer": r"bi-?component|FLEX-1", "named": True},
    {"q": "What did Saffire-III investigate?", "psi": ["PSI-100", "PSI-98", "PSI-99"], "answer": r"flammability limit|large.scale fire|fire grows", "named": True},
    {"q": "What are the objectives of the BASS experiment?", "psi": ["PSI-26", "PSI-25"], "answer": r"NASA-STD-6001|flammability", "named": True},
    {"q": "Which flames did ACME CLD Flame study?", "psi": ["PSI-21"], "answer": r"coflow|methane|ethylene", "named": True},
    {"q": "What did the SLICE experiment investigate?", "psi": ["PSI-106"], "answer": r"lift-?off|structure|stabili", "named": True},
    {"q": "Which fuel and pressures were studied in the Cool Flames Investigation (CFI)?", "psi": ["PSI-39"], "answer": r"dodecane|atm", "named": True},
    {"q": "What soot limits did ACME Flame Design measure?", "psi": ["PSI-10"], "answer": r"soot.inception|soot", "named": True},
    {"q": "What was simulated in the SAME smoke DNS ground study?", "psi": ["PSI-115"], "answer": r"smoke|pyrolysis", "named": True},
    {"q": "What does the BASS flamelet generated manifold (FGM) model study?", "psi": ["PSI-62"], "answer": r"flamelet|concurrent", "named": True},
    # not named
    {"q": "Which particle counter condenses isopropanol around aerosol particles?", "psi": ["PSI-47"], "answer": r"isopropanol|P-Trak", "named": False},
    {"q": "When does a microgravity spherical diffusion flame extinguish?", "psi": ["PSI-10", "PSI-23", "PSI-159"], "answer": r"extinguish|extinction", "named": False},
    {"q": "Does soot formation require a region where the local C/O ratio and temperature exceed critical values?", "psi": ["PSI-10"], "answer": r"C/O", "named": False},
    {"q": "How large is the gap between soot formation in normal-gravity and zero-gravity flames?", "psi": ["PSI-60", "PSI-107"], "answer": r"soot", "named": False},
    {"q": "How fast does a large-scale fire grow in low gravity?", "psi": ["PSI-100", "PSI-98", "PSI-99"], "answer": r"fire|flame spread|grow", "named": False},
    {"q": "Can a steady cool diffusion flame exist with gaseous fuels?", "psi": ["PSI-159"], "answer": r"cool", "named": False},
    {"q": "How does helium dilution affect cool flames of dodecane droplets?", "psi": ["PSI-39", "PSI-142"], "answer": r"helium|cool flame", "named": False},
    {"q": "What share of the world's energy comes from combustion?", "psi": ["PSI-159", "PSI-10", "PSI-23", "PSI-21", "PSI-22", "PSI-20"], "answer": r"86 ?%|world", "named": False},
    {"q": "How do sub-millimeter fuel droplets burn compared with large droplets?", "psi": ["PSI-117"], "answer": r"sub-?millimeter|droplet", "named": False},
    {"q": "How is a ground flammability screening test related to how materials burn in microgravity?", "psi": ["PSI-26", "PSI-25", "PSI-100", "PSI-98", "PSI-99"], "answer": r"NASA-STD-6001|screening|normal gravity", "named": False},
    {"q": "How is a gas burner used to emulate the burning of a solid material?", "psi": ["PSI-20"], "answer": r"emulat", "named": False},
    {"q": "Why do long missions with larger vessels need more sensitive fire detection?", "psi": ["PSI-47", "PSI-101", "PSI-102", "PSI-115"], "answer": r"fire detection|detect", "named": False},
    {"q": "What is concurrent flame spread and why is it hard to model?", "psi": ["PSI-62", "PSI-25", "PSI-26"], "answer": r"concurrent", "named": False},
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
