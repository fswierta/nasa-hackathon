"""Guards for generated (LLM) text: remove sentences that state a known spacecraft-fire misconception (#37).

NASA's ISS Combustion Researcher's Guide (2015, p. 35; in the repo as psi_id REF and under PSI-20/PSI-22)
lists "several critical misconceptions about potential fires in spacecraft":
  1. A steady fire cannot exist in a quiescent microgravity environment.
  2. A fire will self-extinguish if ventilation is shut off.
  3. Because the reduced-gravity environment is inherently safer, terrestrial fire standards and tests
     represent a very conservative approach.
An earlier LLM insight said "Shutting off fans is a real fire-control lever" — exactly misconception 2.

    from ai.guards import remove_myths
    clean, removed = remove_myths(text)   # removed = [{"sentence", "myth"}]

A sentence that presents the claim *as* a misconception ("…is a misconception", "it is not true that…")
is kept. Applied to every generated text: LLM profiles, LLM answers, briefings.
"""
from __future__ import annotations

import re

from .textutil import sentences

SOURCE = {"psi_id": "REF", "file": "reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf", "page": 35}

_VENT = r"(?:ventilation|fans?|air ?flow|air circulation|forced flow)"
_OFF = r"(?:shut(?:ting)?(?: it)? (?:off|down)|turn(?:ing)?(?: it)? off|stop(?:ping)?|cut(?:ting)? off|switch(?:ing)?(?: it)? off|remov(?:e|ing))"
_OUT = r"(?:self-extinguish\w*|extinguish\w*|put\w* (?:it |the fire |the flame )?out|go(?:es)? out|stop\w* the (?:fire|flame)|" \
       r"starv\w*|(?:fire|flame)[- ]control lever|control (?:the )?fire|suppress\w* (?:the )?(?:fire|flame))"

MYTHS = [
    ("ventilation-off extinguishes",
     re.compile(rf"\b{_OFF}\s+(?:the\s+)?(?:cabin\s+)?{_VENT}\b[^.]{{0,120}}\b{_OUT}", re.I)),
    ("ventilation-off extinguishes",
     re.compile(rf"\b{_OUT}\b[^.]{{0,80}}\b(?:if|when|once|after)\b[^.]{{0,40}}\b{_VENT}\b[^.]{{0,30}}\b(?:is|are)?\s*"
                rf"(?:shut|turned|switched|cut)\s+(?:off|down)", re.I)),
    ("no steady fire without flow",
     re.compile(r"\b(?:steady|sustained)\s+(?:fire|flame)s?\s+(?:can ?not|cannot|can't|will not|won't)\s+(?:exist|survive|burn)"
                r"[^.]{0,60}\b(?:quiescent|still air|no (?:air ?)?flow|without (?:air ?)?flow|microgravity)", re.I)),
    ("microgravity is inherently safer",
     re.compile(r"\b(?:reduced|micro|low)[- ]?gravity\b[^.]{0,60}\b(?:inherently|always|intrinsically)\s+safer\b|"
                r"\b(?:terrestrial|1[- ]?g|normal[- ]gravity)\s+(?:fire\s+)?(?:standards?|tests?)[^.]{0,60}\b"
                r"(?:very |overly |always )?conservative\b", re.I)),
]
_FRAMED = re.compile(r"\bmisconception|\bmyth\b|\bnot (?:true|the case|correct)\b|\bincorrect(?:ly)?\b|\bwrong(?:ly)?\b|"
                     r"\bfalse\b|\bcontrary to\b|\bmistaken(?:ly)?\b|\bdoes not (?:hold|apply)\b|"
                     # sentences reporting the opposite (the flame survives) are findings, not the myth — e.g. BASS:
                     # "…the local jet entrains air and sustains the flame even when the ambient airflow is turned off"
                     r"\bsustain\w*|\bpersist\w*|\bkeeps? burning|\bcontinue[sd]? (?:to )?burn|\bstill burn|"
                     r"\bnot (?:adequate|sufficient|enough)|\binadequate\b|\bfails? to (?:extinguish|put)", re.I)


def myth_in(sentence: str) -> str | None:
    """Name of the misconception the sentence states as fact, or None."""
    if _FRAMED.search(sentence):
        return None
    for name, rx in MYTHS:
        if rx.search(sentence):
            return name
    return None


def remove_myths(text: str) -> tuple[str, list[dict]]:
    """Drop sentences that state a misconception as fact; keep everything else verbatim."""
    if not text:
        return text, []
    kept, removed = [], []
    for s in sentences(text):
        name = myth_in(s)
        if name:
            removed.append({"sentence": s, "myth": name, "source": SOURCE})
        else:
            kept.append(s)
    if not removed:
        return text, []
    return " ".join(kept), removed
