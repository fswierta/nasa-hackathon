"""Sentence-level helpers for the no-LLM pipeline: split, clean, score, quote verbatim."""
from __future__ import annotations

import re

RESULT_WORDS = re.compile(
    r"\b(found|find(?:ing)?s?|show(?:s|ed|n)|demonstrat\w+|observ\w+|measur\w+|increas\w+|decreas\w+|"
    r"reduc\w+|extinguish\w*|limit(?:s|ing)?|higher|lower|larger|smaller|faster|slower|"
    r"result(?:s|ed)?|conclu\w+|indicat\w+|revealed|confirm\w+)\b", re.I)
# sentences about plans/requirements/hypotheses are not results
NOT_RESULT = re.compile(r"\b(will|shall|should|hypothes\w+|propos\w+|objective|goal|plan(?:ned)?|"
                        r"requirement|et al\.|proceedings|journal|conference|figure \d|fig\. ?\d|table \d)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(norm(text)) if s.strip()]


def is_clean(sentence: str, min_words: int = 8, max_words: int = 70) -> bool:
    """Reject table rows, garbled extraction, headers, mojibake."""
    if "�" in sentence or "�" in sentence:
        return False
    words = sentence.split()
    if not min_words <= len(words) <= max_words:
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters / max(1, len(sentence)) < 0.65:
        return False
    if sum(len(w) == 1 for w in words) > len(words) * 0.25:  # "f l a m e" style extraction noise
        return False
    return sentence[0].isupper() or sentence[0] in "(["


# numbered procedure steps ("... droplet size 3.") and imperative instructions are not results
_STEP = re.compile(r"\s\d{1,2}\.$|^(?:Select|Increase|Decrease|Continue|Repeat|Ignite|Record|Set|Turn)\b")


_SCORE_RULES = [
    # strong result statements
    (+3, re.compile(r"\bresults? (?:\w+ ){0,2}(?:show|showed|shown|indicate|indicated|suggest|suggested|demonstrate|"
                    r"demonstrated|confirm|confirmed|reveal|revealed)", re.I)),
    (+3, re.compile(r"\b(?:was|were|been) (?:found|observed|shown|measured|demonstrated)\b|\bit was (?:found|observed|shown)", re.I)),
    (+2, re.compile(r"\b(?:experiments?|tests?|data|flights?|measurements?) (?:\w+ ){0,3}(?:showed|demonstrated|revealed|"
                    r"identified|found|confirmed|indicated)", re.I)),
    (+2, re.compile(r"\bwe (?:found|observed|showed|measured|determined)\b", re.I)),
    # quantitative / comparative content
    (+1, re.compile(r"\b(?:higher|lower|larger|smaller|faster|slower|greater|less|more) (?:\w+ ){0,3}(?:than|in microgravity|in normal gravity)", re.I)),
    (+1, re.compile(r"\d+(?:\.\d+)?\s?(?:%|mm|cm/s|mm/s|kPa|atm|psi|bar|K\b|s\b|seconds|nm|µm)", re.I)),
    (+1, re.compile(r"extinguish|extinction|flammability limit|limiting oxygen|\bLOI\b|spread rate|burning rate|soot|"
                    r"particle size|cool flame", re.I)),
    # not this investigation's results
    (-3, re.compile(r"\b(?:they|their|previous(?:ly)?|prior|earlier|literature|et al)\b", re.I)),
    (-3, re.compile(r"\b(?:is|are) shown\b|\bon the (?:left|right)\b|\bfigure\b|\bshown in\b|\bas shown\b", re.I)),
    (-2, re.compile(r"\b(?:designed|manufactured|assembled|built|consists? of|connected|mounted|installed)\b", re.I)),
    # data-portal navigation ("records are found on the Data tab")
    (-4, re.compile(r"\b(?:tab|records?|folders?|links?|zip|\.txt|download\w*|repository)\b", re.I)),
]


def finding_score(sentence: str) -> int:
    return sum(w for w, rx in _SCORE_RULES if rx.search(sentence))


def result_sentences(text: str) -> list[str]:
    return [s for s in sentences(text)
            if is_clean(s) and RESULT_WORDS.search(s) and not NOT_RESULT.search(s) and not _STEP.search(s)]


def first_sentences(text: str, n: int) -> str:
    return " ".join(sentences(text)[:n])


STOP = set("the a an of and or in on for to is are was were be by with at from as that this it its what which how "
           "why do does did than then there their these those into about can could would".split())


def query_terms(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", q.lower()) if t not in STOP and len(t) > 2]


def best_sentence(text: str, query: str) -> str | None:
    """The clean sentence in `text` sharing the most query terms (ties -> earlier sentence)."""
    terms = set(query_terms(query))
    best, best_score = None, 0
    for s in sentences(text):
        if not is_clean(s, min_words=6, max_words=80):
            continue
        words = set(re.findall(r"[a-z0-9][a-z0-9\-]+", s.lower()))
        score = sum(1 for t in terms if t in words or any(w.startswith(t[:5]) for w in words if len(t) > 5))
        if score > best_score:
            best, best_score = s, score
    return best


def highlight(sentence: str, query: str) -> str:
    """Wrap query terms in **bold** (Markdown) for display."""
    terms = sorted(set(query_terms(query)), key=len, reverse=True)
    if not terms:
        return sentence
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\w*", re.I)
    return pat.sub(lambda m: f"**{m.group(0)}**", sentence)
