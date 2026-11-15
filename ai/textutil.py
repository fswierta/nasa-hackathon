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


def dedupe_key(s: str) -> str:
    """Comparison key that treats 'ﬂame' and 'flame' (PDF ligatures) and spacing/punctuation as equal."""
    import unicodedata
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", s or "").lower())[:120]


# ai/ingest.py prefixes metadata chunks with "PSI-26 BASS: <title>\n<field>:\n" — a label we added,
# not PSI text, so it must never end up inside a quoted sentence
_META_HEADER = re.compile(r"^(?:PSI-\d+|REF)\b[^\n]*\n(?:objective|hypothesis|approach|researchImpacts|description):\s*")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(norm(_META_HEADER.sub("", text or "", count=1))) if s.strip()]


def is_clean(sentence: str, min_words: int = 8, max_words: int = 70, min_alpha: float = 0.65) -> bool:
    """Reject table rows, garbled extraction, headers, mojibake. `min_alpha` = minimum share of letters;
    lowered only for text we generate ourselves (data cards), where numbers are expected."""
    if "�" in sentence or "�" in sentence:
        return False
    words = sentence.split()
    if not min_words <= len(words) <= max_words:
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters / max(1, len(sentence)) < min_alpha:
        return False
    if sum(len(w) == 1 for w in words) > len(words) * 0.25:  # "f l a m e" style extraction noise
        return False
    return sentence[0].isupper() or sentence[0] in "(["


# numbered procedure steps ("... droplet size 3.") and imperative instructions are not results
_STEP = re.compile(r"\s\d{1,2}\.$|^(?:Select|Increase|Decrease|Continue|Repeat|Ignite|Record|Set|Turn)\b")


_SCORE_RULES = [
    # strong result statements
    (+3, re.compile(r"\bresults? (?:\w+ ){0,2}(?:show|showed|shown|showing|indicate|indicated|indicating|suggest|"
                    r"suggested|demonstrate|demonstrated|confirm|confirmed|reveal|revealed)", re.I)),
    (+3, re.compile(r"\b(?:was|were|been) (?:found|observed|shown|measured|demonstrated)\b|\bit was (?:found|observed|shown)", re.I)),
    (+2, re.compile(r"\b(?:experiments?|tests?|data|flights?|measurements?) (?:\w+ ){0,3}(?:showed|demonstrated|revealed|"
                    r"identified|found|confirmed|indicated)", re.I)),
    (+2, re.compile(r"\bwe (?:found|observed|showed|measured|determined)\b", re.I)),
    # quantitative / comparative content
    (+1, re.compile(r"\b(?:higher|lower|larger|smaller|faster|slower|greater|less|more) (?:\w+ ){0,3}(?:than|in microgravity|in normal gravity)", re.I)),
    (+1, re.compile(r"\d+(?:\.\d+)?\s?(?:%|percent\b|mm|cm/s|mm/s|kPa|atm|psi|bar|K\b|s\b|seconds|nm|µm)", re.I)),
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
           "why do does did than then there their these those into about can could would "
           # question verbs that carry no topic ("what O2 levels did FLEX use?")
           "use used using run ran get got have has had many much known done".split())


def query_terms(q: str) -> list[str]:
    """Content words of a question. Two-character terms are kept only when they hold a digit
    (chemistry: 'o2', 'n2'), so "What O2 levels…" still matches sentences about O2."""
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", q.lower())
            if t not in STOP and (len(t) > 2 or any(ch.isdigit() for ch in t))]


_CITATION = re.compile(r"\bpp\.|\bvol\.|symposium|proceedings|journal|\(\d{4}\)|, \d{4},|et al\.?|doi\b|"
                       r"\([A-Z][A-Za-z\-]+(?: (?:and|&) [A-Z][A-Za-z\-]+)?(?: et al\.?)?,? \d{4}[a-z]?\)|\[\d+\]|"
                       r"^[A-Z][A-Za-z\-]+, [“\"]")  # bibliography entry: 'Olson, “Concurrent flame growth…'
# someone else's earlier work, quoted in a PSI document's literature review
PRIOR_WORK = re.compile(r"\b(?:they|their|previous(?:ly)?|prior|earlier (?:work|studies|experiments))\b", re.I)
# aims and plans, not results ("will actually spread faster" is a statement, so bare "will" is not penalised)
_AIM = re.compile(r"\bhypothes\w+|\baims? to\b|\bobjectives?\b|\bgoals?\b|\bplanned\b|"
                  r"\bwill (?:be|study|investigate|measure|examine|determine|provide|test|use|allow|involve|include)\b", re.I)
_CAPTION = re.compile(r"^(?:figure|fig\.|table)\s*\d|\bfigure \d+:|\bfig\. ?\d+:", re.I)


def _hits(terms: set[str], words: set[str]) -> set[str]:
    return {t for t in terms if t in words or (len(t) > 5 and any(w.startswith(t[:5]) for w in words))}


def best_sentence_scored(text: str, query: str, extra: str = "", weights: dict[str, float] | None = None,
                         ignore: set[str] | None = None, trusted: bool = False) -> tuple[str | None, float]:
    """The clean sentence in `text` that best answers `query`.

    score = 2·Σw(question terms hit) + Σw(expansion terms hit) + result bonus (finding_score, clipped)
            − 4 if it states an aim / hypothesis / plan.
    `weights`: term -> 0..1 (rarity, e.g. IDF), so 'spread' outweighs 'flame'. `ignore`: terms that
    carry no content here (investigation names already used to scope the search). Citations,
    figure captions and numbered procedure steps are skipped. Returns (sentence, score)."""
    ignore = {t.lower() for t in (ignore or set())}
    terms = set(query_terms(query)) - ignore
    extra_terms = set(query_terms(extra)) - terms - ignore
    w = weights or {}
    m = re.search(r"\bhow many (\w+)", query, re.I)  # "how many tests" -> prefer "<number> tests"
    count_rx = re.compile(rf"\b\d[\d,]*\s+{re.escape(m.group(1).rstrip('s'))}s?\b", re.I) if m else None
    best, best_score = None, 0.0
    for s in sentences(text):
        if (not is_clean(s, min_words=6, max_words=80, min_alpha=TRUSTED_MIN_ALPHA if trusted else 0.65)
                or _CITATION.search(s) or _CAPTION.search(s)
                or _STEP.search(s)):
            continue
        words = set(re.findall(r"[a-z0-9][a-z0-9\-]+", s.lower()))
        q_hit, x_hit = _hits(terms, words), _hits(extra_terms, words)
        if not q_hit and not x_hit:
            continue
        score = (2 * sum(w.get(t, 1.0) for t in q_hit) + sum(w.get(t, 1.0) for t in x_hit)
                 + max(-2, min(3, finding_score(s))) - (4 if _AIM.search(s) else 0)
                 + (3 if count_rx and count_rx.search(s) else 0))
        if score > best_score:
            best, best_score = s, score
    return best, best_score


def best_sentence(text: str, query: str, extra: str = "", trusted: bool = False) -> str | None:
    return best_sentence_scored(text, query, extra, trusted=trusted)[0]


TRUSTED_MIN_ALPHA = 0.45  # data cards: our own sentences, numeric by design (`trusted=True`)


def is_trusted_source(file: str) -> bool:
    """Chunks generated by this project from the ETL tables (not PDF extraction)."""
    return (file or "").startswith("processed/")


def highlight(sentence: str, query: str) -> str:
    """Wrap query terms in **bold** (Markdown) for display."""
    terms = sorted(set(query_terms(query)), key=len, reverse=True)
    if not terms:
        return sentence
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\w*", re.I)
    return pat.sub(lambda m: f"**{m.group(0)}**", sentence)


# ---- numbers (shared by LLM profiles and LLM answers: no number beyond the cited passages, #30/#52)
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")


def numbers_in(text: str) -> set[float]:
    out = set()
    for tok in _NUM.findall(text or ""):
        try:
            out.add(float(tok.replace(",", ".")))
        except ValueError:
            pass
    return out


def unsupported_numbers(text: str, sources: list[str]) -> set[float]:
    """Numbers in `text` that occur in none of the `sources` (PSI-IDs like 'PSI-26' are ignored)."""
    text = re.sub(r"PSI-\d+", "", text or "")
    have = set()
    for s in sources:
        have |= numbers_in(s)
    return {n for n in numbers_in(text) if n not in have}
