"""Claims guard (#37): three statements must never appear in anything we show or submit, however the
text was produced (rules, Gemini, a person):

  1. "0 g and 1 g results bound partial-gravity behaviour" - not safe; flame spread peaks at low flow speeds
     (Olson & Ferkul's Goldilocks zone, PSI-62), so lunar gravity may be worse than both.
  2. "A fire self-extinguishes when ventilation is shut off" - NASA's Researcher's Guide p. 35 lists it as a
     misconception.
  3. Burner O2 of inverse flames (up to 85 %) presented as a tested atmosphere - highest atmosphere O2 is 40 %.

A sentence is fine if it denies or qualifies the claim (not / may not / but / misconception ...).
Run from repo root:  py -m pytest etl/tests/test_claims.py -q
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAVITY = r"(partial[- ]gravity|partial[- ]g\b|lunar|moon|mars|martian|intermediate gravity|this case)"
RULES = {
    "0 g / 1 g 'bound' partial gravity": re.compile(rf"\bbound(s|ed|ing)?\b[^.]*{GRAVITY}|{GRAVITY}[^.]*\bbound(s|ed|ing)?\b", re.I),
    "fire self-extinguishes without ventilation": re.compile(
        r"self-extinguish\w*[^.]{0,100}(ventilation|fans?)[^.]{0,60}(shut|off|stop)|"
        r"(shut\w*|turn\w*|switch\w*) off (the )?(fans?|ventilation)[^.]{0,120}(extinguish|fire-control|puts? out|stops? the fire)", re.I),
    "burner O2 (85 %) as an atmosphere": re.compile(r"(highest|maximum|up to)[^.]{0,60}\b85 ?%[^.]{0,80}(atmosphere|cabin|ambient)", re.I),
}
# What makes a matching sentence acceptable - checked per rule, close to the claim (a stray "no" elsewhere
# in the sentence, as in "PSI has no partial-gravity data; ... only bound this case", must not excuse it).
EXCEPT = {
    "0 g / 1 g 'bound' partial gravity": re.compile(
        r"(not|n't|never|cannot)\W+(\w+\W+){0,3}bound|could bound\W[^.]*\b(but|not)\b|bound\w*[^.]*\b(is|are) not (safe|guaranteed)", re.I),
    "fire self-extinguishes without ventilation": re.compile(r"\b(not|n't|never|misconception|myth|wrong|false)\b", re.I),
    "burner O2 (85 %) as an atmosphere": re.compile(r"burner|inverse", re.I),
}


def _texts():
    """(source, text) for every generated/submitted text: JSON string values and markdown files."""
    def walk(obj, src):
        if isinstance(obj, str):
            yield src, obj
        elif isinstance(obj, dict):
            quoted_myth = obj.get("status") == "misconception"  # the claim is quoted in order to refute it
            for k, v in obj.items():
                if quoted_myth and k == "claim":
                    continue
                if k not in ("raw_row", "quote", "excerpt", "passage", "text_excerpt"):  # verbatim PSI text is not our claim
                    yield from walk(v, f"{src}:{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from walk(v, src)

    for path in [*(ROOT / "processed").rglob("*.json")]:
        if path.name in ("embeddings_meta.json",) or path.stat().st_size > 5_000_000:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        yield from walk(data, str(path.relative_to(ROOT)))
    for path in [ROOT / "README.md", *(ROOT / "submission").glob("*.md"), *(ROOT / "ai").glob("examples*.md")]:
        if path.exists():
            yield str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")


def violations(pairs):
    out = []
    for src, text in pairs:
        # a wrapped markdown line is still one sentence: split on sentence ends and blank lines only
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", re.sub(r"[ \t]*\n[ \t]+", " ", text)):
            sentence = " ".join(sentence.split())
            for name, rule in RULES.items():
                if not rule.search(sentence):
                    continue
                if EXCEPT[name].search(sentence):
                    continue
                out.append(f"[{name}] {src}: {sentence.strip()[:220]}")
    return out


def test_guard_catches_the_three_claims():
    bad = [("t", "PSI has no partial-gravity data; microgravity and 1 g results only bound this case."),
           ("t", "Shutting off fans is a real fire-control lever, the flame will extinguish."),
           ("t", "The highest oxygen tested was up to 85 % in a cabin atmosphere.")]
    assert len(violations(bad)) == 3
    ok = [("t", "0 g and 1 g results may not bound partial gravity."),
          ("t", "NASA lists 'a fire will self-extinguish if ventilation is shut off' as a misconception."),
          ("t", "Inverse flames blow up to 85 % O2 out of the burner into a fuel atmosphere.")]
    assert violations(ok) == []


def test_no_unsafe_claims_in_generated_or_submitted_text():
    found = violations(_texts())
    assert not found, "\n".join(found[:20])


def test_ranker_gap_text_is_safe():
    import sys
    sys.path.insert(0, str(ROOT))
    from ai.habitats import HABITATS
    from ai.ranking import rank
    gaps = [(f"rank({name})", g) for name in ("Lunar base", "Mars base") for r in rank(HABITATS[name]) for g in r["gaps"]]
    assert gaps and violations(gaps) == []
