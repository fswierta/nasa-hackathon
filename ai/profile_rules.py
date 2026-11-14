"""Rule-based (no LLM) investigation profiles, contract 3 in CLAUDE.md — issue #12.

Everything is either quoted verbatim from PSI text (with file + page) or computed:
  summary_plain     first 1–2 sentences of metadata `objective`               (quoted)
  summary_expert    objective + approach, first ~5 sentences                   (quoted)
  key_findings      3–5 verbatim result sentences from retrieved passages      (file + page each)
  measurements      keyword dictionary over the investigation's own chunks     (first file + page)
  fire_safety_pillar  rules over sub_research_area + keywords + title
  habitat_relevance   rank() score per habitat, normalised by that habitat's best score
  earth_applications  research_impacts sentences about Earth / terrestrial / industry (quoted)
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from .catalog import by_id
from .config import CHUNKS_FILE, DATA_DIR
from .retrieval import search
from .textutil import finding_score, first_sentences, result_sentences, sentences

MIN_FINDING_SCORE = 3       # see textutil._SCORE_RULES
FALLBACK_FINDING_SCORE = 2  # used only to top up to 3 findings

# (display name, type, regex). Order matters only for readability of the output.
SENSORS = [
    ("color video camera", "camera", r"colou?r (?:video )?cameras?"),
    ("infrared (IR) camera", "camera", r"\b(?:IR|infrared) (?:video )?cameras?"),
    ("intensified / UV camera", "camera", r"intensified (?:UV |ultraviolet )?(?:video )?cameras?|\bUV cameras?"),
    ("high-speed camera", "camera", r"high[- ]speed (?:video |digital )?cameras?"),
    ("camera / video records (unspecified)", "camera", r"\b(?:video )?cameras?\b|\braw videos?\b"),
    ("radiometer", "radiometer", r"radiometers?"),
    ("thermocouple", "thermocouple", r"thermocouples?"),
    ("thin-filament pyrometry (TFP)", "other", r"thin[- ]filament pyrometr\w+|\bTFP\b"),
    ("photomultiplier tube (PMT)", "other", r"photomultiplier|\bPMTs?\b"),
    ("pressure transducer / sensor", "pressure sensor", r"pressure (?:transducers?|sensors?|gauges?)"),
    ("oxygen sensor / analyzer", "gas analyzer", r"\b(?:O2|oxygen) (?:sensors?|analy[sz]ers?|monitors?)"),
    ("carbon dioxide sensor / analyzer", "gas analyzer", r"\b(?:CO2|carbon dioxide) (?:sensors?|analy[sz]ers?|monitors?)"),
    ("carbon monoxide sensor / analyzer", "gas analyzer", r"\b(?:CO|carbon monoxide) (?:sensors?|analy[sz]ers?|monitors?)"),
    ("gas chromatograph", "gas analyzer", r"gas chromatograph\w*"),
    ("solid-state gas / pressure sensors", "gas analyzer", r"solid[- ]state sensors?"),
    ("TSI P-Trak particle counter", "particle counter / smoke detector", r"P-?Trak"),
    ("TSI DustTrak aerosol monitor", "particle counter / smoke detector", r"Dust ?Trak"),
    ("particle counter", "particle counter / smoke detector", r"(?:condensation (?:nuclei|particle) )?particle counters?"),
    ("ionization smoke detector", "particle counter / smoke detector", r"ionization (?:smoke )?detectors?"),
    ("light-scattering / photoelectric smoke detector", "particle counter / smoke detector",
     r"(?:light[- ]scattering|photoelectric) (?:smoke )?detectors?"),
    ("thermal precipitator (TEM grids)", "particle counter / smoke detector", r"thermal precipitators?"),
    ("laser-induced fluorescence (PLIF)", "other", r"\bPLIF\b|planar laser[- ]induced"),
    ("flow meter / anemometer", "flow sensor", r"mass flow (?:controllers?|meters?)|anemometers?|flow ?meters?"),
]
_SENSOR_RE = [(n, t, re.compile(p, re.I if not p.startswith(r"\b(?:CO|") else 0)) for n, t, p in SENSORS]

PILLAR_RULES = {
    "suppression": re.compile(r"extinguish|suppress|\bCO2\b|carbon dioxide|helium|\bHe dilution|dilution", re.I),
    "detection": re.compile(r"smoke|aerosol|particle|dust|detector", re.I),
    "prevention": re.compile(r"flammab|flame spread|ignition|fire safety|material", re.I),
}
EARTH = re.compile(r"\bEarth\b|terrestrial|industr|engines?\b|submarine|airplane|aircraft|mines?\b|emission|pollut", re.I)

HABITAT_KEY_MAP = {  # contract-3 key -> ai.habitats preset
    "spacecraft": "Exploration atmosphere",  # future crewed vehicles: 34 % O2 / 56.5 kPa
    "iss": "ISS",
    "lunar_base": "Lunar base",
    "mars_base": "Mars base",
    "rover": "Pressurized rover",
}


@lru_cache(maxsize=1)
def _chunks_by_pid() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out.setdefault(c["psi_id"], []).append(c)
    return out


@lru_cache(maxsize=None)
def _meta(pid: str) -> dict:
    p = DATA_DIR / "metadata" / f"{pid}.json"
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


@lru_cache(maxsize=1)
def _habitat_scores() -> dict[str, dict[str, float]]:
    from .habitats import HABITATS
    from .ranking import rank

    out = {}
    for key, name in HABITAT_KEY_MAP.items():
        res = rank(HABITATS[name])
        top = max(r["score"] for r in res) or 1.0
        out[key] = {r["id"]: round(r["score"] / top, 3) for r in res}
    return out


def _src(c: dict, doi: str | None) -> dict:
    return {"psi_id": c["psi_id"], "doi": c.get("doi") or doi, "file": c["file"], "page": c.get("page")}


def _ref_sentences_about(chunk: dict, names: list[str]) -> list[str]:
    """Result sentences in a Researcher's Guide chunk that name the investigation, or follow
    within 3 sentences of a sentence that does ('To date, these experiments have shown...')."""
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])", re.I)
    out, last = [], -99
    for i, s in enumerate(sentences(chunk["text"])):
        if pat.search(s):
            last = i
        if i - last <= 3 and result_sentences(s):
            out.append(s)
    return out


def _names(pid: str) -> list[str]:
    acr = (by_id()[pid].get("acronym") or "").strip()
    cands = {acr, acr.split("/")[0].strip(), acr.split(" / ")[0].strip()}
    return sorted(n for n in cands if len(n) >= 3 and not n.startswith("("))


def key_findings(pid: str, doi: str | None, n_max: int = 5) -> list[dict]:
    """3–5 verbatim result sentences. Priority: the investigation's own documents, then the
    Researcher's Guide where it discusses this investigation, then metadata researchImpacts.
    Metadata objective / hypothesis / approach are aims and methods, never findings."""
    own = []
    for q in ("results conclusions found showed", "measured observed increased decreased extinguished limit"):
        own += search(f"{pid} {q}", k=12, psi_ids=[pid])
    names = _names(pid)
    meta_chunks = _chunks_by_pid().get(pid, [])
    tiers = [
        [(h, result_sentences(h["text"])) for h in own if h["file"].startswith("documents/")],
        [(c, _ref_sentences_about(c, names)) for c in _chunks_by_pid().get("REF", [])] if names else [],
        [(c, result_sentences(c["text"])) for c in meta_chunks if c["file"].endswith("#researchImpacts")],
        # ground studies often state their results in `approach`; only strong statements qualify (see below)
        [(c, result_sentences(c["text"])) for c in meta_chunks if c["file"].endswith("#approach")],
    ]
    # score every candidate; best first, own documents win ties over the Guide over metadata
    seen_text, cands = set(), []
    for tier_no, tier in enumerate(tiers):
        for order, (chunk, sents) in enumerate(tier):
            for s in sents:
                key = s.lower()[:80]
                if key in seen_text:
                    continue
                seen_text.add(key)
                score = finding_score(s)
                floor = MIN_FINDING_SCORE if tier_no == 3 else FALLBACK_FINDING_SCORE
                if score >= floor:
                    cands.append((-score, tier_no, order, s, chunk))
    cands.sort(key=lambda c: c[:3])
    strong = [c for c in cands if -c[0] >= MIN_FINDING_SCORE]
    picked = strong[:n_max]
    if len(picked) < 3:  # top up with weaker (still positive) result sentences
        picked += [c for c in cands if c not in picked][: 3 - len(picked)]
    return [{"text": s, "source": _src(chunk, doi if chunk["psi_id"] == pid else None), "score": -neg}
            for neg, _, _, s, chunk in picked]


def measurements(pid: str, doi: str | None) -> list[dict]:
    chunks = _chunks_by_pid().get(pid, [])
    found, specific_camera = [], False
    for name, typ, rx in _SENSOR_RE:
        if name.startswith("camera / video") and specific_camera:
            continue
        for c in chunks:
            m = rx.search(c["text"])
            if m:
                found.append({"name": name, "type": typ, "matched": m.group(0), "source": _src(c, doi)})
                specific_camera |= typ == "camera"
                break
    return found


def pillars(inv: dict) -> list[str]:
    blob = " ".join(str(inv.get(k) or "") for k in ("sub_research_area", "keywords", "title"))
    hit = [p for p, rx in PILLAR_RULES.items() if rx.search(blob)]
    return hit or ["modeling"]


def earth_applications(pid: str) -> list[str]:
    text = _meta(pid).get("researchImpacts") or ""
    # "Earth benefits" alone is a heading-like phrase, not an application
    return [s for s in sentences(text)
            if EARTH.search(re.sub(r"\bEarth benefits?\b", "", s, flags=re.I)) and "�" not in s][:4]


def build_profile(pid: str) -> dict:
    inv = by_id()[pid]
    meta = _meta(pid)
    doi = inv.get("doi")
    objective = meta.get("objective") or inv.get("objective") or ""
    approach = meta.get("approach") or ""
    expert = " ".join(sentences(objective)[:3] + sentences(approach)[:2])
    findings = key_findings(pid, doi)
    n_docs = len({c["file"] for c in _chunks_by_pid().get(pid, []) if c["file"].startswith("documents/")})
    note = None
    if len(findings) < 3:
        note = (f"Only {len(findings)} explicit result statement(s) found in the {n_docs} PSI document(s) available "
                f"for this investigation" + (" (metadata only)" if n_docs == 0 else "") +
                ". See the experiment table, publications and the PSI landing page for results.")
    return {
        "id": pid,
        "summary_plain": first_sentences(objective, 2) or None,
        "summary_expert": expert or None,
        "summary_source": {"psi_id": pid, "doi": doi, "file": f"metadata/{pid}.json", "fields": ["objective", "approach"],
                           "quoted": True},
        "key_findings": findings,
        "key_findings_note": note,
        "fire_safety_pillar": pillars(inv),
        "habitat_relevance": {k: v.get(pid, 0.0) for k, v in _habitat_scores().items()},
        "habitat_relevance_note": "rank() score for the habitat preset, divided by the best score for that habitat; "
                                  "spacecraft = Exploration atmosphere preset",
        "earth_applications": earth_applications(pid),
        "earth_applications_source": {"psi_id": pid, "doi": doi, "file": f"metadata/{pid}.json#researchImpacts", "quoted": True},
        "measurements": measurements(pid, doi),
        "doi": doi,
        "generated": {"by": "rules (no LLM)", "date": date.today().isoformat()},
    }
