"""Cross-investigation fire-safety insights -> processed/ai/insights.json (issue #14, brief S3/R5/C1).

    python -m ai.insights

Three kinds of insight, all reproducible:
  1. "evidence"  — written by the AI coding agent (Claude) from retrieved PSI passages. Each citation
                   carries a verbatim `quote`; the build FAILS if a quote is not found in the cited
                   file/page of processed/ai/chunks.jsonl. So nothing un-sourced can be shipped.
  2. "data"      — numbers computed here from processed/tests.csv / ai.model (never typed by hand).
  3. "data_gap"  — taken from main's processed/coverage.json plus the FLEX model's valid range.

Output shape (contract for pc3):
{"generated": {...},
 "insights": [{"id", "pillar", "kind", "title", "text", "habitats": [...],
               "citations": [{"psi_id", "doi", "file", "page", "quote"?}]}],
 "earth_applications": [{"text", "citations": [...]}]}
"""
from __future__ import annotations

import json
import re
from datetime import date

from .catalog import by_id
from .config import AI_OUT, CHUNKS_FILE, PROCESSED_DIR

OUT = AI_OUT / "insights.json"
ALL_HABITATS = ["ISS", "Lunar base", "Exploration atmosphere", "Mars base", "Pressurized rover", "Spacesuit"]
EXPLORATION = ["Lunar base", "Exploration atmosphere", "Mars base", "Pressurized rover"]
GUIDE = "reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf"
BASS_FABRIC = "documents/PSI-26/PSI-26_Science Documents_FlammabilityAspectsofFabric_USNationalCombustion.2013.docx"
SAME_REPORT = "documents/PSI-102/PSI-102_Reports_SAME_Final Science Report.pdf"

# ---- 1. evidence insights: (file, page, verbatim quote) per citation --------------------------
EVIDENCE = [
    {
        "id": "prev-opposed-flow", "pillar": "prevention", "habitats": ALL_HABITATS,
        "title": "A material that cannot burn downward on Earth can burn steadily in microgravity",
        "text": "In BASS, a fabric that only sputters and goes out when burning downward in 1 g kept burning "
                "against the airflow in microgravity until the fuel was used up. Without buoyancy the flame is "
                "not blown off, so passing a 1 g test does not prove a material is safe in orbit.",
        "sources": [(BASS_FABRIC, None, "this fuel will support opposed-flow burning in microgravity"),
                    (BASS_FABRIC, None, "the normal-gravity, downward-burning flame is ignited, but sputters and then goes out")],
    },
    {
        "id": "prev-std6001", "pillar": "prevention", "habitats": ALL_HABITATS,
        "title": "NASA's 1 g material screening test may not be conservative for space",
        "text": "Spacecraft materials are selected with NASA-STD-6001 Test 1, an upward-burning test in 1 g. "
                "BASS-II was built to test whether ventilated materials burn as well or better in microgravity, "
                "and NASA's Researcher's Guide notes drop-tower results suggesting Test 1 might not be conservative.",
        "sources": [("metadata/PSI-25.json#hypothesis", None, "it may not be conservative for flammability in space"),
                    ("metadata/PSI-26.json#researchImpacts", None,
                     "segregates material based on 1-g behavior without consideration of low gravity effects"),
                    (GUIDE, 23, "recent drop-tower results suggest that the current Test 1 might not be conservative")],
    },
    {
        "id": "prev-ventilation", "pillar": "prevention", "habitats": ALL_HABITATS,
        "title": "Cabin ventilation speed decides whether a microgravity flame lives or dies",
        "text": "In microgravity the only oxygen supply is the forced airflow, and flames are very sensitive to it. "
                "A low-flow 'quenching' extinction limit exists that can only be observed in microgravity, because "
                "on Earth buoyancy always adds about 20 cm/s of flow. Shutting off fans is a real fire-control lever.",
        "sources": [(BASS_FABRIC, None, "The flames in microgravity are very sensitive to flow speed"),
                    (BASS_FABRIC, None, "The observation of the quenching extinction branch is only possible at microgravity"),
                    (BASS_FABRIC, None, "Normal gravity studies are limited to flow speeds above the minimum buoyant speed in the flame zone (20 cm/s for this system)")],
    },
    {
        "id": "prev-large-scale", "pillar": "prevention", "habitats": ["ISS", "Lunar base", "Pressurized rover"],
        "title": "Saffire measured how fast a large fire grows in a real spacecraft",
        "text": "Saffire I–III burned large fabric and plastic samples inside an uncrewed Cygnus cargo vehicle after it "
                "left the ISS, at 20–30 cm/s airflow, to find out how quickly a large-scale fire grows in low gravity "
                "and whether it settles to a steady spread rate.",
        "sources": [("documents/PSI-100/PSI-100_Science Documents_SAFFIRE-I and SAFFIRE-III Readme.docx", None,
                     "determine how rapidly a large-scale fire grows in low-gravity and whether it reaches a steady-state spread rate"),
                    ("metadata/PSI-100.json#approach", None,
                     "the experiments were conducted at a flow velocity of 30 cm/s compared to 20 cm/s during Saffire I")],
    },
    {
        "id": "det-particle-size", "pillar": "detection", "habitats": ALL_HABITATS,
        "title": "Smoke detectors for spacecraft can't be designed from 1 g smoke data",
        "text": "Without buoyant mixing, smoke stays near its source longer, and SAME was designed around the "
                "hypothesis that this grows larger particles than in 1 g. Its final report concludes that direct "
                "low-g particle data, not extrapolation from 1 g, is needed to design spacecraft smoke detectors.",
        "sources": [("metadata/PSI-102.json#hypothesis", None, "will lead to larger particles compared to those produced in 1g"),
                    (SAME_REPORT, 44, "as opposed to extrapolation from 1-g data, direct knowledge of low-g combustion "
                                      "particulate is needed for more confident design of smoke detectors for spacecraft")],
    },
    {
        "id": "det-detector-type", "pillar": "detection", "habitats": ALL_HABITATS,
        "title": "Detector principle matters: ionization saw every SAME smoke, light scattering missed some",
        "text": "In SAME, the ionization-type Space Shuttle detector gave strong signals in all tests, while "
                "light-scattering detection was less effective for some Teflon tests, whose smoke had few large particles. "
                "Detector choice should match the smoke of the materials actually on board.",
        "sources": [(SAME_REPORT, 57, "All of the tests conducted in this series of experiments showed strong signals on "
                                      "the Space Shuttle detector (ionization)"),
                    (SAME_REPORT, 57, "which was seen for some of the Teflon tests")],
    },
    {
        "id": "det-dust", "pillar": "detection", "habitats": ALL_HABITATS,
        "title": "In orbit, dust stays airborne — a false-alarm risk for smoke detectors",
        "text": "DAFT found more large dust particles on orbit than on the ground, because nothing makes them settle. "
                "Detectors in a weightless (or dusty lunar) cabin must tell smoke from dust; DAFT adapted a TSI P-Trak "
                "particle counter to microgravity to measure this.",
        "sources": [("documents/PSI-47/PSI-47_Reports_DAFT_30Day.PostFlightReport.pdf", 8,
                     "without gravity to drive settling, more of the larger particles from the dust sample should remain entrained"),
                    ("metadata/PSI-47.json#approach", None, "TSI P-Trak was modified for use in microgravity")],
    },
    {
        "id": "sup-loi", "pillar": "suppression", "habitats": ALL_HABITATS,
        "title": "FLEX mapped the oxygen level below which a flame can't survive, by pressure and suppressant",
        "text": "FLEX burned heptane and methanol droplets on the ISS to find the Limiting Oxygen Index (LOI) as a function "
                "of pressure and of the suppressant gas added (CO2, helium). This is the number an engineer needs to "
                "size an extinguishing system.",
        "sources": [("metadata/PSI-69.json#approach", None,
                     "A primary objective of the experiments is to determine the Limiting Oxygen Index (LOI) as a function "
                     "of ambient pressure and determine how the presence of a suppressant influences the LOI")],
    },
    {
        "id": "sup-nitrogen", "pillar": "suppression", "habitats": ALL_HABITATS,
        "title": "BASS tested nitrogen, the agent planned for future spacecraft extinguishers",
        "text": "BASS used nitrogen as a flame suppressant on burning solids in microgravity, giving a direct link to "
                "current and planned extinguishment techniques.",
        "sources": [("metadata/PSI-26.json#researchImpacts", None,
                     "Using nitrogen as a flame suppressant in microgravity provides a direct link to current and planned extinguishment techniques")],
    },
]

EARTH = [
    ("Fire safety in sealed spaces on Earth — mines, airplanes, submarines (Saffire).",
     [("metadata/PSI-98.json#researchImpacts", None, "including inside mines, airplanes, or submarines")]),
    ("Smoke detectors for extreme environments such as submarines and underwater laboratories (SAME, SAME-R, DAFT).",
     [("metadata/PSI-102.json#researchImpacts", None, "useful in other extreme environments on Earth, such as submarines or underwater lab"),
      ("metadata/PSI-47.json#researchImpacts", None, "such as submarines or underwater laboratories where accurate detection of smoke can save lives")]),
    ("Fire safety in confined and reduced-gravity settings such as submarines and high-altitude aircraft (FLEX).",
     [("metadata/PSI-69.json#researchImpacts", None, "such as submarines or high-altitude aircraft")]),
    ("More efficient, cleaner engines from cool-flame and ultra-lean combustion research (CFI, ACME CFI-G, FLEX).",
     [("metadata/PSI-39.json#researchImpacts", None, "Homogeneous Charge Compression Ignition (HCCI) engines"),
      ("metadata/PSI-159.json#researchImpacts", None, "These efficiencies may increase to 60% if future engines can take advantage of ultra-lean combustion")]),
    ("Lower pollutant emissions by controlling flames with electric fields (ACME E-FIELD Flames).",
     [("metadata/PSI-22.json#researchImpacts", None, "The use of electric fields to control flames can reduce the production of harmful pollutants")]),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _chunk_lookup():
    table: dict[tuple, list[dict]] = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            table.setdefault((c["file"], c["page"]), []).append(c)
    return table


def _cite(table, file, page, quote) -> dict:
    """Resolve a quote to its chunk; raise if the quote is not in the cited file/page."""
    q = _norm(quote)
    for c in table.get((file, page), []):
        if q in _norm(c["text"]):
            return {"psi_id": c["psi_id"], "doi": c["doi"], "file": file, "page": page, "quote": quote}
    raise ValueError(f"quote not found in {file} p{page}: {quote[:70]}…")


def _inv_cite(pid: str, file: str) -> dict:
    inv = by_id()[pid]
    return {"psi_id": pid, "doi": inv.get("doi"), "file": file, "page": None}


def data_insights() -> list[dict]:
    import pandas as pd

    from .model import model_card

    t = pd.read_csv(PROCESSED_DIR / "tests.csv")
    out = []
    flex = t[t["investigation_id"] == "PSI-69"]
    hep_done = flex[(flex["fuel"] == "Heptane") & (flex["outcome"] == "completion")]
    ext = flex[flex["outcome"] == "extinction"]
    if len(hep_done) and len(ext):
        low = hep_done.sort_values("o2_frac").iloc[0]
        out.append({
            "id": "data-flex-low-o2", "pillar": "suppression", "kind": "data", "habitats": ALL_HABITATS,
            "title": f"Heptane droplets burned to completion at only {low.o2_frac:.0%} O2",
            "text": (f"In the FLEX table, the lowest O2 at which a heptane droplet burned completely was "
                     f"{low.o2_frac:.0%} (at {low.pressure_kPa:.0f} kPa), well below normal air. Flame extinction was "
                     f"recorded from {ext.o2_frac.min():.0%} up to {ext.o2_frac.max():.0%} O2 across "
                     f"{len(ext)} tests, depending on fuel, droplet size and diluent. Lowering O2 to ~15–17% is not by "
                     f"itself enough to guarantee a liquid-fuel fire goes out."),
            "citations": [_inv_cite("PSI-69", "processed/tests.csv")],
        })
    card = model_card()["extinction_d"]
    per_point = -card["coefficients"]["o2_frac"] / 100
    out.append({
        "id": "data-flex-model", "pillar": "suppression", "kind": "data", "habitats": ALL_HABITATS,
        "title": "Less oxygen makes droplet flames go out sooner — about %.2f mm per O2 point" % per_point,
        "text": (f"A linear model fitted to {card['n_train']} FLEX tests (cross-validated R² {card['cv_r2']:.2f}) "
                 f"shows the flame goes out at a {per_point:.2f} mm larger droplet diameter for every 1 percentage "
                 f"point less O2, with fuel, pressure, CO2/He and initial size held fixed. Use `ai.model.predict_flex` "
                 f"within the tested ranges only."),
        "citations": [_inv_cite("PSI-69", "processed/tests.csv")],
    })
    sf = t[(t["investigation_id"] == "PSI-99") & t["spread_rate_mm_s"].notna()]
    if len(sf):
        out.append({
            "id": "data-saffire2-spread", "pillar": "prevention", "kind": "data", "habitats": ["ISS"],
            "title": "Saffire-II recorded spread rates of %.1f–%.1f mm/s" % (sf.spread_rate_mm_s.min(), sf.spread_rate_mm_s.max()),
            "text": (f"Only {len(sf)} of the Saffire-II table rows carry a flame spread rate "
                     f"({sf.spread_rate_mm_s.min():.1f}–{sf.spread_rate_mm_s.max():.1f} mm/s at ~22% O2, 20 cm/s flow; "
                     f"materials: {', '.join(sorted(sf.material.dropna().unique()))}). The full image-derived results are "
                     f"in the Saffire MergeResult spreadsheets on PSI."),
            "citations": [_inv_cite("PSI-99", "processed/tests.csv")],
        })
    return out


def gap_insights() -> list[dict]:
    from .model import predict_flex

    out = []
    cov_path = PROCESSED_DIR / "coverage.json"
    if cov_path.exists():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        for i, g in enumerate(cov.get("gaps", []), start=1):
            blob = " ".join(str(g.get(k, "")) for k in ("text", "evidence", "closest_proxy"))
            from .retrieval import mentioned_investigations  # also map names like "BASS, SAFFIRE" to IDs
            pids = list(dict.fromkeys(re.findall(r"PSI-\d+", blob) + mentioned_investigations(blob)))
            habitats = g.get("habitats") or ([g["habitat"]] if g.get("habitat") else EXPLORATION)
            text = g["evidence"] + (f" Closest proxy: {g['closest_proxy']}" if g.get("closest_proxy") else "")
            out.append({
                "id": f"gap-coverage-{i}", "pillar": "prevention", "kind": "data_gap", "habitats": habitats,
                "title": g["text"], "text": text,
                "citations": [_inv_cite(p, "processed/coverage.json") for p in pids if p in by_id()],
            })
    p = predict_flex(0.34, 56.5, "N2", 3.0, "heptane")
    lo = min(v["pressure_kPa"][0] for v in (p["valid_range"], predict_flex(0.21, 101.3, fuel="methanol")["valid_range"]))
    out.append({
        "id": "gap-flex-pressure", "pillar": "suppression", "kind": "data_gap", "habitats": EXPLORATION,
        "title": "No droplet-extinction data at exploration-atmosphere pressure",
        "text": (f"FLEX never went below about {lo:.0f} kPa, so its extinction results cannot be applied directly to "
                 f"the 56.5 kPa exploration atmosphere planned for Moon/Mars habitats and rovers; the FLEX model "
                 f"refuses to predict there ({p['reason'].rstrip('.')})"),
        "citations": [_inv_cite("PSI-69", "processed/tests.csv")],
    })
    return out


def build() -> dict:
    table = _chunk_lookup()
    insights = []
    for e in EVIDENCE:
        insights.append({
            "id": e["id"], "pillar": e["pillar"], "kind": "evidence", "title": e["title"], "text": e["text"],
            "habitats": e["habitats"], "citations": [_cite(table, *s) for s in e["sources"]],
        })
    insights += data_insights()
    insights += gap_insights()
    earth = [{"text": text, "citations": [_cite(table, *s) for s in srcs]} for text, srcs in EARTH]
    return {
        "generated": {"date": date.today().isoformat(),
                      "how": "evidence: written by an AI coding agent (Claude) from retrieved PSI passages, every "
                             "quote verified against processed/ai/chunks.jsonl; data/data_gap: computed from "
                             "processed/tests.csv, processed/coverage.json and ai/model.py"},
        "insights": insights,
        "earth_applications": earth,
    }


def main():
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"{len(result['insights'])} insights -> {OUT}")
    print(dict(Counter((i["pillar"], i["kind"]) for i in result["insights"])))
    print(f"{len(result['earth_applications'])} earth applications")


if __name__ == "__main__":
    main()
