"""Habitat atmosphere presets for the scenario ranker.

Every preset's O2 / pressure is traced to a public source in `source` (shown in the UI) and `source_refs`
(structured, for citations) — #64 item 3: the Space Apps challenge statement, or the ACME BRE objective
(PSI-20) for the NASA-considered exploration atmosphere. Values marked in `assumptions` are not from a source
or PSI data and must be shown as such in the UI.
"""
from __future__ import annotations

PSI_TO_KPA = 6.894757

# Default cabin ventilation speed used when the user doesn't give one (assumption, not PSI data).
DEFAULT_FLOW_CM_S = 10.0

CHALLENGE = {"name": "NASA Space Apps 2026 challenge statement 'Flame in Freefall'",
             "file": "project-data-1-pc1/data/reference/challenge_page.txt", "psi_id": None, "doi": None}
# PSI-20 objective (also ACME BRE Final Report 2022, p. 5): "NASA-considered atmosphere for human space missions
# will be examined: 14.7 psia, 21 % oxygen; 10.2 psia, 26.5 %; and 8.2 psia, 34 % oxygen."
BRE_OBJECTIVE = {"name": "ACME BRE (PSI-20) objective: NASA-considered atmospheres for human space missions",
                 "file": "documents/PSI-20/PSI-20_Reports_Final_20220201.pdf", "page": 5, "psi_id": "PSI-20",
                 "doi": "10.60555/vb93-3y72"}

HABITATS: dict[str, dict] = {
    "ISS": {
        "name": "ISS",
        "o2_frac": 0.21,
        "pressure_kPa": 101.3,
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "PMMA", "cotton/fiberglass fabric", "silicone", "Kapton", "Teflon"],
        "pillar": None,
        "gravity": "micro",
        "assumptions": ["flow_cm_s", "materials"],
        "source": "Earth sea-level air: 14.7 psi (Space Apps challenge statement); 14.7 psia with 21 % O2 is one of "
                  "the NASA-considered mission atmospheres in the ACME BRE objective (PSI-20, Final Report 2022, p. 5)",
        "source_refs": [CHALLENGE, BRE_OBJECTIVE],
    },
    "Lunar base": {
        "name": "Lunar base",
        "o2_frac": 0.30,
        "pressure_kPa": round(10.2 * PSI_TO_KPA, 1),
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "PMMA", "cotton/fiberglass fabric", "silicone"],
        "pillar": None,
        "gravity": "lunar",
        "assumptions": ["flow_cm_s", "materials"],
        "source": "Space Apps challenge statement: the Moon Base O2 'will likely be 30% or greater, with a "
                  "corresponding pressure of 10.2 psi or less'",
        "source_refs": [CHALLENGE],
    },
    "Exploration atmosphere": {
        "name": "Exploration atmosphere",
        "o2_frac": 0.34,
        "pressure_kPa": round(8.2 * PSI_TO_KPA, 1),
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "PMMA", "cotton/fiberglass fabric", "silicone"],
        "pillar": None,
        "gravity": "micro",
        "assumptions": ["flow_cm_s", "materials"],
        "source": "NASA-considered atmosphere for human space missions, 8.2 psia with 34 % O2: ACME BRE objective "
                  "(PSI-20, Final Report 2022, p. 5)",
        "source_refs": [BRE_OBJECTIVE],
    },
    "Mars base": {
        "name": "Mars base",
        "o2_frac": 0.34,
        "pressure_kPa": round(8.2 * PSI_TO_KPA, 1),
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "PMMA", "cotton/fiberglass fabric", "silicone"],
        "pillar": None,
        "gravity": "mars",
        "assumptions": ["o2_frac", "pressure_kPa", "flow_cm_s", "materials"],
        "source": "Assumption: no Mars-base atmosphere is specified (challenge statement or PSI); uses the "
                  "exploration atmosphere, 34 % O2 at 8.2 psia (ACME BRE objective, PSI-20)",
        "source_refs": [BRE_OBJECTIVE],
    },
    "Pressurized rover": {
        "name": "Pressurized rover",
        "o2_frac": 0.34,
        "pressure_kPa": round(8.2 * PSI_TO_KPA, 1),
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "PMMA", "silicone"],
        "pillar": None,
        "gravity": "lunar",
        "assumptions": ["flow_cm_s", "materials"],
        "source": "Space Apps challenge statement: Moon Base and pressurized rovers at 30 % O2 or more and 10.2 psi or "
                  "less; the preset uses 34 % O2 at 8.2 psia inside that range, a NASA-considered atmosphere "
                  "(ACME BRE objective, PSI-20, Final Report 2022, p. 5)",
        "source_refs": [CHALLENGE, BRE_OBJECTIVE],
    },
    "Spacesuit": {
        "name": "Spacesuit",
        "o2_frac": 1.00,
        "pressure_kPa": round(4.3 * PSI_TO_KPA, 1),
        "flow_cm_s": DEFAULT_FLOW_CM_S,
        "materials": ["Nomex", "silicone", "Teflon"],
        "pillar": None,
        "gravity": "micro",
        "assumptions": ["flow_cm_s", "materials"],
        "source": "Space Apps challenge statement: 'Spacesuits are filled with 100% oxygen to only 4.3 psi'",
        "source_refs": [CHALLENGE],
    },
}
