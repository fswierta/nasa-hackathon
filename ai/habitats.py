"""Habitat atmosphere presets for the scenario ranker (AGENT_BRIEF §2).

Values marked in `assumptions` are not from PSI data and must be shown as such in the UI.
"""
from __future__ import annotations

PSI_TO_KPA = 6.894757

# Default cabin ventilation speed used when the user doesn't give one (assumption, not PSI data).
DEFAULT_FLOW_CM_S = 10.0

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
        "source": "AGENT_BRIEF §2: ISS ~21% O2 @ 14.7 psi",
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
        "source": "AGENT_BRIEF §2: planned Moon base >=30% O2 @ <=10.2 psi",
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
        "source": "AGENT_BRIEF §2: NASA Exploration Atmosphere 34% O2 @ 8.2 psi",
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
        "source": "Assumed = Exploration Atmosphere (no Mars-base atmosphere given in the brief)",
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
        "source": "AGENT_BRIEF §2: pressurized rover >=30% O2 @ <=10.2 psi; Exploration Atmosphere 34%/8.2 psi",
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
        "source": "AGENT_BRIEF §2: spacesuit 100% O2 @ 4.3 psi",
    },
}
