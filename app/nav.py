"""The page list, shared by main.py (navigation) and the home page (links).
Only pages whose file exists are registered, so pages can be added one issue at a time."""
from pathlib import Path

from app.ui import ICONS

APP_DIR = Path(__file__).resolve().parent

# Project name (team decision, #27; same as submission/summary.md and the deck). Browser tab, home, sidebar.
APP_NAME = "IGNIS"
APP_TAGLINE = "Intelligent Guidance for Navigating Ignition in Space"
CHALLENGE = "NASA Space Apps 2026 · Flame in Freefall challenge"

# (path relative to app/, title, Material icon from app/ui.py, one-line description)
PAGE_SPECS = [
    ("views/home.py", "Home", ICONS["home"], ""),
    ("views/overview.py", "Overview", ICONS["overview"], "All 24 investigations, filterable."),
    ("views/detail.py", "Investigation detail", ICONS["detail"], "Objective, conditions, tests and sources."),
    ("views/explorer.py", "Data explorer", ICONS["explorer"], "Every test row: filter, check corrections, download."),
    ("views/ranker.py", "Scenario ranker", ICONS["ranker"], "Which experiments matter for your habitat?"),
    ("views/scenario.py", "Fire scenario", ICONS["scenario"], "What would a fire do in this atmosphere?"),
    ("views/response.py", "Suppression evidence", ICONS["response"], "What NASA tests say about flames going out."),
    ("views/condition_map.py", "Condition map", ICONS["condition_map"], "Tested O₂ / pressure vs habitat atmospheres."),
    ("views/models.py", "Models", ICONS["models"], "Every model: score vs baseline, limits, a playground."),
    ("views/predict.py", "Predict extinction", ICONS["predict"], "Will a burning droplet go out? FLEX model."),
    ("views/ml_lab.py", "ML Lab", ICONS["ml_lab"], "Explore the data, engineer features, train and test models."),
    ("views/ask.py", "Ask the data", ICONS["ask"], "Questions answered with cited sources."),
    ("views/compare.py", "Compare", ICONS["compare"], "Investigations side by side."),
    ("views/insights.py", "Insights", ICONS["insights"], "Prevention, detection, suppression lessons."),
    ("views/about.py", "About the data", ICONS["about"], "Sources, licence, data issues we found."),
]


# Sidebar groups (#48). "" = no header (Home). A page missing here lands in "More".
SECTIONS = {
    "": ["views/home.py"],
    "Explore": ["views/overview.py", "views/detail.py", "views/explorer.py"],
    "Decide": ["views/ranker.py", "views/response.py", "views/condition_map.py", "views/compare.py"],
    # #58 part 4: every page whose numbers come from a model sits here, the hub first.
    "AI & ML": ["views/models.py", "views/scenario.py", "views/predict.py", "views/ml_lab.py", "views/ask.py"],
    "Learn": ["views/insights.py", "views/about.py"],
}


def available() -> list[tuple[str, str, str, str]]:
    return [s for s in PAGE_SPECS if (APP_DIR / s[0]).exists()]


def grouped() -> dict[str, list[tuple[str, str, str, str]]]:
    """Available pages by sidebar section, in SECTIONS order; empty sections are left out."""
    specs = {s[0]: s for s in available()}
    out = {sec: [specs.pop(p) for p in paths if p in specs] for sec, paths in SECTIONS.items()}
    if specs:
        out["More"] = list(specs.values())
    return {sec: pages for sec, pages in out.items() if pages}
