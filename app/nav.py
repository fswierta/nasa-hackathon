"""The page list, shared by main.py (navigation) and the home page (links).
Only pages whose file exists are registered, so pages can be added one issue at a time."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# (path relative to app/, title, icon, one-line description)
PAGE_SPECS = [
    ("views/home.py", "Home", "🔥", ""),
    ("views/overview.py", "Overview", "🗂️", "All 24 investigations, filterable."),
    ("views/detail.py", "Investigation detail", "🔬", "Objective, conditions, tests and sources."),
    ("views/ranker.py", "Scenario ranker", "🏆", "Which experiments matter for your habitat?"),
    ("views/condition_map.py", "Condition map", "🗺️", "Tested O₂ / pressure vs habitat atmospheres."),
    ("views/predict.py", "Predict extinction", "🧯", "Will a burning droplet go out? FLEX model."),
    ("views/ask.py", "Ask the data", "💬", "Questions answered with cited sources."),
    ("views/compare.py", "Compare", "⚖️", "Investigations side by side."),
    ("views/insights.py", "Insights", "💡", "Prevention, detection, suppression lessons."),
    ("views/about.py", "About the data", "ℹ️", "Sources, licence, data issues we found."),
]


def available() -> list[tuple[str, str, str, str]]:
    return [s for s in PAGE_SPECS if (APP_DIR / s[0]).exists()]
