"""Smoke tests: every page renders against the real processed/ data without an exception.

    python -m pytest -q app/tests
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.nav import APP_DIR, available  # noqa: E402

TIMEOUT = 120


def test_main_renders_home():
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("Flame in Freefall" in t.value for t in at.title)
    labels = [m.label for m in at.metric]
    assert {"Investigations", "Test records", "Documents indexed"} <= set(labels)
    assert at.metric[0].value == "24"


@pytest.mark.parametrize("path", [p for p, *_ in available() if p != "pages/home.py"])
def test_page_renders(path):
    at = AppTest.from_file(str(APP_DIR / path), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
