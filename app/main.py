"""Flame in Freefall — Streamlit entry point.  Run from the repo root:

    streamlit run app/main.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.data import warmup  # noqa: E402
from app.nav import APP_NAME, available  # noqa: E402

st.set_page_config(page_title=APP_NAME, page_icon="🔥", layout="wide")

# Streamlit Community Cloud: API keys are Streamlit secrets, not a .env file. Copy them into the
# environment, where ai/ looks for them (GEMINI_API_KEY = free tier; FIF_LLM picks the provider).
# Locally there is usually no secrets.toml, which is fine.
try:
    for _key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "FIF_LLM", "FIF_MODEL"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

# Pages are listed in app/nav.py; add new ones there.
PAGES = [st.Page(path, title=title, icon=icon, default=(path == "views/home.py"),
                 url_path=None if path == "views/home.py" else path.split("/")[-1][:-3])
         for path, title, icon, _ in available()]

nav = st.navigation(PAGES)
warmup()
with st.sidebar:
    st.caption("Data: NASA Physical Sciences Informatics (PSI), CC0. "
               "Not endorsed by NASA. Every answer and summary says whether it is quoted, computed or AI-written.")
nav.run()
