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
from app.nav import available  # noqa: E402

st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")

# Streamlit Community Cloud: the API key is a Streamlit secret, not a .env file. Copy it into the
# environment, where ai/ looks for it. Locally there is usually no secrets.toml, which is fine.
try:
    for _key in ("ANTHROPIC_API_KEY", "FIF_MODEL"):
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
               "Not endorsed by NASA. AI-assisted summaries are labelled.")
nav.run()
