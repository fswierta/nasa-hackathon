"""Layout proposals for the redesign (#48) — NOT the live app. Run from the repo root:

    py -m streamlit run app/proposals/main.py --server.port 8503

A = dashboard-first (numbers, map preview, gap banner). B = story-first (guided 3 steps, #42).
The team picks one; the winner then moves into app/views/ and app/ui.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="IGNIS · layout proposals", page_icon="🔥", layout="wide")
nav = st.navigation({
    "A · dashboard-first": [st.Page("home_a.py", title="Home (A)", icon="📊", url_path="home_a", default=True),
                            st.Page("ranker_a.py", title="Ranker (A)", icon="🏆", url_path="ranker_a")],
    "B · story-first": [st.Page("home_b.py", title="Home (B)", icon="🧭", url_path="home_b"),
                        st.Page("ranker_b.py", title="Ranker (B)", icon="🏆", url_path="ranker_b")],
})
with st.sidebar:
    st.caption("Layout proposals for issue #48. Same data and wording as the live app; only the layout differs.")
nav.run()
