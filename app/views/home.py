import streamlit as st

from app import data
from app.nav import APP_NAME, available

invs = data.investigations()
tests = data.tests()
docs = data.doc_stats()
cov = data.coverage()

st.title(f"🔥 {APP_NAME}")
st.markdown(
    "**Fire behaves differently in space.** Without gravity there is no buoyancy: hot gas does "
    "not rise, flames turn spherical and live on airflow from fans. NASA has flown dozens of "
    "combustion experiments to learn how materials ignite, spread fire, make smoke and go out "
    "in microgravity. This dashboard **summarizes, ranks and interprets** the "
    f"{len(invs)} combustion investigations in NASA's PSI archive so engineers can find the "
    "evidence that matters for a spacecraft, the ISS, a Moon or Mars base, or a pressurized rover."
)

c1, c2, c3 = st.columns(3)
c1.metric("Investigations", len(invs),
          help="Combustion Science investigations in NASA PSI (flight + ground studies)")
c2.metric("Test records", f"{len(tests):,}",
          help="Rows parsed from the PSI experimental tables (processed/tests.csv)")
c3.metric("Documents indexed", docs["n_documents"] or "—",
          help="Science documents and reports the AI searches and cites")

if cov and cov.get("gaps"):
    g = cov["gaps"][0]
    st.warning(f"**Biggest data gap:** {g['text']}  \n{g.get('evidence', '')}", icon="⚠️")

st.subheader("Explore")
shown = [s for s in available() if s[0] != "views/home.py"]
if shown:
    cols = st.columns(min(3, len(shown)))
    for n, (path, title, icon, desc) in enumerate(shown):
        with cols[n % len(cols)]:
            st.page_link(path, label=title, icon=icon)
            st.caption(desc)
else:
    st.caption("More pages are on the way.")

tiers = {t: sum(i["relevance_tier"] == t for i in invs) for t in "ABC"}
st.caption(
    f"Relevance tiers: A (core spacecraft fire safety) {tiers['A']} · B (related) {tiers['B']} · "
    f"C (other combustion) {tiers['C']}. Every investigation links to its PSI DOI."
)
