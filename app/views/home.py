import streamlit as st

from app import data, ui
from app.nav import APP_NAME, APP_TAGLINE, CHALLENGE, grouped

invs = data.investigations()
tests = data.tests()
docs = data.doc_stats()
cov = data.coverage()

ui.page_header(APP_NAME, icon="🔥")
st.caption(f"**{APP_TAGLINE}** · {CHALLENGE}")
st.markdown(
    "**Fire behaves differently in space.** Without gravity there is no buoyancy: hot gas does "
    "not rise, flames turn spherical and live on airflow from fans. NASA has flown dozens of "
    "combustion experiments to learn how materials ignite, spread fire, make smoke and go out "
    "in microgravity. This dashboard **summarizes, ranks and interprets** the "
    f"{len(invs)} combustion investigations in NASA's PSI archive so engineers can find the "
    "evidence that matters for a spacecraft, the ISS, a Moon or Mars base, or a pressurized rover."
)

# ---------------------------------------------------------------- guided tour (#42)
# Same story as submission/video_script.md: rank -> see the gap -> ask. Each button preloads the page
# through session state (works on every Streamlit version); the pages also accept URL parameters.
TOUR_QUESTION = "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?"
TOUR = [
    ("1", "Rank the evidence for a Moon base", "🏆",
     "Pick a habitat and IGNIS ranks the 24 investigations by how well their test conditions match it, "
     "with the reason behind every score.",
     "views/ranker.py", {"preset": "Lunar base"}),
    ("2", "See what was never tested", "🗺️",
     "The planned Moon-base atmosphere has no solid-material test in PSI, yet fabrics and plastics are "
     "the main fire load in a habitat.",
     "views/condition_map.py", {"gap_hl": True}),
    ("3", "Ask NASA's documents", "💬",
     "Ask in plain words; every answer cites the PSI investigation, DOI, file and page.",
     "views/ask.py", {"pending": TOUR_QUESTION}),
]
st.subheader("Start here: a 3-step tour")
for col, (n, title, icon, why, page, state) in zip(st.columns(3), TOUR):
    with col.container(border=True):
        st.markdown(f"**{n} · {title}**")
        st.caption(why)
        if st.button(f"{icon} Step {n}", key=f"tour_{n}", type="primary" if n == "1" else "secondary",
                     width="stretch"):
            st.session_state.update(state)
            st.switch_page(page)

c1, c2, c3 = st.columns(3)
ui.metric_card(c1, "Investigations", data.key_number("investigations", len(invs)),
                   help="Combustion Science investigations in NASA PSI (flight + ground studies)")
ui.metric_card(c2, "Test records", f"{data.key_number('test_records', len(tests)):,}",
                   help="Rows parsed from the PSI experimental tables (processed/tests.csv)")
ui.metric_card(c3, "Documents indexed", data.key_number("documents_indexed", docs["n_documents"]) or "—",
                   help="Science documents and reports the AI searches and cites")

if cov and cov.get("gaps"):
    g = cov["gaps"][0]
    ui.callout("gap", f"**Biggest data gap:** {g['text']}  \n{g.get('evidence', '')}", icon="⚠️")

# Same groups as the sidebar (app/nav.py SECTIONS).
for section, specs in grouped().items():
    specs = [sp for sp in specs if sp[0] != "views/home.py"]
    if not specs:
        continue
    st.subheader(section)
    cols = st.columns(3)
    for n, (path, title, icon, desc) in enumerate(specs):
        with cols[n % 3]:
            st.page_link(path, label=title, icon=icon)
            st.caption(desc)

tiers = {t: sum(i["relevance_tier"] == t for i in invs) for t in "ABC"}
st.caption(
    f"Relevance tiers: A (core spacecraft fire safety) {tiers['A']} · B (related) {tiers['B']} · "
    f"C (other combustion) {tiers['C']}. Every investigation links to its PSI DOI."
)
