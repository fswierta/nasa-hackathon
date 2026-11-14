"""Proposal A — dashboard-first Home: numbers, map preview and the gap banner above the fold."""
import streamlit as st

from app.nav import APP_NAME, APP_TAGLINE, CHALLENGE
from app.proposals.common import goto, data, gaps_for, label, map_preview, ranked, ui

invs = data.investigations()
cov = data.coverage() or {}

ui.page_header(APP_NAME, f"**{APP_TAGLINE}** · {CHALLENGE}", "🔥")

# Row 1: four headline numbers in bordered cards
cols = st.columns(4)
for col, (lab, key, fallback, hlp) in zip(cols, [
    ("Investigations", "investigations", len(invs), "Combustion Science investigations in NASA PSI"),
    ("Test records", "test_records", len(data.tests()), "Rows parsed from the PSI experimental tables"),
    ("Documents indexed", "documents_indexed", data.doc_stats()["n_documents"], "Searched and cited by Ask"),
    ("Data gaps found", "data_gaps", len(cov.get("gaps", [])), "Conditions with no PSI test (coverage.json)"),
]):
    with col.container(border=True):
        v = data.key_number(key, fallback)
        ui.metric_card(st, lab, f"{v:,}" if isinstance(v, int) else v, help=hlp)

# Row 2: the gap banner across the page
if cov.get("gaps"):
    g = cov["gaps"][0]
    ui.callout("alert", f"**Biggest data gap:** {g['text']}  \n{g.get('evidence', '')}", icon="🟥")

# Row 3: map preview | top evidence for the Moon base
left, right = st.columns([1.6, 1], gap="large")
with left.container(border=True):
    st.markdown("**Tested conditions vs habitat atmospheres**")
    st.plotly_chart(map_preview(), width="stretch", config={"displayModeBar": False})
    ui.provenance_label("computed", "Ambient O₂ and pressure from the PSI tables; red = no solid-material test near.")
    goto("Open the condition map", "🗺️")
with right.container(border=True):
    st.markdown("**Top evidence for a Lunar base**")
    for n, r in enumerate(ranked("Lunar base")[:5], 1):
        st.markdown(f"{n}. **{label(r['id'])}** · score {r['score']:.2f}")
    ui.provenance_label("computed", "Scenario ranker, default weights.")
    goto("Open the scenario ranker", "🏆")
    st.markdown("**Other gaps**")
    for g in (cov.get("gaps") or [])[1:4]:
        st.caption(f"• {g['text']}")
    st.caption(f"{len(gaps_for('Lunar base'))} gap(s) concern the Lunar base.")
