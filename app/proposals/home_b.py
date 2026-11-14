"""Proposal B — story-first Home: one question, then the 3 guided steps of the demo tour (#42)."""
import streamlit as st

from app.nav import APP_NAME, APP_TAGLINE
from app.proposals.common import goto, HABITATS, data, gaps_for, label, ranked, ui

ui.page_header(APP_NAME, f"**{APP_TAGLINE}**", "🔥")
st.markdown("### A fire on a Moon base: what does NASA's evidence say?")
st.markdown("Fire behaves differently without gravity. Follow three steps through "
            f"{data.key_number('investigations', 24)} NASA combustion investigations.")

name = st.segmented_control("Your habitat", [n for n in HABITATS if n != "Spacesuit"], default="Lunar base",
                            key="story_habitat") or "Lunar base"
h = HABITATS[name]
st.caption(f"{h['o2_frac'] * 100:.0f} % O₂ · {h['pressure_kPa']:.1f} kPa · source: {h.get('source', '—')}")

steps = st.columns(3, gap="medium")
with steps[0].container(border=True, height=330):
    st.markdown("#### ① The evidence")
    st.caption("Experiments whose conditions match this habitat best:")
    for r in ranked(name)[:3]:
        st.markdown(f"- **{label(r['id'])}** ({r['score']:.2f})")
    goto("See the full ranking", "🏆")
with steps[1].container(border=True, height=330):
    st.markdown("#### ② What is missing")
    gs = gaps_for(name)
    if gs:
        ui.callout("alert", f"**{gs[0]['text']}**")
    else:
        st.caption("No gap recorded for this habitat in coverage.json.")
    goto("See the gap on the map", "🗺️")
with steps[2].container(border=True, height=330):
    st.markdown("#### ③ Ask the documents")
    st.text_input("Question", "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?",
                  key="story_q", label_visibility="collapsed")
    st.caption("Every answer cites the PSI investigation, DOI, file and page.")
    goto("Ask", "💬")

st.divider()
st.caption(f"{data.key_number('test_records', 1947):,} test records · {data.key_number('documents_indexed', 93)} "
           "documents · every number links to its PSI DOI. Explore everything from the sidebar.")
