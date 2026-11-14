"""Proposal B — story-first Scenario ranker: choose → read the evidence → see what is missing."""
import streamlit as st

from ai.ranking import rank
from app.proposals.common import goto, HABITATS, data, gaps_for, label, ui

ui.page_header("Scenario ranker", "Three steps: choose a habitat, read the evidence, see what is missing.", "🏆")

st.markdown("#### Step 1 · Choose your habitat")
name = st.segmented_control("Habitat", list(HABITATS), default="Lunar base", key="b_habitat",
                            label_visibility="collapsed") or "Lunar base"
h = HABITATS[name]
with st.expander(f"{h['o2_frac'] * 100:.0f} % O₂ · {h['pressure_kPa']:.1f} kPa · "
                 f"{h['flow_cm_s']:.0f} cm/s airflow — adjust"):
    c = st.columns(3)
    o2 = c[0].slider("O₂ %", 10.0, 100.0, h["o2_frac"] * 100, 0.5, key=f"b_o2_{name}")
    p = c[1].slider("Pressure kPa", 5.0, 400.0, float(h["pressure_kPa"]), 0.5, key=f"b_p_{name}")
    f = c[2].slider("Airflow cm/s", 0.0, 50.0, float(h["flow_cm_s"]), 0.5, key=f"b_f_{name}")
res = rank({**h, "o2_frac": o2 / 100, "pressure_kPa": p, "flow_cm_s": f})

st.markdown("#### Step 2 · The best evidence")
for col, r in zip(st.columns(3, gap="medium"), res[:3]):
    with col.container(border=True, height=250):
        st.markdown(f"**{label(r['id'])}**")
        st.progress(min(1.0, r["score"]), text=f"relevance {r['score']:.2f}")
        st.caption(r["why"])
        if r.get("doi"):
            st.caption("DOI " + ui.source_link(r["doi"]))
with st.expander(f"All {len(res)} investigations"):
    for n, r in enumerate(res, 1):
        st.markdown(f"{n}. {label(r['id'])} · {r['score']:.2f}")

st.markdown("#### Step 3 · What is missing")
gs = gaps_for(name)
for g in gs[:2]:
    ui.callout("alert" if g is gs[0] else "gap", f"**{g['text']}**  \n{data.partial_g_fix(g.get('evidence', ''))}")
if h.get("gravity") in ("lunar", "mars"):
    ui.callout("note", data.PARTIAL_G_SHORT, icon="🌙")
c1, c2 = st.columns(2)
with c1:
    goto("See the gap on the condition map", "🗺️")
with c2:
    goto("Ask the documents about it", "💬")
