"""Proposal A — dashboard-first Scenario ranker: scenario in one row on top, KPIs, then a full-width ranking."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.proposals.common import HABITATS, data, gaps_for, habitat_cov, label, ui
from ai.ranking import rank

ui.page_header("Scenario ranker", "Which PSI experiments are the most relevant evidence for this habitat?", "🏆")

# Scenario bar: every input in one row
with st.container(border=True):
    c = st.columns([1.2, 0.8, 0.8, 0.8, 2.2])
    name = c[0].selectbox("Habitat", list(HABITATS), index=list(HABITATS).index("Lunar base"))
    h = HABITATS[name]
    o2 = c[1].number_input("O₂ %", 10.0, 100.0, h["o2_frac"] * 100, 0.5, key=f"o2_{name}")
    p = c[2].number_input("Pressure kPa", 5.0, 400.0, float(h["pressure_kPa"]), 0.5, key=f"p_{name}")
    f = c[3].number_input("Airflow cm/s", 0.0, 50.0, float(h["flow_cm_s"]), 0.5, key=f"f_{name}")
    g = c[4].segmented_control("Gravity", list(data.GRAVITY_OPTIONS), default=h.get("gravity") or "micro",
                               format_func=data.GRAVITY_OPTIONS.get, key=f"g_{name}")
res = rank({**h, "o2_frac": o2 / 100, "pressure_kPa": p, "flow_cm_s": f, "gravity": g})

# KPI row
cov = habitat_cov(name)
k = st.columns(4)
ui.metric_card(k[0], "Best match", f"{res[0]['score']:.2f}", help=label(res[0]["id"]))
ui.metric_card(k[1], "Investigations ≥ 0.5", sum(r["score"] >= 0.5 for r in res))
ui.metric_card(k[2], "Tests near this atmosphere", cov.get("n_tests_near", "—"))
ui.metric_card(k[3], "…with solid materials", (cov.get("by_fuel_phase") or {}).get("solid", "—"))
if g in ("lunar", "mars"):
    ui.callout("note", data.PARTIAL_G_SHORT, icon="🌙")
for gap in gaps_for(name)[:1]:
    ui.callout("gap", f"**{gap['text']}**")

# Full-width ranking: score bars + a table with the reason
top = res[:10]
fig = go.Figure(go.Bar(x=[r["score"] for r in top][::-1], y=[label(r["id"]) for r in top][::-1], orientation="h",
                       marker=dict(color="#F26419"), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
fig.update_layout(height=360, margin=dict(l=0, r=10, t=10, b=0),
                  xaxis=dict(range=[0, 1], title="Relevance score (0–1)", gridcolor="rgba(128,128,128,0.15)"))
left, right = st.columns([1, 1.4], gap="large")
left.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
right.dataframe(pd.DataFrame([{"#": n, "Investigation": label(r["id"]), "Score": round(r["score"], 2),
                               "Why": r["why"]} for n, r in enumerate(top, 1)]),
                hide_index=True, height=360, column_config={"Why": st.column_config.TextColumn(width="large")})
ui.provenance_label("computed", "Scores from ai.ranking.rank with default weights; the live page explains each part.")
