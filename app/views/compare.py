import pandas as pd
import streamlit as st

from app import data

import plotly.graph_objects as go  # noqa: E402

# Cluster identity: validated categorical slots in fixed order + a marker shape per cluster
# (second encoding, because a scatter with >3 hues is not reliably colour-distinguishable).
CLUSTER_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLUSTER_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "star", "hexagon", "triangle-down"]

st.title("⚖️ Compare investigations")
st.caption("Pick 2–4 investigations. Rows where they differ are marked ≠.")

invs = data.investigations()
ids = [i["id"] for i in invs]
default = [x for x in ("PSI-26", "PSI-25", "PSI-99") if x in ids]
picked = st.multiselect("Investigations", ids, default=default, max_selections=4,
                        format_func=lambda x: data.label(data.investigation(x)))


def similarity_map(sim: dict, highlight: list[str]) -> None:
    clusters = sim.get("clusters") or []
    by_cluster = {c["id"]: c for c in clusters}
    order = [c["id"] for c in clusters] or sorted({p["cluster"] for p in sim["points"]}, key=str)
    fig = go.Figure()
    for n, cid in enumerate(order):
        pts = [p for p in sim["points"] if p["cluster"] == cid]
        lab = (by_cluster.get(cid) or {}).get("label") or f"Group {cid}"
        fig.add_scatter(
            x=[p["x"] for p in pts], y=[p["y"] for p in pts], mode="markers+text", name=lab,
            text=[p.get("acronym") or p["id"] for p in pts], textposition="top center", textfont=dict(size=11),
            marker=dict(size=[17 if p["id"] in highlight else 11 for p in pts],
                        color=CLUSTER_COLOURS[n % len(CLUSTER_COLOURS)], symbol=CLUSTER_SYMBOLS[n % 8],
                        line=dict(width=[3 if p["id"] in highlight else 1 for p in pts], color="#ffffff")),
            customdata=[p["id"] for p in pts], hovertemplate="%{text} (%{customdata})<br>" + lab + "<extra></extra>")
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, x=0, title=None))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Each dot is an investigation; close dots are similar ({sim.get('method', 'method not stated')}). "
               "Groups are computed, not hand-made. Larger dots = your selection. Axes have no units.")


with st.expander("🧭 Similarity map of all investigations", expanded=len(picked) < 2):
    sim = data.similarity()
    if sim and sim.get("points"):
        similarity_map(sim, picked)
    else:
        st.info("The similarity map is not available yet (processed/analysis/similarity.json, issue #24).")

if len(picked) < 2:
    st.info("Select at least two investigations.")
    st.stop()

tests = data.tests()


def join(v):
    return ", ".join(v) if v else "—"


def outcomes(pid: str) -> str:
    o = tests.loc[tests["investigation_id"] == pid, "outcome"].dropna()
    if o.empty:
        return "—"
    return ", ".join(f"{k} ({n})" for k, n in o.value_counts().head(5).items())


ROWS = [
    ("Title", lambda i: i.get("title") or "—"),
    ("Relevance tier", lambda i: i.get("relevance_tier") or "—"),
    ("Type · platform", lambda i: f"{i.get('type') or '—'} · {i.get('platform') or '—'}"),
    ("Hardware family", lambda i: i.get("hardware_family") or "—"),
    ("Fuel phase", lambda i: i.get("fuel_phase") or "—"),
    ("Fuels / materials", lambda i: join(i.get("fuels"))),
    ("Sample geometry", lambda i: join(i.get("sample_geometry"))),
    ("O₂ (%)", lambda i: data.fmt_range(i.get("o2_range"), scale=100, digits=1)),
    ("Pressure (kPa)", lambda i: data.fmt_range(i.get("pressure_range_kPa"), digits=1)),
    ("Airflow (cm/s)", lambda i: data.fmt_range(i.get("flow_range_cm_s"), digits=1)),
    ("Diluents", lambda i: join(i.get("diluents"))),
    ("Varied between tests", lambda i: data.pretty_vars(i.get("varied_variables"))),
    ("Held fixed", lambda i: data.pretty_vars(i.get("fixed_variables"))),
    ("Table rows", lambda i: "{} {}".format(*data.table_rows(i["id"]))),
    ("Recorded outcomes", lambda i: outcomes(i["id"])),
    ("Publications", lambda i: str(len(i.get("publications") or []))),
    ("DOI", lambda i: i.get("doi") or "—"),
]

sel = [data.investigation(p) for p in picked]
table = {"": [], "≠": []}
for p in picked:
    table[data.label(data.investigation(p))] = []
for name, fn in ROWS:
    vals = [fn(i) for i in sel]
    table[""].append(name)
    table["≠"].append("≠" if len(set(vals)) > 1 else "")
    for p, v in zip(picked, vals):
        table[data.label(data.investigation(p))].append(v)
df = pd.DataFrame(table)

only_diff = st.toggle("Show only rows that differ")
if only_diff:
    df = df[df["≠"] == "≠"]
st.dataframe(df, hide_index=True, height=38 + 35 * len(df),
             column_config={"": st.column_config.TextColumn(width="medium"),
                            "≠": st.column_config.TextColumn(width="small")})

st.subheader("Summaries")
cols = st.columns(len(sel))
for col, i in zip(cols, sel):
    with col:
        st.markdown(f"**{data.label(i)}**")
        prof = data.profile(i["id"])
        if prof and prof.get("summary_plain"):
            st.write(prof["summary_plain"])
            st.caption(("Summary from PSI metadata." if data.written_by_rules(prof.get("generated"))
                        else "✨ AI-written summary") + " Sources on the detail page.")
        else:
            st.write((i.get("objective") or "No objective on record.")[:600])
            st.caption("Objective from PSI metadata.")
        if i.get("doi"):
            st.markdown(f"[DOI {i['doi']}]({data.doi_url(i['doi'])})")
