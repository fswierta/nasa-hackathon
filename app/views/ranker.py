import plotly.graph_objects as go
import streamlit as st

from ai.habitats import HABITATS
from ai.ranking import DEFAULT_WEIGHTS, PILLAR_FOCUS_WEIGHT, rank
from app import data

# Fixed component order + colours from the validated categorical palette (never re-cycled). Green
# (slot 6) is left out: next to aqua ("Airflow overlap") it read as the same colour (#18). Checked:
# worst adjacent OKLab dE x100 = 19.6 normal, 9.1 protan, 9.2 deutan (>= 15 / 8).
COMPONENTS = [
    ("o2", "O₂ overlap", "#2a78d6"),
    ("pressure", "Pressure overlap", "#eb6834"),
    ("flow", "Airflow overlap", "#1baf7a"),
    ("material", "Material match", "#eda100"),
    ("pillar", "Fire-safety pillar", "#e87ba4"),
    ("tier", "Relevance tier", "#6250d6"),
    ("richness", "Data richness", "#e34948"),
]
NAMES = {k: n for k, n, _ in COMPONENTS}
FIELD_LABEL = {"o2_frac": "O₂", "pressure_kPa": "pressure", "flow_cm_s": "airflow",
               "materials": "materials", "gravity": "gravity"}
MATERIALS = sorted({m for h in HABITATS.values() for m in h["materials"]} |
                   {"PMMA", "Nomex", "silicone", "Kapton", "Teflon", "Ultem", "cotton/fiberglass fabric"},
                   key=str.lower)

st.title("🏆 Scenario ranker")
st.markdown("Describe a crewed environment and see **which PSI experiments are the most relevant evidence** "
            "for its fire safety, with the reason behind every score.")

left, right = st.columns([1.25, 2.2], gap="large")

# ---------------------------------------------------------------- scenario form
with left:
    with st.container(border=True):
        preset = st.selectbox("Habitat preset", list(HABITATS), index=list(HABITATS).index("Lunar base"))
        h = HABITATS[preset]
        k = preset  # widget keys per preset -> switching preset resets the fields to its values
        o2_pct = st.slider("O₂ (% by volume)", 10.0, 100.0, float(h["o2_frac"] * 100), 0.5, key=f"o2_{k}")
        p_kpa = st.number_input("Cabin pressure (kPa)", 5.0, 400.0, float(h["pressure_kPa"]), 0.5,
                                key=f"p_{k}", help="101.3 kPa = 14.7 psi (sea level)")
        st.caption(f"= {p_kpa / 6.894757:.1f} psi")
        flow = st.slider("Ventilation airflow (cm/s)", 0.0, 50.0, float(h["flow_cm_s"]), 0.5, key=f"f_{k}")
        mats = st.multiselect("Materials in the cabin", MATERIALS,
                              default=[m for m in h["materials"] if m in MATERIALS], key=f"m_{k}")
        pillar = st.selectbox("Focus", ["any", "prevention", "detection", "suppression"], key=f"pl_{k}",
                              help="Prevention = ignition & flammability · detection = smoke sensing · "
                                   "suppression = extinguishing")

    assumed = [FIELD_LABEL.get(a, a) for a in h.get("assumptions", [])]
    st.caption(f"**Preset source:** {h.get('source', '—')}")
    if assumed:
        st.caption("**Assumed, not from PSI data:** " + ", ".join(assumed) + ".")

    with st.expander("Scoring weights"):
        st.caption("Each component scores 0–1; the total is the weighted average. "
                   f"Choosing a focus raises the pillar weight to {PILLAR_FOCUS_WEIGHT}.")
        if st.button("Reset weights"):
            for key in DEFAULT_WEIGHTS:
                for focused in (True, False):
                    st.session_state.pop(f"w_{key}_{focused}", None)
        base = dict(DEFAULT_WEIGHTS)
        if pillar != "any":
            base["pillar"] = PILLAR_FOCUS_WEIGHT
        weights = {key: st.slider(NAMES[key], 0.0, 1.0, float(base[key]), 0.05, key=f"w_{key}_{pillar != 'any'}")
                   for key, *_ in COMPONENTS}

scenario = {**h, "o2_frac": o2_pct / 100, "pressure_kPa": p_kpa, "flow_cm_s": flow,
            "materials": mats, "pillar": None if pillar == "any" else pillar}
results = rank(scenario, weights)
total_w = sum(weights.values()) or 1.0

# ---------------------------------------------------------------- results
with right:
    top_n = st.radio("Show", [5, 10, 24], index=1, horizontal=True,
                     format_func=lambda n: f"Top {n}" if n < 24 else "All 24")
    shown = results[:top_n]

    # Stacked contribution bar: each segment = weight × component score / Σ weights, summing to the score.
    labels = [data.label(data.investigation(r["id"])) for r in shown][::-1]
    fig = go.Figure()
    for key, name, colour in COMPONENTS:
        vals = [((r["breakdown"][key] or 0) * weights[key] / total_w) for r in shown][::-1]
        raw = [r["breakdown"][key] for r in shown][::-1]
        fig.add_bar(
            y=labels, x=vals, name=name, orientation="h", marker_color=colour,
            marker_line_width=1, marker_line_color="rgba(255,255,255,0.9)",
            customdata=[("no data" if v is None else f"{v:.2f}") for v in raw],
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{customdata}} → +%{{x:.3f}}<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=90 + 34 * len(shown), margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(title="Relevance score (0–1)", range=[0, 1], gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(automargin=True), bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, traceorder="normal"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    for n, r in enumerate(shown, 1):
        inv = data.investigation(r["id"])
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{n}. {data.label(inv)}** — {inv.get('title')}")
            c2.markdown(f"<div style='text-align:right;font-size:1.4rem;font-weight:600'>{r['score']:.2f}</div>",
                        unsafe_allow_html=True)
            st.progress(min(1.0, r["score"]))
            st.write(r["why"])
            st.caption(" · ".join(f"{NAMES[key]} **{'no data' if v is None else f'{v:.2f}'}**"
                                  for key, v in r["breakdown"].items()))
            for g in r["gaps"]:
                st.caption(f"⚠️ {g}")
            b1, b2 = st.columns([1, 2])
            if b1.button("Open detail →", key=f"open_{r['id']}", type="tertiary"):
                st.session_state["detail_id"] = r["id"]
                st.switch_page("views/detail.py")
            if r.get("doi"):
                b2.markdown(f"[DOI {r['doi']}]({data.doi_url(r['doi'])})")

    st.caption("Scores are deterministic (no AI) so every number can be traced. 'no data' means the PSI "
               "table has no such column; it contributes nothing and is never treated as a match.")
