import plotly.graph_objects as go
import streamlit as st

from app import data, ui

# Outcome identity: fixed order + validated categorical slots (bars) and marker shapes (map).
OUTCOMES = [("extinction", "Flame went out", "#2a78d6", "x"),
            ("disruption", "Test ended early", "#eb6834", "diamond"),
            ("completion", "Droplet burned completely", "#1baf7a", "circle")]
# Sequential single-hue ramp for P(extinction), light -> dark blue.
P_SCALE = [[0.0, "#eef4fc"], [0.5, "#7fb0ea"], [1.0, "#0b3d7a"]]
POINT_COLOUR = "#eb6834"

ui.page_header("Predict extinction", ("Would a burning **fuel droplet** go out on its own in a given atmosphere? A model trained on the "
            "FLEX experiments (ISS) estimates the outcome and refuses to answer outside the conditions that "
            "were actually tested."), "🧯")
ui.callout("limit", "Scope: liquid fuel droplets (heptane, methanol) in microgravity, from FLEX (PSI-69). "
        "**Not** solid materials like fabrics or plastics — PSI has no comparable solid-fuel tests.", icon="💧")

fb = data.flex_boundary()
card = (fb or {}).get("model_card") or {}
problem = data.flex_model_ready()

left, right = st.columns([1, 1.4], gap="large")

# ---------------------------------------------------------------- inputs + prediction
with left:
    with st.container(border=True):
        fuel = st.radio("Fuel", ["Heptane", "Methanol"], horizontal=True)
        vr = (card.get("valid_range") or {}).get(fuel, {})

        def hint(key, scale=1.0, unit=""):
            lo, hi = vr.get(key, (None, None))
            return f"Tested: {lo * scale:g}–{hi * scale:g} {unit}".strip() if lo is not None else None

        o2 = st.slider("O₂ (% by volume)", 10.0, 40.0, 21.0, 0.5, help=hint("o2_frac", 100, "%"))
        p = st.number_input("Pressure (kPa)", 20.0, 400.0, 101.3, 0.5, help=hint("pressure_kPa", 1, "kPa"))
        co2 = st.slider("CO₂ added (mole fraction)", 0.0, 0.8, 0.0, 0.01, help=hint("x_co2"))
        he = st.slider("Helium added (mole fraction)", 0.0, 0.6, 0.0, 0.01, help=hint("x_he"))
        d0 = st.slider("Initial droplet diameter (mm)", 0.5, 6.0, 2.5, 0.1, help=hint("d0_mm", 1, "mm"))

    if problem:
        ui.callout("warning", f"Prediction not available: {problem}.")
        res = None
    else:
        from analysis.flex_classifier import predict_outcome
        res = predict_outcome(fuel, o2 / 100, p, co2, he, d0)

    if res and res.get("probs"):
        probs = res["probs"]
        fig = go.Figure(go.Bar(
            y=[lab for _, lab, _, _ in OUTCOMES][::-1], x=[probs[k] for k, *_ in OUTCOMES][::-1],
            orientation="h", marker_color=[c for _, _, c, _ in OUTCOMES][::-1],
            text=[f"{probs[k]:.0%}" for k, *_ in OUTCOMES][::-1], textposition="outside",
            hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        fig.update_layout(height=190, margin=dict(l=0, r=40, t=10, b=0), showlegend=False,
                          xaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="rgba(128,128,128,0.15)"))
        st.markdown(f"**Most likely: {dict((k, lab) for k, lab, *_ in OUTCOMES)[res['predicted']].lower()}**")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    elif res:
        ui.callout("warning", f"No prediction: {res.get('reason')}", icon="🚫")

    if card:
        cv = (card.get("cv") or {}).get(card.get("model"), {})
        c1, c2, c3 = st.columns(3)
        ui.metric_card(c1, "CV accuracy", f"{cv.get('cv_accuracy', 0):.0%}",
                           help=f"{card.get('cv_scheme')}; ± {cv.get('cv_accuracy_sd', 0):.0%}")
        ui.metric_card(c2, "Baseline", f"{card.get('baseline_accuracy', 0):.0%}", help=card.get("baseline"))
        ui.metric_card(c3, "Training tests", card.get("n_train"))
        rec = cv.get("recall") or {}
        ui.provenance_label("ml", f"Model: {card.get('model')} on {', '.join(card.get('features', []))}. "
                   "Recall per outcome: " + ", ".join(f"{k} {v:.0%}" for k, v in rec.items()) +
                   ". 'Baseline' always guesses the most common outcome (extinction). "
                   f"Source: FLEX, PSI-69, DOI {ui.source_link(card.get('source', {}).get('doi'))}.")

# ---------------------------------------------------------------- boundary map
with right:
    st.subheader("Where droplets go out")
    if not fb or not fb.get("grids"):
        ui.empty_state("The extinction map is not available yet (processed/analysis/flex_boundary.json).")
    else:
        grids = [g for g in fb["grids"] if g["fuel"] == fuel]
        c1, c2 = st.columns(2)
        pressures = sorted({g["pressure_kPa"] for g in grids})
        pk = c1.radio("Pressure", pressures, index=len(pressures) - 1, horizontal=True,
                      format_func=lambda v: f"{v:g} kPa")
        dil = c2.radio("Diluent", sorted({g["diluent"] for g in grids}), horizontal=True,
                       format_func=lambda d: {"CO2": "CO₂", "HE": "Helium"}.get(d, d))
        g = next(x for x in grids if x["pressure_kPa"] == pk and x["diluent"] == dil)
        key = "x_co2" if dil == "CO2" else "x_he"
        dname = {"CO2": "CO₂", "HE": "Helium"}.get(dil, dil)

        fig = go.Figure(go.Heatmap(
            x=[v * 100 for v in g["o2_frac"]], y=g["x_diluent"], z=g["p_extinction"], zmin=0, zmax=1,
            colorscale=P_SCALE, hoverongaps=False,
            colorbar=dict(title="P(goes out)", tickformat=".0%", thickness=12),
            hovertemplate=f"O₂ %{{x:.1f}} %<br>{dname} %{{y:.2f}}<br>P(extinction) %{{z:.0%}}<extra></extra>"))
        for k, lab, _, sym in OUTCOMES:
            pts = [t for t in g["tests"] if t["outcome"] == k]
            if pts:
                fig.add_scatter(x=[t["o2_frac"] * 100 for t in pts], y=[t[key] for t in pts], mode="markers",
                                name=f"Real test: {lab.lower()}",
                                marker=dict(symbol=sym, size=9, color=POINT_COLOUR,
                                            line=dict(width=1.5, color="#ffffff")),
                                text=[t["test_id"] for t in pts],
                                hovertemplate=f"FLEX test %{{text}}: {lab.lower()}<extra></extra>")
        other = he if dil == "CO2" else co2
        if other == 0:   # the map is a slice with the other diluent at 0; mark the input only if it lies on it
            fig.add_scatter(x=[o2], y=[co2 if dil == "CO2" else he], mode="markers", name="Your input",
                            marker=dict(symbol="star", size=18, color="#ffffff", line=dict(width=2, color="#0b0b0b")),
                            hovertemplate="Your input<extra></extra>")
        fig.update_layout(
            height=470, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="O₂ (% by volume)"), yaxis=dict(title=f"{dname} added (mole fraction)"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, title=None))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"{fuel}, {pk:g} kPa, droplet {g['d0_mm']} mm. Shaded = model's probability that the flame goes "
                   "out; blank = no real FLEX test nearby, so no prediction is shown there. Markers = real tests "
                   "(shape = how they ended).")
