import plotly.graph_objects as go
import streamlit as st

from app import data, ml_ui, ui

# Outcome identity: fixed order + validated categorical slots (bars) and marker shapes (map).
OUTCOMES = [("extinction", "Flame went out", ui.OUTCOME_COLOURS["extinction"], "x"),
            ("disruption", "Test ended early", ui.OUTCOME_COLOURS["disruption"], "diamond"),
            ("completion", "Droplet burned completely", ui.OUTCOME_COLOURS["completion"], "circle")]
# Sequential single-hue blue for P(extinction), shared with the Condition map overlay: the high end always has
# the most contrast against the page (light -> dark on the light theme, dark -> light on the dark theme).
P_SCALE = ui.seq_blue()
POINT_COLOUR = ui.MAGENTA   # real FLEX tests on the blue map (orange is the UI accent, not a data colour)

ui.page_header("Predict extinction", ("Would a burning **fuel droplet** go out on its own in a given atmosphere? A model trained on the "
            "FLEX experiments (ISS) estimates the outcome and refuses to answer outside the conditions that "
            "were actually tested."), ui.ICONS["predict"])
ui.callout("limit", "Scope: liquid fuel droplets (heptane, methanol) in microgravity, from FLEX (PSI-69). "
        "**Not** solid materials like fabrics or plastics — PSI has no comparable solid-fuel tests.")

fb = data.flex_boundary()
card = (fb or {}).get("model_card") or {}
problem = data.flex_model_ready()

left, right = st.columns([1, 1.4], gap="large")

# ---------------------------------------------------------------- inputs + prediction
with left:
    with ui.card():
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

    # #58: the registry's FLEX model, shown with the shared prediction card (number or reason, ranges, real tests).
    inputs = {"fuel": fuel, "o2_frac": round(o2 / 100, 4), "pressure_kPa": float(p), "x_co2": float(co2),
              "x_he": float(he), "d0_mm": float(d0)}
    model, res = None, None
    if problem:
        ui.callout("warning", f"Prediction not available: {problem}.")
    else:
        model = ml_ui.spec("flex_outcome")
        res = ml_ui.predict("flex_outcome", **inputs)
        if res["ok"]:
            ending = dict((k, lab) for k, lab, *_ in OUTCOMES)[res["prediction"]["predicted"]].lower()
            st.markdown(f"**Most likely ending: {ending}**")
        ml_ui.prediction_card(res, model, key="predict", title="Will this droplet's flame go out?")
        with ui.card():
            st.markdown("**How good is this model?** Cross-validated accuracy next to a baseline that always "
                        "guesses the most common ending.")
            ml_ui.score_chart(model, key="predict_score")
            rec = model["metrics"].get("recall") or {}
            if rec:
                st.caption("Recall per ending: " + ", ".join(f"{k} {v:.0%}" for k, v in rec.items()) +
                           f". Source: FLEX, PSI-69, DOI {ui.source_link(model['source']['doi'])}.")

# ---------------------------------------------------------------- boundary map
with right:
    ui.section("Where droplets go out")
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

# ---------------------------------------------------------------- why this result? (#58)
if model is not None:
    ui.section("Why this result?")
    e1, e2 = st.columns(2, gap="large")
    with ui.card(e1):
        st.markdown("**For these inputs** (what-if against a typical FLEX test)")
        ml_ui.explanation_chart(ml_ui.explain("flex_outcome", **inputs) if res and res["ok"] else [],
                                key="predict_explain")
    with ui.card(e2):
        st.markdown("**Overall** (which inputs the model relies on)")
        ml_ui.importance_chart(model, key="predict")
