"""Shared parts of the Fire scenario and Response advisor pages (#49).

Everything shown comes from main's engine, analysis/scenario.py (assess, response_brief). This module only
collects the inputs, caches the calls and renders the returned blocks. No modelling happens here.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.habitats import HABITATS
from app import data, ui

FUEL_CLASSES = {"droplet": "Liquid droplet", "gas": "Gas flame", "solid": "Solid material"}
DROPLET_FUELS = ["heptane", "methanol"]
GAS_FUELS = ["ethane", "propane", "butane"]
SOLID_FIRST = ["SIBAL", "PMMA", "silicone", "Nomex"]          # the rest follow by number of tests
SOLID_IDS = ("PSI-25", "PSI-26", "PSI-98", "PSI-99", "PSI-100", "PSI-101", "PSI-102")   # same as the engine
CUSTOM = "Custom atmosphere"
GRAVITY_KEYS = ["micro", "lunar", "mars", "earth"]
# Outcome identity as on the Predict page: fixed order + validated categorical slots.
OUTCOMES = [("extinction", "Flame went out", "#2a78d6"), ("disruption", "Test ended early", "#eb6834"),
            ("completion", "Droplet burned completely", "#1baf7a")]
AGENT_COLOURS = {"CO2": "#2a78d6", "He": "#eb6834"}
AGENT_NAMES = {"CO2": "CO₂", "He": "Helium", "N2 only": "N₂ only"}
RANGE_LABELS = {"o2_frac": ("O₂ (% by volume)", 100), "pressure_kPa": ("Pressure (kPa)", 1),
                "x_co2": ("CO₂ in the atmosphere (%)", 100), "x_he": ("Helium in the atmosphere (%)", 100),
                "d0_mm": ("Droplet diameter (mm)", 1), "fuel_frac": ("Fuel share of the burner gas (%)", 100),
                "hc_rate_mg_s": ("Fuel flow (mg/s)", 1)}
GRID = "rgba(128,128,128,0.15)"


# ---------------------------------------------------------------- engine (cached; never re-implemented)
def disclaimer_text() -> str:
    from analysis.scenario import DISCLAIMER
    return DISCLAIMER


def gravity_levels() -> dict:
    from analysis.scenario import GRAVITY
    return GRAVITY


@st.cache_data(show_spinner="Running the scenario engine (the first run trains the models, about 30 s)…")
def _assess(items: tuple) -> dict:
    from analysis.scenario import assess
    return assess(dict(items))


@st.cache_data(show_spinner="Collecting the suppression evidence…")
def _brief(items: tuple) -> dict:
    from analysis.scenario import response_brief
    return response_brief(dict(items))


@st.cache_data(show_spinner=False)
def gravity_evidence() -> list[dict]:
    from analysis.scenario import gravity_evidence as ge
    return ge()


def _call(fn, scenario: dict) -> dict | None:
    """Engine result, or None (with a message on the page) when the engine cannot run here."""
    try:
        return fn(tuple(sorted(scenario.items())))
    except ImportError as e:
        ui.empty_state(f"The scenario engine cannot run in this Python environment ({e}). "
                       "It needs the packages in requirements.txt (scikit-learn).", icon="🧩")
        return None


def assess(scenario: dict) -> dict | None:
    return _call(_assess, scenario)


def brief(scenario: dict) -> dict | None:
    return _call(_brief, scenario)


# ---------------------------------------------------------------- disclaimer (on screen at all times)
def disclaimer_banner() -> None:
    """Banner at the top of the page and a copy in the sidebar, which stays visible while scrolling."""
    text = disclaimer_text()
    ui.callout("warning", f"**{text}**", icon="⚠️")
    ui.callout("warning", text, icon="⚠️", where=st.sidebar)


# ---------------------------------------------------------------- inputs
@st.cache_data(show_spinner=False)
def _solid_materials(mtime: float) -> list[str]:
    t = data.tests()
    t = t[(t["row_kind"] == "test") & t["investigation_id"].isin(SOLID_IDS) & t["fuel"].notna()]
    counts = t[~t["fuel"].str.contains(r"\+")]["fuel"].value_counts()
    rest = [m for m in counts.index if m not in SOLID_FIRST]
    return [m for m in SOLID_FIRST if m in counts.index] + rest


def scenario_form() -> dict:
    """The scenario inputs, shared by both pages: what burns, the atmosphere, gravity.
    The last scenario is kept in st.session_state so the other page opens with the same one."""
    for k, v in (st.session_state.get("fire_widgets") or {}).items():
        st.session_state.setdefault(k, v)
    ss = st.session_state
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1.5, 1], gap="large")
        with c1:
            st.markdown("**What is burning?**")
            cls = st.radio("Fuel type", list(FUEL_CLASSES), format_func=FUEL_CLASSES.get, key="fs_class",
                           label_visibility="collapsed")
            if cls == "droplet":
                fuel = st.selectbox("Liquid fuel", DROPLET_FUELS, key="fs_fuel_droplet")
            elif cls == "gas":
                fuel = st.selectbox("Gas fuel", GAS_FUELS, key="fs_fuel_gas")
            else:
                fuel = st.selectbox("Material", _solid_materials(data._mtime(data.TESTS)), key="fs_fuel_solid",
                                    help="Materials that appear in the PSI solid-fuel tables "
                                         "(BASS, BASS-II, Saffire, SAME).")
        with c2:
            st.markdown("**Atmosphere**")
            ss.setdefault("fs_preset", "ISS")
            preset = st.selectbox("Habitat preset", [CUSTOM] + list(HABITATS), key="fs_preset")
            h = HABITATS.get(preset, {"o2_frac": 0.21, "pressure_kPa": 101.3, "gravity": "micro"})
            k = preset   # keys per preset: choosing a preset resets the sliders to its values
            ss.setdefault(f"fs_o2_{k}", float(h["o2_frac"] * 100))
            ss.setdefault(f"fs_p_{k}", float(h["pressure_kPa"]))
            ss.setdefault(f"fs_g_{k}", h.get("gravity") if h.get("gravity") in GRAVITY_KEYS else "micro")
            a, b = st.columns(2)
            o2 = a.slider("O₂ (% by volume)", 10.0, 100.0, step=0.5, key=f"fs_o2_{k}")
            p = b.slider("Pressure (kPa)", 20.0, 320.0, step=0.5, key=f"fs_p_{k}",
                         help="101.3 kPa = 14.7 psi (sea level)")
            extra = {}
            if cls == "droplet":
                a, b, c = st.columns(3)
                extra["x_co2"] = a.slider("CO₂ in the atmosphere", 0.0, 0.7, step=0.01, key="fs_co2",
                                          help="Mole fraction of CO₂ replacing nitrogen (a suppressant gas).")
                extra["x_he"] = b.slider("Helium in the atmosphere", 0.0, 0.6, step=0.01, key="fs_he",
                                         help="Mole fraction of helium replacing nitrogen.")
                ss.setdefault("fs_d0", 2.5)
                extra["d0_mm"] = c.slider("Droplet diameter (mm)", 0.5, 6.0, step=0.1, key="fs_d0")
            elif cls == "gas":
                a, b = st.columns(2)
                ss.setdefault("fs_ff", 1.0)
                ss.setdefault("fs_hc", 1.0)
                extra["fuel_frac"] = a.slider("Fuel share of the burner gas", 0.0, 1.0, step=0.05, key="fs_ff",
                                              help="1.0 = pure fuel; lower = fuel diluted with nitrogen.")
                extra["hc_rate_mg_s"] = b.slider("Fuel flow (mg/s)", 0.0, 4.0, step=0.1, key="fs_hc")
        with c3:
            st.markdown("**Gravity**")
            levels = gravity_levels()
            gravity = st.radio("Gravity", GRAVITY_KEYS, key=f"fs_g_{k}", label_visibility="collapsed",
                               format_func=lambda g: levels[g]["label"])
    ss["fire_widgets"] = {key: ss[key] for key in ss if isinstance(key, str) and key.startswith("fs_")}
    return {"fuel_class": cls, "fuel": fuel, "o2_frac": round(o2 / 100, 4), "pressure_kPa": float(p),
            "gravity": gravity, **{key: float(v) for key, v in extra.items()}}


def scenario_line(s: dict) -> str:
    return (f"{FUEL_CLASSES[s['fuel_class']].lower()}, {s['fuel']} · {s['o2_frac'] * 100:g} % O₂ · "
            f"{s['pressure_kPa']:g} kPa · {gravity_levels()[s['gravity']]['label']}")


# ---------------------------------------------------------------- result blocks
def gravity_block(a: dict) -> None:
    """Moon / Mars: the engine's statement and never a number. Earth: the Saffire-II 0 g vs 1 g table."""
    g = a["gravity"]
    if g["data"] is False:
        ui.callout("note", f"**{g['label']}: no prediction.** {g['statement']}", icon="🌙")
    elif g["level"] == "earth":
        ui.callout("limit", f"**{g['label']}.** {g['statement']}", icon="🌍")
        st.markdown("**The same samples in microgravity and at 1 g (Saffire-II)**")
        saffire_table(gravity_evidence())
    else:
        st.caption(f"**{g['label']}.** {g['statement']}")


def saffire_table(rows: list[dict]) -> None:
    if not rows:
        ui.empty_state("Saffire-II did not fly this material, so PSI has no microgravity vs 1 g comparison for it.")
        return
    st.dataframe(pd.DataFrame([{
        "Sample": r["sample"], "Material": r["material"], "Thickness (mm)": r["thickness_mm"] or "—",
        "Flow": r["flow_direction"], "O₂ %": f"{r['o2_frac'] * 100:.1f}" if r["o2_frac"] else "—",
        "Burn length, microgravity": r["microgravity"]["burn_length"],
        "Spread rate, microgravity": r["microgravity"]["spread_rate"],
        "Burn length, 1 g": r["earth_1g"]["burn_length"], "Spread rate, 1 g": r["earth_1g"]["spread_rate"],
    } for r in rows]), hide_index=True)
    src = rows[0]["source"]
    ui.provenance_label("psi", f"Cell values as written in the Saffire-II table ({src['psi_id']}, DOI "
                               f"{ui.source_link(src['doi'])}); 'Note n' refers to that table's footnotes.")


def range_table(a: dict) -> None:
    rng = a.get("valid_range")
    if not rng:
        return
    rows = []
    for key, (lo, hi) in rng.items():
        label, scale = RANGE_LABELS.get(key, (key, 1))
        v = a["inputs"].get(key)
        inside = v is not None and lo - 1e-9 <= v <= hi + 1e-9
        rows.append({"Input": label, "Tested range": f"{lo * scale:g} – {hi * scale:g}",
                     "Your value": "—" if v is None else f"{v * scale:g}",
                     "": "—" if v is None else "inside" if inside else "⚠ outside"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def prediction_block(a: dict) -> None:
    """The engine's prediction, or its no-prediction state (a normal result, not an error)."""
    cls, pred = a["fuel_class"], a.get("prediction")
    if cls == "solid":
        solid_block(a)
        return
    if not pred:
        if a["gravity"]["data"] is False:      # Moon / Mars: the gravity box already says why; no number anywhere
            st.markdown("**No prediction for this gravity level.** The models are trained on microgravity tests only.")
        else:
            ui.callout("limit", f"**No prediction.** {a.get('why_no_prediction')}", icon="🚫")
            st.markdown("**What was tested**")
            range_table(a)
        return
    if a.get("prediction_valid_for"):
        ui.callout("warning", f"This estimate is valid for **{a['prediction_valid_for']}**.")
    if cls == "droplet":
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            probs = pred["probs"]
            fig = go.Figure(go.Bar(
                y=[lab for _, lab, _ in OUTCOMES][::-1], x=[probs[k] for k, *_ in OUTCOMES][::-1], orientation="h",
                marker_color=[c for *_, c in OUTCOMES][::-1], textposition="outside",
                text=[f"{probs[k]:.0%}" for k, *_ in OUTCOMES][::-1], hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
            fig.update_layout(height=190, margin=dict(l=0, r=40, t=10, b=0), showlegend=False,
                              xaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor=GRID))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with right:
            ui.metric_card(st, "Goes out on its own", _pct(pred.get("p_goes_out")),
                           help="Among burns that were not cut short: the chance the flame goes out with fuel left.")
            k = pred.get("burning_rate_k_mm2_s")
            if k:
                ui.metric_card(st, "Burning rate K (mm²/s)", f"{k['value']:.2f}",
                               help="How fast the droplet shrinks (d² law).")
                st.caption(f"interval {k['low']:.2f} – {k['high']:.2f} mm²/s")
        m = a.get("model") or {}
        c1, c2, c3 = st.columns(3)
        ui.metric_card(c1, "Model accuracy", _pct(m.get("cv_accuracy")), help="Cross-validated on FLEX tests.")
        ui.metric_card(c2, "Baseline", _pct(m.get("baseline_accuracy")),
                       help="Always guessing the most common outcome.")
        ui.metric_card(c3, "Training tests", m.get("n_train"))
    else:   # gas
        c1, c2 = st.columns(2)
        ui.metric_card(c1, "Hot ignition", _pct(pred.get("p_hot_ignition")),
                       help="Estimated chance that the flame ignites as a hot flame.")
        ui.metric_card(c2, "Goes out on its own, if ignited", _pct(pred.get("p_self_extinguish_if_ignited")),
                       help="Estimated chance that an ignited flame self-extinguishes.")
        names = {"hot_ignition": "Hot ignition", "self_extinguished": "Self-extinguish"}
        st.dataframe(pd.DataFrame([{
            "Model": names.get(t, t), "Accuracy (cross-validated)": _pct(m.get("cv_accuracy")),
            "Baseline": _pct(m.get("baseline_accuracy")), "Training tests": m.get("n_train"),
        } for t, m in (a.get("model") or {}).items()]), hide_index=True)
    src = a["sources"][0]
    ui.provenance_label("ml", f"Estimate from a model trained on {src['name']} ({src['psi_id']}) microgravity tests. "
                              "'Baseline' = always guessing the most common outcome.")
    with st.expander("Tested range of the model's inputs"):
        range_table(a)


def solid_block(a: dict) -> None:
    ui.callout("limit", "**No prediction — PSI has no outcome data for solid materials.** "
                        f"{a.get('why_no_prediction')}", icon="🚫")
    t = a.get("tests_of_this_material") or {}
    fuel = a["inputs"].get("fuel")
    c1, c2 = st.columns(2)
    ui.metric_card(c1, f"PSI tests of {fuel}", t.get("n", 0))
    rng = t.get("o2_range")
    ui.metric_card(c2, "O₂ range of those tests (%)",
                   f"{rng[0] * 100:.1f} – {rng[1] * 100:.1f}" if rng else "no data")
    if t.get("by_investigation"):
        st.caption("By investigation: " + " · ".join(
            f"{data.label(data.investigation(pid)) if data.investigation(pid) else pid}: {n}"
            for pid, n in sorted(t["by_investigation"].items(), key=lambda kv: -kv[1])))
    if a.get("gap"):
        ui.callout("gap", f"**Gap:** {a['gap']}")
    st.markdown(f"**{fuel} in microgravity and at 1 g (Saffire-II)**")
    saffire_table(a.get("gravity_comparison") or [])


def limits_and_sources(a: dict, extra_sources: list[dict] | None = None) -> None:
    if a.get("limits"):
        st.markdown("**Limits**")
        st.markdown("\n".join(f"- {x}" for x in a["limits"]))
    st.markdown("**Sources**")
    seen, lines = set(), []
    for s in list(a.get("sources") or []) + list(extra_sources or []):
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        doi = f" · DOI {ui.source_link(s['doi'])}" if s.get("doi") else ""
        lines.append(f"- {'**' + s['psi_id'] + '** ' if s.get('psi_id') else ''}{s['name']}{doi}")
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------- suppression (Response advisor)
def scan_chart(scan: dict) -> go.Figure:
    """P(flame goes out) vs fraction of suppressant gas released into the cabin, for one agent."""
    agent, colour = scan["agent"], AGENT_COLOURS[scan["agent"]]
    name = AGENT_NAMES[agent]
    pts = scan["points"]
    ok = [p for p in pts if p["p_goes_out"] is not None]
    step = pts[1]["agent_frac"] - pts[0]["agent_frac"] if len(pts) > 1 else 0.04
    fig = go.Figure()
    run = None                                    # grey bands over runs of points the engine returned as null
    for p in pts + [{"agent_frac": None, "p_goes_out": 0}]:
        if p["p_goes_out"] is None:
            run = [p["agent_frac"], p["agent_frac"]] if run is None else [run[0], p["agent_frac"]]
        elif run:
            fig.add_vrect(x0=run[0] - step / 2, x1=run[1] + step / 2, fillcolor="rgba(140,140,140,0.22)",
                          line_width=0, layer="below")
            fig.add_annotation(x=(run[0] + run[1]) / 2, y=0.5, text="outside tested range", textangle=-90,
                               showarrow=False, font=dict(size=11, color="rgba(140,140,140,1)"))
            run = None
    fig.add_hline(y=0.9, line=dict(color="rgba(140,140,140,0.8)", dash="dot", width=1))
    fig.add_scatter(x=[p["agent_frac"] for p in ok], y=[p["p_goes_out"] for p in ok], mode="lines+markers",
                    name=name, line=dict(color=colour, width=2), marker=dict(size=8, color=colour),
                    customdata=[p["o2_frac"] for p in ok],
                    hovertemplate=f"{name} released: %{{x:.0%}}<br>O₂ falls to %{{customdata:.1%}}<br>"
                                  "P(flame goes out): %{y:.0%}<extra></extra>")
    reach = scan.get("first_fraction_reaching_90pct")
    if reach is not None:
        p90 = next(p for p in ok if p["agent_frac"] == reach)
        fig.add_scatter(x=[reach], y=[p90["p_goes_out"]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=2, color=colour)))
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
                      xaxis=dict(title=f"{name} released into the cabin (share of the atmosphere)", tickformat=".0%",
                                 range=[-0.02, 0.62], gridcolor=GRID),
                      yaxis=dict(title="P(flame goes out)", tickformat=".0%", range=[0, 1.05], gridcolor=GRID))
    return fig


def scan_summary(scan: dict) -> str:
    name, reach = AGENT_NAMES[scan["agent"]], scan.get("first_fraction_reaching_90pct")
    if not scan.get("supported_points"):
        return f"**{name}:** every point is outside FLEX's tested range, so there is no estimate."
    if reach is None:
        return f"**{name}:** the estimate stays below 90 % at every tested fraction."
    p = next(x for x in scan["points"] if x["agent_frac"] == reach)
    if reach == 0:
        return f"**{name}:** the estimate is already ≥ 90 % with no agent added."
    return (f"**{name}:** the estimate first reaches 90 % at **{reach:.0%}** of the atmosphere "
            f"(O₂ falls to {p['o2_frac'] * 100:.1f} %). Circled on the chart.")


def evidence_table(sup: dict) -> None:
    agents = ["N2 only", "CO2", "He"]
    st.dataframe(pd.DataFrame([{
        "O₂ in the atmosphere": row["o2_band"],
        **{AGENT_NAMES[ag]: (f"{row[ag]['share_went_out']:.0%} (n={row[ag]['n']})"
                             if row[ag]["share_went_out"] is not None else "no tests")
           for ag in agents},
    } for row in sup["table"]]), hide_index=True)


def misconceptions(items: list[dict]) -> None:
    """NASA's wording, verbatim. Misconceptions are struck through; the detection item is a finding."""
    for it in items:
        src = it["source"]["name"]
        with st.container(border=True):
            if it["status"] == "misconception":
                st.markdown(f":red-badge[Misconception] ~~{it['claim']}~~")
            else:
                st.markdown(f":blue-badge[Finding] {it['claim']}")
            st.markdown(it["verdict"])
            st.caption(f"Quoted word for word from: {src}.")   # NASA's guide, not a PSI dataset: no PSI badge
