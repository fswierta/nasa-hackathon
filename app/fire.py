"""Shared parts of the Fire scenario and Suppression evidence pages (#49, #65).

Everything shown comes from main's engine, analysis/scenario.py (assess, response_brief). This module only
collects the inputs, caches the calls and renders the returned blocks. No modelling happens here.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.habitats import HABITATS
from app import data, ml_ui, ui

FUEL_CLASSES = {"droplet": "Liquid droplet", "gas": "Gas flame", "solid": "Solid material"}
DROPLET_FUELS = ["heptane", "methanol"]
GAS_FUELS = ["ethane", "propane", "butane"]
SOLID_FIRST = ["SIBAL", "PMMA", "silicone", "Nomex"]          # the rest follow by number of tests
SOLID_IDS = ("PSI-25", "PSI-26", "PSI-98", "PSI-99", "PSI-100", "PSI-101", "PSI-102")   # same as the engine
CUSTOM = "Custom atmosphere"
GRAVITY_KEYS = ["micro", "lunar", "mars", "earth"]
# Outcome identity as on the Predict page: fixed order + validated categorical slots.
AGENT_COLOURS = ui.AGENT_COLOURS
AGENT_NAMES = {"CO2": "CO₂", "He": "Helium", "N2 only": "N₂ only"}
RANGE_LABELS = {"o2_frac": ("O₂ (% by volume)", 100), "pressure_kPa": ("Pressure (kPa)", 1),
                "x_co2": ("CO₂ in the atmosphere (%)", 100), "x_he": ("Helium in the atmosphere (%)", 100),
                "d0_mm": ("Droplet diameter (mm)", 1), "fuel_frac": ("Fuel share of the burner gas (%)", 100),
                "hc_rate_mg_s": ("Fuel flow (mg/s)", 1)}
GRID = ui.GRID


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
                       "It needs the packages in requirements.txt (scikit-learn).")
        return None


def assess(scenario: dict) -> dict | None:
    return _call(_assess, scenario)


def brief(scenario: dict) -> dict | None:
    return _call(_brief, scenario)


# ---------------------------------------------------------------- disclaimer (on screen at all times)
def disclaimer_banner() -> None:
    """Banner at the top of the page and a copy in the sidebar, which stays visible while scrolling."""
    text = disclaimer_text()
    ui.callout("warning", f"**{text}**")
    ui.callout("warning", text, where=st.sidebar)


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
    with ui.card():
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
        ui.callout("note", f"**{g['label']}: no prediction.** {g['statement']}")
    elif g["level"] == "earth":
        ui.callout("limit", f"**{g['label']}.** {g['statement']}")
        st.markdown("**The same samples in microgravity and at 1 g (Saffire-II)**")
        saffire_table(gravity_evidence())
    else:
        st.caption(f"**{g['label']}.** {g['statement']}")


def saffire_table(rows: list[dict]) -> None:
    if not rows:
        ui.empty_state("Saffire-II did not fly this material, so PSI has no microgravity vs 1 g comparison for it.")
        return
    ui.table(pd.DataFrame([{
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
    ui.table(pd.DataFrame(rows), hide_index=True)


def registry_inputs(s: dict) -> tuple[str, dict]:
    """The scenario as inputs for the model registry (#58): (model_id, inputs)."""
    atm = {"o2_frac": s["o2_frac"], "pressure_kPa": s["pressure_kPa"]}
    if s["fuel_class"] == "droplet":
        return "flex_outcome", {"fuel": s["fuel"].capitalize(), **atm, "x_co2": s.get("x_co2", 0.0),
                                "x_he": s.get("x_he", 0.0), "d0_mm": s.get("d0_mm", 2.5)}
    return "cfig_ignition", {"fuel": s["fuel"], **atm, "fuel_frac": s.get("fuel_frac", 1.0),
                             "hc_rate_mg_s": s.get("hc_rate_mg_s", 1.0)}


def prediction_block(a: dict) -> None:
    """The prediction, or its no-prediction state (a normal result, not an error). Solids and Moon / Mars come from
    the scenario engine (analysis/scenario.py); droplet and gas numbers from the model registry (analysis/registry.py)
    through the shared ML cards, with the explanation of what moved them (#58)."""
    cls = a["fuel_class"]
    if cls == "solid":
        solid_block(a)
        return
    if a["gravity"]["data"] is False:          # Moon / Mars: the gravity box already says why; no number anywhere
        st.markdown("**No prediction for this gravity level.** The models are trained on microgravity tests only.")
        return
    if a.get("prediction_valid_for"):
        ui.callout("warning", f"This estimate is valid for **{a['prediction_valid_for']}**.")
    model_id, inputs = registry_inputs(a["inputs"])
    res = ml_ui.predict(model_id, **inputs)
    second_id = "flex_burning_rate" if cls == "droplet" else "cfig_self_extinguish"
    left, right = st.columns([1.3, 1], gap="large")
    ml_ui.prediction_card(res, ml_ui.spec(model_id), where=left, key=f"fs_{model_id}")
    if res["ok"]:
        ml_ui.prediction_card(ml_ui.predict(second_id, **inputs), ml_ui.spec(second_id), where=right, compact=True,
                              key=f"fs_{second_id}")
        with ui.card(right):
            st.markdown("**Why this result?** (what-if against a typical test)")
            ml_ui.explanation_chart(ml_ui.explain(model_id, **inputs), key=f"fs_explain_{model_id}")
    with st.expander("Tested ranges as a table"):
        range_table(a)


def solid_block(a: dict) -> None:
    ui.callout("limit", "**No prediction — PSI has no outcome data for solid materials.** "
                        f"{a.get('why_no_prediction')}")
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
    rep = data.report_rows()   # #65 item 3: ACME BRE report rows (main, #63 item 2), the closest solid-like evidence
    if not rep.empty:
        st.markdown("**Closest evidence at exploration-type atmospheres:** ACME BRE's gas burner *emulating* a solid "
                    "(it is not a real fabric or plastic sample), with the atmospheres stated in the BRE Final Report:")
        st.markdown("\n".join(f"- {data.report_row_line(r)}" for _, r in rep.iterrows()))
        ui.provenance_label("psi", "From the report pages cited on each line (rows added by the ETL, flagged "
                                   "'from_report'); see them on the Condition map as ★.")
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


# ---------------------------------------------------------------- suppression evidence (#65 item 2)
NIOSH_CO2_IDLH = "https://www.cdc.gov/niosh/idlh/124389.html"
DILUENT_CAVEAT = ("FLEX used CO₂/He to map flammability limits, not as a cabin-flooding agent; CO₂ above 4 % is "
                  f"immediately dangerous to crew ([NIOSH IDLH]({NIOSH_CO2_IDLH})).")


@st.cache_data(show_spinner="Computing the model curves…")
def _diluent_scans(fuel: str, o2_frac: float, pressure_kPa: float, d0_mm: float) -> list[dict]:
    """The scenario engine's scans in 'replace_n2' mode: the diluent replaces nitrogen at a constant O2 fraction,
    which is how FLEX pre-set its tests (not a gas released into a cabin)."""
    from analysis.scenario import suppressant_scan
    return [suppressant_scan(fuel, o2_frac, pressure_kPa, agent, d0_mm, mode="replace_n2") for agent in ("CO2", "He")]


def diluent_scans(s: dict) -> list[dict]:
    """Model curves for a droplet scenario, [] for any other fuel class (the model only covers FLEX droplets)."""
    if s.get("fuel_class") != "droplet":
        return []
    try:
        return _diluent_scans(s["fuel"], s["o2_frac"], s["pressure_kPa"], s.get("d0_mm", 2.5))
    except ImportError:
        return []


def scan_chart(scan: dict) -> go.Figure:
    """Model P(flame goes out) vs the diluent fraction pre-set in the atmosphere, for one agent. No threshold line
    or 'reaches X %' marker: this is a flammability map from FLEX, not a dose (#65 item 2)."""
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
    fig.add_scatter(x=[p["agent_frac"] for p in ok], y=[p["p_goes_out"] for p in ok], mode="lines+markers",
                    name=name, line=dict(color=colour, width=2), marker=dict(size=8, color=colour),
                    customdata=[p["o2_frac"] for p in ok],
                    hovertemplate=f"{name} pre-set (replacing N₂): %{{x:.0%}}<br>O₂ held at %{{customdata:.1%}}<br>"
                                  "model P(flame goes out): %{y:.0%}<extra></extra>")
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
                      xaxis=dict(title=f"{name}: pre-set diluent fraction in FLEX tests", tickformat=".0%",
                                 range=[-0.02, 0.62], gridcolor=GRID),
                      yaxis=dict(title="model P(flame goes out)", tickformat=".0%", range=[0, 1.05], gridcolor=GRID))
    return fig


def evidence_table(sup: dict) -> None:
    agents = ["N2 only", "CO2", "He"]
    ui.table(pd.DataFrame([{
        "O₂ in the atmosphere": row["o2_band"],
        **{AGENT_NAMES[ag]: (f"{row[ag]['share_went_out']:.0%} (n={row[ag]['n']})"
                             if row[ag]["share_went_out"] is not None else "no tests")
           for ag in agents},
    } for row in sup["table"]]), hide_index=True)


def misconceptions(items: list[dict]) -> None:
    """NASA's wording, verbatim. Misconceptions are struck through; the detection item is a finding."""
    for it in items:
        src = it["source"]["name"]
        with ui.card():
            if it["status"] == "misconception":
                st.markdown(f":red-badge[Misconception] ~~{it['claim']}~~")
            else:
                st.markdown(f":blue-badge[Finding] {it['claim']}")
            st.markdown(it["verdict"])
            st.caption(f"Quoted word for word from: {src}.")   # NASA's guide, not a PSI dataset: no PSI badge
