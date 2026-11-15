"""Scenario ranker (#55, dashboard-first): the scenario in one bar on top, headline numbers, then the
ranking with the reason behind every score. Scores come from ai.ranking.rank (deterministic, no AI)."""
import plotly.graph_objects as go
import streamlit as st

from ai.habitats import HABITATS
from ai.ranking import DEFAULT_WEIGHTS, PILLAR_FOCUS_WEIGHT, rank
from app import data, ml_ui, ui

# Score components, in the order the breakdown table shows them.
COMPONENTS = [("o2", "O₂ overlap"), ("pressure", "Pressure overlap"), ("flow", "Airflow overlap"),
              ("material", "Material match"), ("pillar", "Fire-safety pillar"), ("tier", "Relevance tier"),
              ("richness", "Data richness")]
NAMES = dict(COMPONENTS)
FIELD_LABEL = {"o2_frac": "O₂", "pressure_kPa": "pressure", "flow_cm_s": "airflow",
               "materials": "materials", "gravity": "gravity"}
MATERIALS = sorted({m for h in HABITATS.values() for m in h["materials"]} |
                   {"PMMA", "Nomex", "silicone", "Kapton", "Teflon", "Ultem", "cotton/fiberglass fabric"},
                   key=str.lower)

ui.page_header("Scenario ranker", "Describe a crewed environment and see **which PSI experiments are the most "
                                  "relevant evidence** for its fire safety, with the reason behind every score.",
               ui.ICONS["ranker"])

# ?habitat=Mars+base preselects a preset (shareable link); Home sets st.session_state.preset.
if st.query_params.get("habitat") in HABITATS and not st.session_state.get("_url_habitat_done"):
    st.session_state["preset"] = st.query_params["habitat"]
    st.session_state["_url_habitat_done"] = True
if st.session_state.get("preset") not in HABITATS:
    st.session_state["preset"] = "Lunar base"

# ---------------------------------------------------------------- scenario bar
with ui.card():
    c = st.columns([1.2, 1, 1, 1, 1.5], gap="medium")
    preset = c[0].selectbox("Habitat preset", list(HABITATS), key="preset")
    h = HABITATS[preset]
    k = preset  # widget keys per preset -> switching preset resets the fields to its values
    o2_pct = c[1].slider("O₂ (% by volume)", 10.0, 100.0, float(h["o2_frac"] * 100), 0.5, key=f"o2_{k}")
    p_kpa = c[2].slider("Cabin pressure (kPa)", 5.0, 400.0, float(h["pressure_kPa"]), 0.1, key=f"p_{k}",
                        help="101.3 kPa = 14.7 psi (sea level)")
    flow = c[3].slider("Ventilation airflow (cm/s)", 0.0, 50.0, float(h["flow_cm_s"]), 0.5, key=f"f_{k}")
    grav_keys = list(data.GRAVITY_OPTIONS)
    gravity = c[4].radio("Gravity", grav_keys, horizontal=True, key=f"g_{k}",
                         index=grav_keys.index(h.get("gravity")) if h.get("gravity") in grav_keys else 0,
                         format_func=data.GRAVITY_OPTIONS.get,
                         help="Every PSI flight test ran in microgravity; PSI has no Moon or Mars gravity tests.")
    c = st.columns([2.2, 1, 2.5], gap="medium")
    mats = c[0].multiselect("Materials in the cabin", MATERIALS,
                            default=[m for m in h["materials"] if m in MATERIALS], key=f"m_{k}")
    pillar = c[1].selectbox("Focus", ["any", "prevention", "detection", "suppression"], key=f"pl_{k}",
                            help="Prevention = ignition & flammability · detection = smoke sensing · "
                                 "suppression = extinguishing")
    with c[2].expander("Scoring weights"):
        st.caption("Each component scores 0–1; the total is the weighted average. "
                   f"Choosing a focus raises the pillar weight to {PILLAR_FOCUS_WEIGHT}.")
        if st.button("Reset weights", icon=ui.ICONS["reset"]):
            for key in DEFAULT_WEIGHTS:
                for focused in (True, False):
                    st.session_state.pop(f"w_{key}_{focused}", None)
        base = dict(DEFAULT_WEIGHTS)
        if pillar != "any":
            base["pillar"] = PILLAR_FOCUS_WEIGHT
        weights = {key: st.slider(NAMES[key], 0.0, 1.0, float(base[key]), 0.05, key=f"w_{key}_{pillar != 'any'}")
                   for key, _ in COMPONENTS}
    assumed = [FIELD_LABEL.get(a, a) for a in h.get("assumptions", [])]
    st.caption(f"= {p_kpa / 6.894757:.1f} psi · **Preset source:** {h.get('source', '—')}"
               + (f" · **Assumed, not from PSI data:** {', '.join(assumed)}." if assumed else ""))

scenario = {**h, "o2_frac": o2_pct / 100, "pressure_kPa": p_kpa, "flow_cm_s": flow,
            "materials": mats, "pillar": None if pillar == "any" else pillar, "gravity": gravity}
results = rank(scenario, weights)
label = {r["id"]: data.label(data.investigation(r["id"])) for r in results}

# ---------------------------------------------------------------- headline numbers
# #65 item 4: one definition of "near" everywhere (coverage.json's), at the atmosphere set above.
near = data.near_atmosphere(o2_pct / 100, p_kpa)
phase_of = {i["id"]: i.get("fuel_phase") for i in data.investigations()}
k1, k2, k3, k4 = st.columns(4)
ui.metric_card(k1, "Best match", f"{results[0]['score']:.2f}", help=label[results[0]["id"]])
ui.metric_card(k2, "Investigations scoring ≥ 0.5", sum(r["score"] >= 0.5 for r in results),
               help="Out of the 24 PSI combustion investigations.")
ui.metric_card(k3, "Tests near this atmosphere", sum(v["n"] for v in near.values()),
               help="PSI tests with O₂ within ±6 points and pressure within ±25 % of the atmosphere set above (ambient "
                    "O₂ only; rows with a range count if it overlaps) — the same count as the gap banner.")
ui.metric_card(k4, "…with solid materials", sum(v["n"] for pid, v in near.items() if phase_of.get(pid) == "solid"),
               help="Fabrics and plastics are the main fire load in a habitat.")

if gravity in ("lunar", "mars"):
    with ui.card():
        st.markdown(f"**{data.GRAVITY_OPTIONS[gravity]}: no PSI test at this gravity.** "
                    f"{data.PARTIAL_G_SHORT} Rankings below are microgravity evidence only.")
        with st.expander("Sources"):
            st.markdown(data.PARTIAL_G_NOTE)
elif gravity == "earth":
    st.caption("1 g: PSI flight tests ran in microgravity, so they show how flames differ from 1 g ground "
               "screening (NASA-STD-6001), not 1 g behaviour itself.")
for gap in data.gaps_for(preset)[:1]:
    ui.callout("gap", f"**{gap['text']}**")

# ---------------------------------------------------------------- evidence ranking (formula) | ML view (models)
# Two different tools side by side, each labelled: the ranking scores which experiments are relevant evidence
# (a transparent formula); the models estimate what a flame would do in this atmosphere (#58).
rank_col, ml_col = st.columns([1.7, 1], gap="large")
with rank_col:
    st.markdown(f"**Evidence ranking**  \n{ml_ui.KIND_BADGE['formula']}")
    top_n = st.segmented_control("Show", [5, 10, 24], default=10, key="top_n",
                                 format_func=lambda n: f"Top {n}" if n < 24 else "All 24") or 10
    shown = results[:top_n]
    fig = go.Figure(go.Bar(
        x=[r["score"] for r in shown][::-1], y=[label[r["id"]] for r in shown][::-1], orientation="h",
        marker=dict(color=ui.PALETTE["flame"], cornerradius=4), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_layout(height=90 + 30 * len(shown), margin=dict(l=0, r=10, t=10, b=0), bargap=0.3,
                      xaxis=dict(range=[0, 1], title="Relevance score (0–1)", gridcolor=ui.GRID),
                      yaxis=dict(automargin=True))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    picked = ui.table([{"#": n, "Investigation": label[r["id"]], "Score": round(r["score"], 2),
                        "Tests near this atmosphere": data.near_text(near.get(r["id"])), "Why": r["why"]}
                       for n, r in enumerate(shown, 1)],
                      height=90 + 30 * min(len(shown), 10), key="rank_table", on_select="rerun",
                      selection_mode="single-row", column_config={"Why": st.column_config.TextColumn(width="large")})
    ui.provenance_label("computed", "Scores are deterministic (no AI, no ML), so every number can be traced. "
                                    "Select a row for its breakdown.")
    if "gravity" not in DEFAULT_WEIGHTS:
        st.caption("Gravity is not yet a scoring component: every PSI flight test is microgravity, so it adds "
                   "a data-gap warning instead of changing scores.")


def _first_ok(model_id: str, fuels: list[str], **kw) -> str:
    """The first fuel the model gives an estimate for here (the picker still offers every fuel)."""
    return next((f for f in fuels if ml_ui.predict(model_id, fuel=f, **kw)["ok"]), fuels[0])


with ml_col:
    st.markdown(f"**{ml_ui.ICON} ML view of this atmosphere**  \n{ml_ui.KIND_BADGE['ml']}")
    st.caption(f"What models trained on specific NASA microgravity experiments estimate at {o2_pct:g} % O₂ and "
               f"{p_kpa:g} kPa. They predict an experiment's outcome; the ranking beside it scores evidence.")
    if gravity in ("lunar", "mars"):
        ml_ui.no_prediction(f"{data.PARTIAL_G_SHORT} The estimates below are for microgravity only.",
                            title=f"No prediction at {data.GRAVITY_OPTIONS[gravity]}")
    elif gravity == "earth":
        ml_ui.no_prediction("The models are trained on microgravity tests and are not valid at 1 g. The estimates "
                            "below are for microgravity only.", title="No prediction at 1 g")
    atm = {"o2_frac": round(o2_pct / 100, 4), "pressure_kPa": float(p_kpa)}
    drop_fuels = next(i["choices"] for i in ml_ui.spec("flex_outcome")["inputs"] if i["name"] == "fuel")
    drop = st.segmented_control("Droplet fuel", drop_fuels, key="rk_drop_fuel",
                                default=_first_ok("flex_outcome", drop_fuels, **atm)) or drop_fuels[0]
    ml_ui.prediction_card(ml_ui.predict("flex_outcome", fuel=drop, **atm), ml_ui.spec("flex_outcome"), compact=True,
                          key="rk_drop", title=f"FLEX experiment: does a burning {drop.lower()} droplet go out?")
    if ml_ui.gas_framing_ready():   # #65 item 1: CFI-G burner cards wait for the reviewed model names (#64 item 4)
        gas_fuels = next(i["choices"] for i in ml_ui.spec("cfig_ignition")["inputs"] if i["name"] == "fuel")
        gas = st.segmented_control("Gas fuel", gas_fuels, key="rk_gas_fuel",
                                   default="propane" if "propane" in gas_fuels else gas_fuels[0]) or gas_fuels[0]
        g_res = ml_ui.predict("cfig_ignition", fuel=gas, **atm)
        with ml_ui.prediction_card(g_res, ml_ui.spec("cfig_ignition"), compact=True, key="rk_gas",
                                   title=f"{ml_ui.spec('cfig_ignition')['title']} ({gas})"):
            if g_res["ok"]:
                se = ml_ui.spec("cfig_self_extinguish")
                st.caption(f"{ml_ui.headline_label(se, se['headline'])}: "
                           f"**{g_res['prediction']['p_self_extinguish_if_ignited'] * 100:.0f} %** "
                           f"(second model, {ml_ui.score_text(se)}).")
    if st.button("How good are these models?", icon=ui.ICONS["models"], key="rk_models", type="tertiary"):
        st.switch_page("views/models.py")

# ---------------------------------------------------------------- the selected investigation
rows = (picked.selection.rows if picked and getattr(picked, "selection", None) else []) or [0]
r = shown[min(rows[0], len(shown) - 1)]
inv = data.investigation(r["id"])
ui.section(f"{label[r['id']]} · score {r['score']:.2f}")
with ui.card():
    st.markdown(f"**{inv.get('title')}**")
    st.write(r["why"])
    ui.provenance_label("computed", f"At {o2_pct:g} % O₂ / {p_kpa:g} kPa: {data.near_text(near.get(r['id']))} "
                                    "(near = O₂ ±6 points and pressure ±25 %, as in the gap banner).")
    ui.table([{NAMES.get(key, key): v for key, v in r["breakdown"].items()}])
    st.caption("Each component scores 0–1. '—' = no data: the PSI table has no such column; it contributes "
               "nothing and is never treated as a match.")
    for g in r["gaps"]:
        if "partial-gravity" in g and gravity in ("lunar", "mars"):
            continue   # shown once above the results, with sources
        st.caption(f":material/report: {data.partial_g_fix(g)}")
    b1, b2 = st.columns([1, 3])
    if b1.button("Open detail", icon=ui.ICONS["open"], key=f"open_{r['id']}"):
        st.session_state["detail_id"] = r["id"]
        st.switch_page("views/detail.py")
    if r.get("doi"):
        b2.markdown(ui.source_link(r["doi"], "DOI " + r["doi"]))

with st.expander(f"Score breakdown of all {len(shown)} shown"):
    ui.table([{"Investigation": label[x["id"]], "Score": round(x["score"], 2),
               **{NAMES.get(key, key): v for key, v in x["breakdown"].items()}} for x in shown])
