"""Home (#55): dashboard-first. A habitat picker drives the gap banner, the map preview and the top-5
evidence list; under it the 3-step demo tour (#42) and the page index."""
import math

import plotly.graph_objects as go
import streamlit as st

from ai.habitats import HABITATS
from ai.ranking import rank
from app import data, ml_ui, ui
from app.nav import APP_NAME, APP_TAGLINE, CHALLENGE, grouped

invs = data.investigations()
cov = data.coverage() or {}

ui.page_header(APP_NAME, icon=ui.ICONS["home"])
st.caption(f"**{APP_TAGLINE}** · {CHALLENGE}")
st.markdown(
    "**Fire behaves differently in space.** Without gravity there is no buoyancy: hot gas does not rise, "
    "flames turn spherical and live on airflow from fans. This dashboard **summarizes, ranks and interprets** "
    f"the {len(invs)} combustion investigations in NASA's PSI archive, so engineers can find the evidence that "
    "matters for a spacecraft, the ISS, a Moon or Mars base, or a pressurized rover."
)

# ---------------------------------------------------------------- habitat picker: drives everything below
habitat = st.segmented_control("Your habitat", list(HABITATS), default="Lunar base", key="home_habitat",
                               help="Changes the gap banner, the map highlight and the evidence list.") or "Lunar base"
h = HABITATS[habitat]
hc = data.habitat_coverage(habitat)
st.caption(f"{habitat}: {h['o2_frac'] * 100:g} % O₂ · {h['pressure_kPa']:g} kPa · "
           f"{data.GRAVITY_OPTIONS.get(h.get('gravity'), 'gravity not set')} · source: {h.get('source', '—')}")

# ---------------------------------------------------------------- headline numbers
ml_models = [m for m in data.ml_specs() if m.get("kind") == "ml"]
cols = st.columns(5)
for col, (label, key, fallback, hlp) in zip(cols, [
    ("Investigations", "investigations", len(invs),
     "Combustion Science investigations in NASA PSI (flight + ground studies)"),
    ("Test records", "test_records", len(data.tests()),
     "Rows parsed from the PSI experimental tables (processed/tests.csv)"),
    ("Documents indexed", "documents_indexed", data.doc_stats()["n_documents"],
     "Science documents and reports the AI searches and cites"),
    ("Data gaps found", "data_gaps", len(cov.get("gaps", [])),
     "Conditions with no PSI test (processed/coverage.json)"),
]):
    v = data.key_number(key, fallback)
    ui.metric_card(col, label, f"{v:,}" if isinstance(v, int) else (v or "—"), help=hlp)
ui.metric_card(cols[4], "ML models", len(ml_models) or "—",
               help="Models trained and cross-validated on the PSI test tables (FLEX droplets, ACME CFI-G gas flames).")
cols[4].page_link("views/models.py", label="See the models", icon=ui.ICONS["models"])

# ---------------------------------------------------------------- gap banner for this habitat
gaps = data.gaps_for(habitat)
if gaps:
    g = gaps[0]
    # #65 item 3: the closest evidence goes with the gap. Report rows (ACME BRE, #63 item 2) once main adds them,
    # unless main's gap text already names them; until then coverage.json's closest proxy.
    rep = data.report_rows()
    if not rep.empty and "BRE" not in g["text"]:
        closest = ("**Closest evidence: ACME BRE's gas burner emulating a solid (not a real fabric or plastic sample),** "
                   + "; ".join(data.report_row_line(r) for _, r in rep.iterrows()) + ".")
    elif g.get("closest_proxy") and "BRE" not in g["text"]:
        closest = f"**Closest evidence:** {g['closest_proxy']}"
    else:
        closest = ""
    ui.callout("gap", f"**Data gap for {habitat}:** {g['text']}  \n{data.partial_g_fix(g.get('evidence', ''))}"
                      + (f"  \n{closest}" if closest else ""))
elif hc:
    phases = hc.get("by_fuel_phase") or {}
    o2_only = ((hc.get("o2_match_pressure_unknown") or {}).get("by_fuel_phase") or {}).get("solid", 0)
    ui.callout("note", f"**{habitat}:** {hc.get('n_tests_near', 0)} PSI tests are near this atmosphere "
                       f"({phases.get('solid', 0)} solid, {phases.get('droplet', 0)} droplet, {phases.get('gas', 0)} gas). "
                       f"{o2_only} solid-material tests match its O₂ level but their PSI table has no pressure column.")


# ---------------------------------------------------------------- what would a fire do here? (ML, #58)
def _pct(x) -> str:
    return f"{x * 100:.0f} %"


def _inputs_line(inputs: dict) -> str:
    keep = {k: v for k, v in inputs.items() if k not in ("fuel", "o2_frac", "pressure_kPa")}
    return ", ".join(f"{ml_ui.COLUMN.get(k, k).lower()} {v:g}" for k, v in keep.items() if isinstance(v, (int, float)))


hp = data.habitat_prediction(habitat)
# #65 item 1: these are models of specific NASA microgravity experiments (FLEX droplets, CFI-G burner flames), so the
# section says exactly that, never "what a fire would do". The refusals below are part of the answer.
ui.section("What do models trained on NASA microgravity tests say at this O₂ and pressure?")
if not hp:
    ui.empty_state("Model estimates are not available (processed/analysis/habitat_predictions.json is missing).")
else:
    mg = hp.get("in_microgravity") or {}
    st.caption(f"{habitat}: {hp['o2_frac'] * 100:g} % O₂ and {hp['pressure_kPa']:g} kPa. Each model answers only for the "
               "experiment it was trained on, **in microgravity**; each value is a number or a reason, never both.")
    show_gas = ml_ui.gas_framing_ready()     # CFI-G cards wait for the reviewed model names (#64 item 4)
    cols = st.columns(3 if show_gas else 2)
    with ui.card(cols[0]):
        st.markdown(f"**{ml_ui.ICON} Fuel droplet (FLEX experiment)**  \n{ml_ui.KIND_BADGE['ml']}")
        for r in mg.get("droplet_goes_out", []):
            if r.get("value") is not None:
                ui.metric_card(st, f"{r['fuel']} droplet: flame goes out on its own", _pct(r["value"]),
                               help=f"FLEX random forest (a ~2.5 mm droplet); inputs: {_inputs_line(r['inputs'])}.")
            else:
                ml_ui.no_prediction(r["reason"], title=f"{r['fuel']}: no prediction")
    if show_gas:
        with ui.card(cols[1]):
            gas_specs = {"gas_ignites": ml_ui.spec("cfig_ignition"), "gas_goes_out": ml_ui.spec("cfig_self_extinguish")}
            st.markdown(f"**{ml_ui.ICON} Gas burner flame (ACME CFI-G experiment)**  \n{ml_ui.KIND_BADGE['ml']}")
            for key, model in gas_specs.items():
                label = ml_ui.headline_label(model, model["headline"])
                for r in mg.get(key, []):
                    if r.get("value") is not None:
                        ui.metric_card(st, f"{r['fuel'].capitalize()}: {label[0].lower()}{label[1:]}", _pct(r["value"]),
                                       help=f"{model['title']} Inputs: {_inputs_line(r['inputs'])}.")
                    else:
                        ml_ui.no_prediction(r["reason"], title=f"{r['fuel'].capitalize()}: no prediction")
    with ui.card(cols[-1]):
        st.markdown("**Fabrics and plastics (solids)**  \n:gray-badge[No model: no outcome data]")
        sol = hp.get("solids") or {}
        reason = (sol.get("reason") or "PSI has no outcome data for solid materials.").replace(
            "(except Saffire-II below)", "(except the Saffire-II comparison, shown on Fire scenario)")
        ml_ui.no_prediction(reason, title="No data, so no prediction for solids")
        if sol.get("gap"):
            st.caption(f":material/report: {sol['gap']}")
    grav = (hp.get("at_habitat_gravity") or {}).get("statement")
    if grav:
        ml_ui.no_prediction(grav, title=f"At {data.GRAVITY_OPTIONS.get(hp.get('gravity'), 'this')} gravity: "
                                        "no prediction")
    ui.provenance_label("ml", "Random forests trained on FLEX (PSI-69) and ACME CFI-G (PSI-159) tests, cross-validated; "
                              "see Models for each score next to its baseline.")
    if st.button("Explore this in Fire scenario", icon=ui.ICONS["scenario"], key="home_fire"):
        st.session_state["fs_preset"] = habitat
        st.switch_page("views/scenario.py")

# ---------------------------------------------------------------- map preview | top evidence
def map_preview(focus: str) -> go.Figure:
    """Small O2-vs-pressure scatter with the habitat atmospheres; red box = no solid-material test near."""
    th = ui.chart_theme()
    by_id = {i["id"]: i for i in invs}
    t = data.tests().copy()
    t["o2_role"] = data.o2_role(t)
    t = t.dropna(subset=["o2_frac", "pressure_kPa"])
    t = t[(t["pressure_kPa"] > 0) & (t["o2_role"] != "burner")]
    t["phase"] = t["investigation_id"].map(lambda x: by_id.get(x, {}).get("fuel_phase") or "other")
    habs = [x for x in cov.get("habitats", []) if x["o2_frac"] < 0.99]      # spacesuit (100 % O2) is off the scale
    fig = go.Figure()
    for x in habs:
        red, on = data.no_solid_evidence(x), x["name"] == focus
        fig.add_shape(type="rect", x0=x["pressure_kPa"] * 0.75, x1=x["pressure_kPa"] * 1.25, layer="below",
                      y0=(x["o2_frac"] - 0.06) * 100, y1=(x["o2_frac"] + 0.06) * 100,
                      line=dict(color=ui.RED if red else th["muted"], width=3 if on else 1,
                                dash="solid" if on else "dot"),
                      fillcolor="rgba(227,73,72,0.18)" if red and on else "rgba(0,0,0,0)")
    for phase, colour in ui.PHASE_COLOURS.items():
        d = t[t["phase"] == phase]
        if not d.empty:
            fig.add_scatter(x=d["pressure_kPa"], y=d["o2_frac"] * 100, mode="markers", name=phase,
                            marker=dict(size=7, color=colour, opacity=0.85), hoverinfo="skip")
    fig.add_scatter(x=[x["pressure_kPa"] for x in habs], y=[x["o2_frac"] * 100 for x in habs], mode="markers",
                    name="habitat", text=[x["name"] for x in habs], hovertemplate="%{text}<extra></extra>",
                    marker=dict(symbol="diamond", size=11, color=th["neutral"],
                                line=dict(width=1.5, color=th["neutral_line"])))
    x = data.habitat_coverage(focus)
    if x and x["o2_frac"] < 0.99:
        red = data.no_solid_evidence(x)
        fig.add_annotation(
            x=math.log10(x["pressure_kPa"]), y=x["o2_frac"] * 100, ax=math.log10(230), ay=60, axref="x", ayref="y",
            text=f"<b>{focus}</b>: " + ("no solid-material test" if red else f"{x.get('n_tests_near', 0)} tests near"),
            showarrow=True, arrowhead=2, arrowcolor=ui.RED if red else th["leader"], borderpad=4,
            font=dict(color="#ffffff" if red else th["label_fg"], size=12),
            bgcolor="rgba(179,38,30,0.95)" if red else th["label_bg"],
            bordercolor=ui.RED if red else th["label_border"], borderwidth=1)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=1.1, x=0, title=None),
                      xaxis=dict(type="log", title="Pressure (kPa)", range=[math.log10(25), math.log10(560)],
                                 tickvals=[30, 50, 70, 100, 150, 200, 300, 500], gridcolor=ui.GRID),
                      yaxis=dict(title="O₂ (% by volume)", range=[0, 70], gridcolor=ui.GRID))
    return fig


left, right = st.columns([1.6, 1], gap="large")
with ui.card(left):
    st.markdown("**Tested conditions vs habitat atmospheres**")
    if h["o2_frac"] >= 0.99:
        st.caption(f"{habitat} (100 % O₂) is off this chart's scale; nothing in PSI was tested there.")
    st.plotly_chart(map_preview(habitat), width="stretch", config={"displayModeBar": False})
    ui.provenance_label("computed", "Ambient O₂ and pressure from the PSI tables. Boxes: O₂ within ±6 points and "
                                    "pressure within ±25 % of a habitat; red = no solid-material test near.")
    if st.button("Open the condition map", icon=ui.ICONS["condition_map"], key="home_map"):
        st.session_state["gap_hl"] = True
        st.switch_page("views/condition_map.py")
with ui.card(right):
    st.markdown(f"**Top evidence for {habitat}**")
    near = data.near_atmosphere(h["o2_frac"], h["pressure_kPa"])   # same "near" as the gap banner (#65 item 4)
    for n, r in enumerate(rank(h)[:5], 1):   # the ranker's own function, at its default weights
        inv = data.investigation(r["id"])
        info = near.get(r["id"]) or {}
        st.markdown(f"{n}. **{data.label(inv) if inv else r['id']}** · score {r['score']:.2f}  \n"
                    f"<small>{data.near_text(info)}</small>", unsafe_allow_html=True)
    ui.provenance_label("computed", "Scenario ranker with its default weights; open it for the reason behind "
                                    "each score.")
    if st.button("Open the scenario ranker", icon=ui.ICONS["ranker"], key="home_ranker"):
        st.session_state["preset"] = habitat
        st.switch_page("views/ranker.py")

# ---------------------------------------------------------------- guided tour (#42)
# Same story as submission/video_script.md: rank -> see the gap -> ask. Each button preloads the page
# through session state (works on every Streamlit version); the pages also accept URL parameters.
TOUR_QUESTION = "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?"
TOUR = [
    ("1", "Rank the evidence for a Moon base", ui.ICONS["ranker"],
     "Pick a habitat and IGNIS ranks the 24 investigations by how well their test conditions match it, "
     "with the reason behind every score.",
     "views/ranker.py", {"preset": "Lunar base"}),
    ("2", "See what was never tested", ui.ICONS["condition_map"],
     "The planned Moon-base atmosphere has no solid-material test in PSI, yet fabrics and plastics are "
     "the main fire load in a habitat.",
     "views/condition_map.py", {"gap_hl": True}),
    ("3", "Ask NASA's documents", ui.ICONS["ask"],
     "Ask in plain words; every answer cites the PSI investigation, DOI, file and page.",
     "views/ask.py", {"pending": TOUR_QUESTION}),
]
ui.section("Start here: a 3-step tour")
for col, (n, title, icon, why, page, state) in zip(st.columns(3), TOUR):
    with ui.card(col):
        st.markdown(f"**{n} · {title}**")
        st.caption(why)
        if st.button(f"Step {n}", icon=icon, key=f"tour_{n}", type="primary" if n == "1" else "secondary",
                     width="stretch"):
            st.session_state.update(state)
            st.switch_page(page)

# ---------------------------------------------------------------- page index, same groups as the sidebar
for section, specs in grouped().items():
    specs = [sp for sp in specs if sp[0] != "views/home.py"]
    if not specs:
        continue
    ui.section(section)
    cols = st.columns(3)
    for n, (path, title, icon, desc) in enumerate(specs):
        with cols[n % 3]:
            st.page_link(path, label=title, icon=icon)
            st.caption(desc)

tiers = {t: sum(i["relevance_tier"] == t for i in invs) for t in "ABC"}
st.caption(
    f"Relevance tiers: A (core spacecraft fire safety) {tiers['A']} · B (related) {tiers['B']} · "
    f"C (other combustion) {tiers['C']}. Every investigation links to its PSI DOI."
)
