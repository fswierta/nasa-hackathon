"""Response advisor (#49): what does the research say about fighting a fire in this atmosphere?
Calls analysis.scenario.response_brief only; the related passages come from ai.qa.ask()."""
import streamlit as st

from app import data, fire, ui

ui.page_header("Response advisor", "**There is a fire in this atmosphere. What does the research say about "
                                   "fighting it?** Evidence from NASA's microgravity experiments, with its limits.",
               ui.ICONS["response"])
# 1. Disclaimer first: research decision support, not a crew procedure.
fire.disclaimer_banner()

scenario = fire.scenario_form()
b = fire.brief(scenario)
if b is None:
    st.stop()
a = b["scenario"]
st.caption(f"Scenario: {fire.scenario_line(scenario)}")

# ---------------------------------------------------------------- 2. will it go out by itself?
ui.section("1 · Will it go out by itself?")
fire.gravity_block(a)
fire.prediction_block(a)

# ---------------------------------------------------------------- 3. suppressant gases
ui.section("2 · Suppressant gases")
sup = b["suppression"]
scans = b.get("suppressant_scans") or []
if scans:
    st.markdown("Estimated chance that the flame goes out when a suppressant gas is **released into the cabin**. "
                "Releasing gas dilutes everything else, so the O₂ level falls too (shown on hover).")
    for col, scan in zip(st.columns(len(scans), gap="large"), scans):
        with col:
            st.markdown(f"**{fire.AGENT_NAMES[scan['agent']]}**")
            st.plotly_chart(fire.scan_chart(scan), width="stretch", config={"displayModeBar": False})
            st.caption(fire.scan_summary(scan))
    ui.provenance_label("ml", f"{scans[0]['note']} Dotted line = 90 %. Grey = the engine gives no estimate there. "
                              f"Source: FLEX ({sup['source']['psi_id']}).")
    with st.expander("Chart data as a table"):
        ui.table([{"Agent": fire.AGENT_NAMES[s["agent"]], "Agent released (%)": round(p["agent_frac"] * 100, 1),
                       "Resulting O₂ (%)": round(p["o2_frac"] * 100, 1),
                       "P(flame goes out)": "outside tested range" if p["p_goes_out"] is None
                       else f"{p['p_goes_out']:.0%}"} for s in scans for p in s["points"]], hide_index=True)
else:
    ui.empty_state("The suppressant-gas estimate exists only for liquid droplets in microgravity, inside the range "
                   "FLEX tested. For this scenario the engine gives no estimate; the measured evidence is below.")

st.markdown("**What FLEX measured: share of droplet flames that went out**")
fire.evidence_table(sup)
st.markdown(sup["headline"])
st.markdown("\n".join(f"- {c}" for c in sup["caveats"]))
ui.provenance_label("computed", f"Counted from the FLEX test table ({sup['source']['psi_id']}, DOI "
                                f"{ui.source_link(sup['source']['doi'])}). n = number of tests in the group.")

# ---------------------------------------------------------------- 4. don't rely on these
ui.section("3 · Don't rely on these")
st.caption("Assumptions that NASA lists as wrong, and one finding about detection. NASA's wording, unchanged.")
fire.misconceptions(b["misconceptions"])

# ---------------------------------------------------------------- 5. related evidence (optional)
ui.section("4 · Related evidence in the PSI documents")
what = {"droplet": f"{scenario['fuel']} droplet", "gas": f"{scenario['fuel']} gas flame"}.get(
    scenario["fuel_class"], scenario["fuel"])
question = f"How was a {what} fire suppressed or extinguished in microgravity?"
st.markdown(f"Question sent to the document search: *{question}*")
found = st.session_state.setdefault("fire_evidence", {})
if st.button("Search the PSI documents", icon=ui.ICONS["search"], key="fire_search"):
    data.wait_for_index()
    with st.spinner("Searching the PSI documents…"):
        try:
            from ai.qa import ask
            found[question] = ask(question)
        except Exception as e:   # the page works without the AI layer
            found[question] = {"answer": f"The document search is not available here ({type(e).__name__}).",
                               "citations": []}
res = found.get(question)
if res:
    lab = data.answer_label(res)
    if lab:
        ui.provenance_label("ai" if lab[0] == "✨" else "psi", f"{lab[0]} {lab[1]}")
    st.markdown(res.get("answer") or "No passage found.")
    for n, c in enumerate(res.get("citations") or [], 1):
        inv = data.investigation(c.get("psi_id")) if c.get("psi_id") else None
        name = data.label(inv) if inv else (c.get("psi_id") or "Reference")
        page = f", p. {c['page']}" if c.get("page") else ""
        doi = f" · DOI {ui.source_link(c['doi'])}" if c.get("doi") else ""
        st.caption(f"[{n}] **{name}** — {(c.get('file') or '').split('/')[-1]}{page}{doi}")

st.divider()
fire.limits_and_sources(a, extra_sources=[sup["source"]] + [m["source"] for m in b["misconceptions"]])
st.caption(b["disclaimer"])
