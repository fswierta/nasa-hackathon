"""Fire scenario (#49): what would a fire do in this atmosphere? Calls analysis.scenario.assess only."""
import streamlit as st

from app import fire, ui

ui.page_header("Fire scenario", "**What would a fire do here?** Choose what is burning, the atmosphere and the "
                                "gravity level. Where NASA's experiments allow, a model estimates the outcome; "
                                "where they do not, this page says so and shows the evidence that exists.", ui.ICONS["scenario"])
fire.disclaimer_banner()

scenario = fire.scenario_form()
a = fire.assess(scenario)
if a is None:
    st.stop()

st.caption(f"Scenario: {fire.scenario_line(scenario)}")
ui.section("Gravity")
fire.gravity_block(a)

ui.section("Will it keep burning?")
fire.prediction_block(a)

st.divider()
fire.limits_and_sources(a)
st.caption(a["disclaimer"])
