"""Models hub (#58): every model the app ships, what it can and cannot do, and a playground to try one.
Written by main; all data comes from analysis/registry.py - no modelling in this page."""
import streamlit as st

from analysis import registry
from app import ml_ui, ui

ICON = ":material/hub:"
PAGE_OF = {"predict": "views/predict.py", "scenario": "views/scenario.py", "compare": "views/compare.py", "ranker": "views/ranker.py"}
PAGE_LABEL = {"predict": "Predict extinction", "scenario": "Fire scenario", "compare": "Compare", "ranker": "Scenario ranker"}

ui.page_header("Models", "Every model behind IGNIS in one place: what it was trained on, how well it does **next to a "
               "baseline that knows nothing**, what it must not be used for — and a playground to try it.", ICON)

models = registry.specs()
ml = [m for m in models if m["kind"] == "ml"]

# ---------------------------------------------------------------- what the models can and cannot do
c1, c2, c3, c4 = st.columns(4)
ui.metric_card(c1, "ML models", len(ml), help="Trained and cross-validated on PSI test tables.")
per_dataset = {}
for m in ml:
    per_dataset[m["dataset"]] = max(per_dataset.get(m["dataset"], 0), m["metrics"]["n_train"])
ui.metric_card(c2, "Tests they learned from", sum(per_dataset.values()),
               help="FLEX droplet tests + ACME CFI-G normal-flame tests (each dataset counted once).")
best = max((m for m in ml if "cv_accuracy" in m["metrics"]), key=lambda m: m["metrics"]["cv_accuracy"] - m["metrics"]["baseline_accuracy"])
ui.metric_card(c3, "Largest gain over baseline",
               f"+{(best['metrics']['cv_accuracy'] - best['metrics']['baseline_accuracy']) * 100:.1f} pts", help=best["title"])
ui.metric_card(c4, "Predictions for solids / Moon / Mars", "0", help="By design: PSI has no outcome data for solid materials "
               "and no partial-gravity data, so the models return a reason instead of a number.")

can, cannot = st.columns(2)
ui.callout("note", "**What the models can do:** estimate how a burning **fuel droplet** or **gas flame** behaves in a given "
           "atmosphere in microgravity (goes out? ignites? how fast?), inside the conditions NASA actually tested.", where=can)
ui.callout("limit", "**What they refuse to do:** predict for **solid materials** (fabrics, plastics), for **Moon or Mars "
           "gravity**, or outside the tested ranges. There the app shows the reason and the evidence that exists.", where=cannot)

# ---------------------------------------------------------------- one card per model
ui.section("The models")
cols = st.columns(2)
for n, m in enumerate(models):
    with cols[n % 2]:
        ml_ui.model_card(m, key="hub")
        b1, b2 = st.columns(2)
        if m["page"] in PAGE_OF and b1.button(f"Try it: {PAGE_LABEL[m['page']]}", key=f"try_{m['id']}", icon=ui.ICONS["open"]):
            st.switch_page(PAGE_OF[m["page"]])
        if m["kind"] == "ml" and b2.button("Open in ML Lab", key=f"lab_{m['id']}", icon=ui.ICONS["ml_lab"]):
            st.switch_page("views/ml_lab.py")

# ---------------------------------------------------------------- playground
ui.section("Try a model")
st.caption("Set the inputs; the model answers, shows how sure it can be, which inputs moved the result, and the closest real tests.")
pick = st.selectbox("Model", [m["id"] for m in ml], format_func=lambda i: next(m["title"] for m in ml if m["id"] == i))
model = next(m for m in ml if m["id"] == pick)

left, right = st.columns([1, 1.5], gap="large")
values = {}
with left, ui.card():
    st.markdown("**Inputs**")
    for i in model["inputs"]:
        if i["type"] == "choice":
            values[i["name"]] = st.selectbox(i["label"], i["choices"], index=i["choices"].index(i["default"]) if i["default"] in i["choices"] else 0,
                                             key=f"in_{pick}_{i['name']}")
        else:
            lo, hi = float(i["min"]), float(i["max"])
            step = round((hi - lo) / 100, 4) or 0.01
            values[i["name"]] = st.slider(f"{i['label']}", lo, hi, float(min(max(i["default"], lo), hi)), step=step,
                                          key=f"in_{pick}_{i['name']}", help=f"Tested range {lo:g} – {hi:g} {i['unit']}")
    st.caption("Sliders span the tested range over all fuels; a single fuel may cover less - then the model says so.")


@st.cache_data(show_spinner="Running the model…")
def _run(model_id: str, items: tuple):
    kw = dict(items)
    return registry.predict(model_id, **kw), registry.explain(model_id, **kw)


result, explanation = _run(pick, tuple(sorted(values.items())))
with right:
    ml_ui.prediction_card(result, model, key="play")

ui.section("Why this result?")
e1, e2 = st.columns(2)
with e1, ui.card():
    st.markdown("**For these inputs** (what-if against a typical test)")
    ml_ui.explanation_chart(explanation, key=f"explain_{pick}", unit="mm²/s" if model["headline"] == "k" else "percentage points")
with e2, ui.card():
    st.markdown("**Overall** (which inputs the model relies on)")
    ml_ui.importance_chart(model, key="play")

if model.get("reliability"):
    ui.section("Can the percentages be trusted?")
    with ui.card():
        ml_ui.reliability_chart(model, key="play")

ui.provenance_label("ml", "All numbers on this page come from analysis/registry.py: models trained on the PSI tables "
                    "(scikit-learn, cross-validated). The scenario ranking is a formula and is labelled as such.")
