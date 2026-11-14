"""ML components for every page (#58, parts 1-2; written by main, built on app/ui.py).

    from app import ml_ui
    ml_ui.model_card(model)                      # one registry entry: what it is, CV score vs baseline, scope
    ml_ui.prediction_card(result, model)         # headline number, class bars, tested-range gauges, nearest tests
    ml_ui.explanation_chart(explanation)         # which inputs push the prediction up / down
    ml_ui.importance_chart(model); ml_ui.reliability_chart(model)
    ml_ui.no_prediction(reason)                  # the designed "no data" state (a result, not an error)

`model` = an entry of analysis.registry.specs(); `result` = analysis.registry.predict(...);
`explanation` = analysis.registry.explain(...). No modelling happens here.
Colours follow app/ui.py: blue / magenta / aqua for data, flame orange only as the accent.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import ui

# Violet means "AI-written" (language model) everywhere else, so unsupervised ML stays in the orange ML family.
KIND_BADGE = {"ml": ui.badge("ml"), "unsupervised": ":orange-badge[Unsupervised ML]",
              "formula": ":gray-badge[Transparent formula — not ML]"}
HEADLINE_LABEL = {"p_goes_out": "Flame goes out on its own", "p_hot_ignition": "Hot flame ignites",
                  "p_self_extinguish_if_ignited": "Goes out by itself (if ignited)", "k": "Burning rate K"}
ICON = ":material/model_training:"
COLUMN = {"o2_frac": "O₂ (mole fr.)", "pressure_kPa": "Pressure (kPa)", "x_co2": "CO₂ (mole fr.)",
          "x_he": "He (mole fr.)", "d0_mm": "Diameter (mm)", "fuel": "Fuel", "fuel_frac": "Fuel share",
          "hc_rate_mg_s": "Fuel flow (mg/s)"}


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    t = ui.chart_theme()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=t["label_fg"], size=12), showlegend=False, **kw)
    fig.update_xaxes(gridcolor=ui.GRID, zerolinecolor=t["muted"])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def is_ml(model: dict) -> bool:
    return model.get("kind") == "ml"


def score_text(model: dict) -> str:
    """One line with the model's cross-validated score next to its baseline."""
    m = model.get("metrics", {})
    if "cv_accuracy" in m:
        return f"accuracy {m['cv_accuracy'] * 100:.1f} % vs {m['baseline_accuracy'] * 100:.1f} % baseline"
    if "r2" in m:
        return f"R² {m['r2']:.2f} (baseline 0)"
    if "clusters" in m:
        return f"{m['clusters']} clusters · {m['related_links_found']} of {m['related_links_total']} PSI links recovered"
    return "weights shown and adjustable"


def score_chart(model: dict, where=None, key: str | None = None) -> None:
    """Model vs baseline as two bars (classification) - the honest way to read an accuracy."""
    m = model.get("metrics", {})
    if "cv_accuracy" not in m:
        return
    vals = [m["baseline_accuracy"] * 100, m["cv_accuracy"] * 100]
    fig = go.Figure(go.Bar(x=vals, y=["Baseline: always the most common outcome", "This model (cross-validated)"],
                           orientation="h", marker_color=[ui.chart_theme()["muted"], ui.BLUE],
                           text=[f"{v:.1f} %" for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    _layout(fig, 110, xaxis=dict(range=[0, 112], ticksuffix=" %", tickvals=[0, 25, 50, 75, 100]))
    (where or st).plotly_chart(fig, width="stretch", config={"displayModeBar": False},
                               key=key or f"score_{model['id']}")


def model_card(model: dict, where=None, compact: bool = False, key: str = ""):
    """What the model is, how good it is next to its baseline, what it is for - and its limits."""
    box = ui.card(where)
    with box:
        st.markdown(f"**{ICON} {model['title']}**  \n{KIND_BADGE.get(model['kind'], '')}")
        src = model.get("source") or {}
        data = f"{src.get('name', '')} ({src['psi_id']})" if src.get("psi_id") else src.get("name", "")
        n = model.get("metrics", {}).get("n_train")
        st.caption(f"{model['algorithm']} · trained on {n} tests from {data} {ui.source_link(src.get('doi'), 'DOI')}"
                   if is_ml(model) and n else f"{model['algorithm']}")
        if is_ml(model) and "cv_accuracy" in model["metrics"]:
            score_chart(model, key=f"score_{model['id']}_{key}")
        else:
            st.markdown(f"**{score_text(model)}**")
        st.caption(model["scope"])
        if not compact:
            st.markdown("\n".join(f"- {limit}" for limit in model["limits"]))
    return box   # `with ml_ui.model_card(m):` adds buttons inside the same card


def no_prediction(reason: str, where=None, title: str = "No prediction") -> None:
    """'The data does not allow a number here' - shown as a result with its reason, never as an error."""
    ui.callout("limit", f"**{title}.** {reason}", where=where)


def range_gauges(result: dict, model: dict, where=None, key: str = "") -> None:
    """Where each numeric input sits inside the range the model was trained on (red marker = outside)."""
    nums = [i for i in model["inputs"] if i["type"] == "number"]
    if not nums:
        return
    t = ui.chart_theme()
    fig = go.Figure()
    for row, i in enumerate(nums):
        v = float(result["inputs"].get(i["name"], i["default"]))
        span = (i["max"] - i["min"]) or 1.0
        pos = (v - i["min"]) / span
        inside = -1e-9 <= pos <= 1 + 1e-9
        fig.add_trace(go.Scatter(x=[0, 1], y=[row, row], mode="lines", line=dict(color=t["muted"], width=6), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[min(max(pos, -0.08), 1.08)], y=[row], mode="markers",
                                 marker=dict(size=14, color=ui.BLUE if inside else ui.RED, line=dict(color=t["neutral_line"], width=1.5)),
                                 hovertemplate=f"{i['label']}: {v:g} (tested {i['min']:g} – {i['max']:g})<extra></extra>"))
    _layout(fig, 34 * len(nums) + 30, xaxis=dict(range=[-0.12, 1.12], tickvals=[0, 1], ticktext=["lowest tested", "highest tested"]),
            yaxis=dict(tickvals=list(range(len(nums))), ticktext=[f"{i['label']} = {float(result['inputs'].get(i['name'], i['default'])):g}" for i in nums],
                       autorange="reversed"))
    (where or st).plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"gauge_{model['id']}_{key}")


def prediction_card(result: dict, model: dict, where=None, key: str = "") -> None:
    """One prediction: the headline, class probabilities, tested-range gauges and the nearest real tests."""
    box = ui.card(where)
    with box:
        st.markdown(f"**{model['title']}**  \n{KIND_BADGE.get(model['kind'], '')}")
        if not result.get("ok"):
            no_prediction(result.get("reason") or "The model gives no estimate for these inputs.")
        else:
            label = HEADLINE_LABEL.get(result["headline"], result["headline"])
            pred = result["prediction"]
            if result["headline"] == "k":
                ui.metric_card(st, label, f"{result['value']:.2f} mm²/s",
                               help="d²-law burning-rate constant; the interval is ±1.96 × cross-validated error.")
                st.caption(f"interval {pred['k_low']:.2f} – {pred['k_high']:.2f} mm²/s")
            else:
                ui.metric_card(st, label, f"{result['value'] * 100:.0f} %",
                               help="A model estimate from cross-validated training on real tests - read it with the "
                                    "accuracy below.")
            probs = pred.get("probs") if isinstance(pred, dict) else None
            if probs:
                names = {"extinction": "Flame went out", "disruption": "Test ended early", "completion": "Droplet burned completely"}
                fig = go.Figure(go.Bar(x=[probs[c] * 100 for c in names], y=list(names.values()), orientation="h",
                                       marker_color=[ui.OUTCOME_COLOURS[c] for c in names],
                                       text=[f"{probs[c] * 100:.0f} %" for c in names], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x:.0f} %<extra></extra>"))
                _layout(fig, 140, xaxis=dict(range=[0, 112], ticksuffix=" %", tickvals=[0, 25, 50, 75, 100]),
                        yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"probs_{model['id']}_{key}")
            ui.provenance_label("ml", f"{model['algorithm']} · {score_text(model)} · n = {model['metrics'].get('n_train')}")
        st.markdown("**Are these inputs inside what was tested?**")
        range_gauges(result, model, key=key)
        near = result.get("nearest_tests") or []
        if near:
            st.markdown("**Closest real tests** (what actually happened)")
            ui.table([{"Test": t["test_id"], "What happened": t["actual"],
                       **{COLUMN.get(k, k): v for k, v in t["conditions"].items()}} for t in near])
            src = model.get("source") or {}
            ui.provenance_label("psi", f"Rows from the {src.get('name', 'PSI')} table {ui.source_link(src.get('doi'), 'DOI')}")


def explanation_chart(explanation: list[dict], where=None, key: str = "explain", unit: str = "percentage points") -> None:
    """What-if bars: how much each input moves the prediction compared with a typical (median) test."""
    w = where or st
    if not explanation:
        ui.empty_state("No explanation: the model gave no prediction for these inputs.", where=w)
        return
    scale = 100 if unit == "percentage points" else 1
    ex = sorted(explanation, key=lambda e: abs(e["effect"]))
    fig = go.Figure(go.Bar(
        x=[e["effect"] * scale for e in ex],
        y=[f"{e['label']} = {e['value']:g}" if isinstance(e["value"], (int, float)) else f"{e['label']} = {e['value']}" for e in ex],
        orientation="h", marker_color=[ui.BLUE if e["effect"] >= 0 else ui.MAGENTA for e in ex],
        text=[f"{e['effect'] * scale:+.0f}" if scale == 100 else f"{e['effect']:+.2f}" for e in ex], textposition="outside", cliponaxis=False,
        customdata=[e["reference"] for e in ex], hovertemplate="%{y}<br>vs %{customdata}: %{x:+.1f}<extra></extra>"))
    _layout(fig, 40 * len(ex) + 50, xaxis=dict(title=f"change in the prediction ({unit})", zeroline=True))
    w.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
    w.caption("How to read: each bar compares your value with a typical test (the training median for this fuel). "
              "Blue pushes the prediction up, magenta pushes it down. A what-if on the model, not a law of physics.")


def importance_chart(model: dict, where=None, key: str = "") -> None:
    """Which inputs the model relies on overall (permutation importance)."""
    imp = [i for i in model.get("importance", [])]
    w = where or st
    if not imp:
        return
    imp = sorted(imp, key=lambda i: i["value"])
    fig = go.Figure(go.Bar(x=[i["value"] for i in imp], y=[i["label"] for i in imp], orientation="h", marker_color=ui.BLUE,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    _layout(fig, 34 * len(imp) + 50, xaxis=dict(title=model.get("importance_note") or "drop in balanced accuracy when this input is shuffled"))
    w.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"imp_{model['id']}_{key}")


def reliability_chart(model: dict, where=None, key: str = "") -> None:
    """Predicted probability vs how often it really happened (out-of-fold). On the diagonal = trustworthy."""
    rel = model.get("reliability") or []
    w = where or st
    if not rel:
        return
    t = ui.chart_theme()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=t["muted"], dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["p_predicted"] * 100 for r in rel], y=[r["p_observed"] * 100 for r in rel], mode="lines+markers",
                             marker=dict(size=[8 + min(r["n"], 120) / 8 for r in rel], color=ui.BLUE), line=dict(color=ui.BLUE),
                             customdata=[r["n"] for r in rel],
                             hovertemplate="model said %{x:.0f} % → happened in %{y:.0f} % of %{customdata} tests<extra></extra>"))
    _layout(fig, 260, xaxis=dict(title=f"model's {model.get('reliability_of', 'probability')} (%)", range=[0, 100]),
            yaxis=dict(title="how often it happened (%)", range=[0, 100], gridcolor=ui.GRID))
    w.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"rel_{model['id']}_{key}")
    w.caption("Each point = tests the model gave a similar probability (held-out folds); bigger = more tests. "
              "Points near the dotted line mean the percentages can be read as percentages.")
