"""Shared bits for the layout proposals (#48): a compact condition-map preview and ranking helpers."""
import math

import plotly.graph_objects as go

from ai.habitats import HABITATS
from ai.ranking import rank
from app import data, ui

PHASE_COLOURS = {"solid": "#2a78d6", "droplet": "#eb6834", "gas": "#1baf7a"}   # same as the condition map


def habitat_cov(name: str) -> dict:
    return next((h for h in (data.coverage() or {}).get("habitats", []) if h["name"] == name), {})


def gaps_for(name: str) -> list[dict]:
    return [g for g in (data.coverage() or {}).get("gaps", []) if name in (g.get("habitats") or [])
            or name.lower() in g["text"].lower()]


def ranked(name: str) -> list[dict]:
    return rank(HABITATS[name])


def map_preview(height: int = 330, focus: str | None = "Lunar base") -> go.Figure:
    """Small O2-vs-pressure scatter with habitat diamonds; red box where no solid was tested."""
    invs = {i["id"]: i for i in data.investigations()}
    t = data.tests().copy()
    t["o2_role"] = data.o2_role(t)
    t = t.dropna(subset=["o2_frac", "pressure_kPa"])
    t = t[(t["pressure_kPa"] > 0) & (t["o2_role"] != "burner")]
    t["phase"] = t["investigation_id"].map(lambda x: invs.get(x, {}).get("fuel_phase") or "other")
    fig = go.Figure()
    for h in (data.coverage() or {}).get("habitats", []):
        if h["o2_frac"] >= 0.99:
            continue
        solid = (h.get("by_fuel_phase") or {}).get("solid", 0) + (
            ((h.get("o2_match_pressure_unknown") or {}).get("by_fuel_phase") or {}).get("solid", 0))
        fig.add_shape(type="rect", x0=h["pressure_kPa"] * 0.75, x1=h["pressure_kPa"] * 1.25,
                      y0=(h["o2_frac"] - 0.06) * 100, y1=(h["o2_frac"] + 0.06) * 100, layer="below",
                      line=dict(color="#e34948" if not solid else "rgba(150,150,150,0.6)", width=2),
                      fillcolor="rgba(227,73,72,0.15)" if not solid else "rgba(0,0,0,0)")
    for phase, colour in PHASE_COLOURS.items():
        d = t[t["phase"] == phase]
        if not d.empty:
            fig.add_scatter(x=d["pressure_kPa"], y=d["o2_frac"] * 100, mode="markers", name=phase,
                            marker=dict(size=6, color=colour, opacity=0.8), hoverinfo="skip")
    habs = [h for h in (data.coverage() or {}).get("habitats", []) if h["o2_frac"] < 0.99]
    fig.add_scatter(x=[h["pressure_kPa"] for h in habs], y=[h["o2_frac"] * 100 for h in habs], mode="markers",
                    name="habitat", text=[h["name"] for h in habs], hovertemplate="%{text}<extra></extra>",
                    marker=dict(symbol="diamond", size=11, color="#e8e8e3", line=dict(width=1.5, color="#1a1a19")))
    if focus:
        h = habitat_cov(focus)
        if h:
            fig.add_annotation(x=math.log10(h["pressure_kPa"]), y=h["o2_frac"] * 100, ax=math.log10(200), ay=62,
                               axref="x", ayref="y", text=f"<b>{focus}</b>: no solid-material test",
                               showarrow=True, arrowcolor="#e34948", font=dict(color="#ffffff", size=11),
                               bgcolor="rgba(179,38,30,0.95)", borderpad=3)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0, title=None),
                      xaxis=dict(type="log", title="Pressure (kPa)", range=[math.log10(25), math.log10(560)],
                                 tickvals=[30, 50, 70, 100, 150, 200, 300, 500],
                                 gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(title="O₂ %", range=[0, 70], gridcolor="rgba(128,128,128,0.15)"))
    return fig


def goto(label: str, icon: str) -> None:
    """Stand-in for st.page_link in the mock-up (the live pages are not registered in this app)."""
    import streamlit as st
    st.button(label, icon=icon, key=f"goto_{label}")


def label(pid: str) -> str:
    inv = data.investigation(pid)
    return data.label(inv) if inv else pid


__all__ = ["HABITATS", "data", "ui", "habitat_cov", "gaps_for", "ranked", "map_preview", "label", "goto"]
