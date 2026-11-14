import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data

# Fuel phase colours: validated categorical slots 1-3 (scatter: max 3 hues, all pairs distinct).
PHASE_COLOURS = {"solid": "#2a78d6", "droplet": "#eb6834", "gas": "#1baf7a"}
MUTED = "rgba(150,150,150,0.35)"
HABITAT_COLOUR = "#e8e8e3"
SHORT_NAMES = {"Exploration atmosphere": "Exploration atm.", "Pressurized rover": "Rover"}

st.title("🗺️ Condition map")
st.markdown("Every PSI test with a known **O₂ level and pressure**, against the atmospheres planned for crewed "
            "habitats. Empty space near a habitat = no direct evidence there.")

invs = {i["id"]: i for i in data.investigations()}
cov = data.coverage() or {}
t = data.tests().copy()
t["fuel_phase"] = t["investigation_id"].map(lambda x: invs.get(x, {}).get("fuel_phase") or "other")
t["investigation"] = t["investigation_id"].map(lambda x: data.label(invs[x]) if x in invs else x)
plot = t.dropna(subset=["o2_frac", "pressure_kPa"])
plot = plot[plot["pressure_kPa"] > 0]

c1, c2 = st.columns([2, 1])
inv_opts = ["(none)"] + sorted(plot["investigation"].unique())
focus = c1.selectbox("Highlight one investigation", inv_opts,
                     help="Colours stay by fuel phase; other investigations are greyed out.")
phases = c2.multiselect("Fuel phases", list(PHASE_COLOURS), default=list(PHASE_COLOURS))

fig = go.Figure()

# Habitat "near" boxes (same definition as coverage.json): O2 ±0.06, pressure ±25 %.
defn = cov.get("definition")
near_txt = defn if isinstance(defn, str) else "O₂ within ±0.06 mole fraction and pressure within ±25 %"

# Habitats sharing one atmosphere (Exploration atm / Mars base / rover) become one marker.
groups: dict[tuple, dict] = {}
for h in cov.get("habitats", []):
    key = (h["o2_frac"], h["pressure_kPa"])
    short = SHORT_NAMES.get(h["name"], h["name"])
    if key in groups:
        groups[key]["name"] += " / " + short
    else:
        groups[key] = {**h, "name": short}
habs = list(groups.values())


def no_solid_evidence(h: dict) -> bool:
    """Red box only when there is no solid test near this atmosphere at all — not even one at a
    matching O2 whose table lacks pressure (ISS: BASS-II/SAFFIRE ran at ~21 % O2, pressure unrecorded)."""
    near = (h.get("by_fuel_phase") or {}).get("solid", 0)
    o2_only = ((h.get("o2_match_pressure_unknown") or {}).get("by_fuel_phase") or {}).get("solid", 0)
    return near == 0 and o2_only == 0


for h in habs:
    if h["o2_frac"] >= 0.99:   # spacesuit: pure O2, box would be off-scale
        continue
    no_solids = no_solid_evidence(h)
    fig.add_shape(type="rect", x0=h["pressure_kPa"] * 0.75, x1=h["pressure_kPa"] * 1.25,
                  y0=(h["o2_frac"] - 0.06) * 100, y1=(h["o2_frac"] + 0.06) * 100,
                  line=dict(color="#e34948" if no_solids else "rgba(160,160,160,0.6)", dash="dot", width=1.5),
                  fillcolor="rgba(227,73,72,0.07)" if no_solids else "rgba(0,0,0,0)", layer="below")

for phase, colour in PHASE_COLOURS.items():
    if phase not in phases:
        continue
    d = plot[plot["fuel_phase"] == phase]
    if focus != "(none)":
        rest, d = d[d["investigation"] != focus], d[d["investigation"] == focus]
        if not rest.empty:
            fig.add_scatter(x=rest["pressure_kPa"], y=rest["o2_frac"] * 100, mode="markers",
                            marker=dict(size=7, color=MUTED), name=f"{phase} (others)", showlegend=False,
                            hoverinfo="skip")
    if d.empty:
        continue
    fig.add_scatter(
        x=d["pressure_kPa"], y=d["o2_frac"] * 100, mode="markers", name=phase,
        marker=dict(size=9, color=colour, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.8)")),
        customdata=d[["investigation", "test_id", "fuel", "outcome"]].fillna("—").values,
        hovertemplate="<b>%{customdata[0]}</b><br>test %{customdata[1]} · %{customdata[2]}<br>"
                      "O₂ %{y:.1f} % · %{x:.1f} kPa<br>outcome: %{customdata[3]}<extra></extra>",
    )

if habs:
    fig.add_scatter(
        x=[h["pressure_kPa"] for h in habs], y=[h["o2_frac"] * 100 for h in habs], mode="markers+text",
        name="Habitat atmosphere", text=[h["name"] for h in habs], textposition="middle right",
        textfont=dict(size=12), marker=dict(symbol="diamond", size=14, color=HABITAT_COLOUR,
                                            line=dict(width=2, color="#1a1a19")),
        customdata=[[h["n_tests_near"], (h.get("by_fuel_phase") or {}).get("solid", 0), h.get("source", "")]
                    for h in habs],
        hovertemplate="<b>%{text}</b><br>%{y:.0f} % O₂ · %{x:.1f} kPa<br>%{customdata[0]} tests nearby, "
                      "%{customdata[1]} with solids<br>%{customdata[2]}<extra></extra>",
    )

fig.update_layout(
    height=620, margin=dict(l=0, r=10, t=10, b=0),
    xaxis=dict(title="Pressure (kPa, log scale)", type="log", gridcolor="rgba(128,128,128,0.15)"),
    yaxis=dict(title="O₂ (% by volume)", range=[0, 105], gridcolor="rgba(128,128,128,0.15)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
)
left, right = st.columns([2.3, 1], gap="large")
with left:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{len(plot):,} of {len(t):,} test records have both O₂ and pressure. Dotted boxes: "
               f"{near_txt.split(';')[0]} around each habitat; "
               "red = no solid-material test at this O₂ level at all. Spacesuit (100 % O₂) has no box.")
    n_solid = int((t["fuel_phase"] == "solid").sum())
    n_solid_plotted = int((plot["fuel_phase"] == "solid").sum())
    if n_solid and n_solid_plotted == 0:
        st.info(f"**No solid-material test appears on this map.** The {n_solid} solid-fuel test records "
                "(BASS, BASS-II, SAFFIRE, SAME…) have no pressure column in their PSI tables, so they cannot be "
                "placed here. BASS-II and SAFFIRE ran near ISS cabin conditions (~17–22 % O₂).", icon="ℹ️")

with right:
    st.subheader("Data gaps")
    if cov.get("gaps"):
        for g in cov["gaps"]:
            with st.container(border=True):
                st.markdown(f"**{g['text']}**")
                if g.get("evidence"):
                    st.caption(g["evidence"])
                if g.get("closest_proxy"):
                    st.caption(f"Closest proxy: {g['closest_proxy']}")
    else:
        st.info("Coverage analysis (processed/coverage.json) not available yet.")

st.subheader("Tests near each habitat atmosphere")
if cov.get("habitats"):
    st.dataframe(pd.DataFrame([{
        "Habitat": h["name"], "O₂ %": round(h["o2_frac"] * 100, 1), "Pressure kPa": h["pressure_kPa"],
        "Tests near": h["n_tests_near"],
        **{f"{p} tests": (h.get("by_fuel_phase") or {}).get(p, 0) for p in PHASE_COLOURS},
        "O₂ match, pressure unknown": (h.get("o2_match_pressure_unknown") or {}).get("n", 0),
        "Source": h.get("source"),
    } for h in cov["habitats"]]), hide_index=True)
    st.caption("'O₂ match, pressure unknown' = tests at a similar O₂ whose PSI table has no pressure column "
               "(e.g. BASS-II); they cannot be placed on the map.")
