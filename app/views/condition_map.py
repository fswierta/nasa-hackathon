import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data, ui

PHASE_COLOURS = ui.PHASE_COLOURS      # fuel phase colours are shared by every chart (app/ui.py)
TH = ui.chart_theme()                 # label / neutral colours for the current light or dark theme
MUTED = TH["muted"]
HABITAT_COLOUR = TH["neutral"]
KPA_TICKS = [30, 50, 70, 100, 150, 200, 300, 500]
PSI_TO_KPA = 6.894757
# Where each habitat's label goes, relative to its marker: (log10 kPa, O2 percentage points).
LABEL_OFFSETS = {"ISS": (0.28, -13), "Lunar base": (-0.27, 9), "Exploration atm.": (-0.22, 19),
                 "Spacesuit": (0.2, -4)}
SHORT_NAMES = {"Exploration atmosphere": "Exploration atm.", "Pressurized rover": "Rover"}

ui.page_header("Condition map", ("Every PSI test with a known **O₂ level and pressure**, against the atmospheres planned for crewed "
            "habitats. Empty space near a habitat = no direct evidence there."), ui.ICONS["condition_map"])

invs = {i["id"]: i for i in data.investigations()}
cov = data.coverage() or {}
t = data.tests().copy()
t["fuel_phase"] = t["investigation_id"].map(lambda x: invs.get(x, {}).get("fuel_phase") or "other")
t["investigation"] = t["investigation_id"].map(lambda x: data.label(invs[x]) if x in invs else x)
t["o2_role"] = data.o2_role(t)
placed = t.dropna(subset=["o2_frac", "pressure_kPa"])
placed = placed[placed["pressure_kPa"] > 0]
# Inverse flames: O2 is the burner stream, not the atmosphere -> never on the atmosphere axis by default.
burner = placed[placed["o2_role"] == "burner"]
plot = placed[placed["o2_role"] != "burner"]

# ?gap=solid (or the Home tour) switches the gap highlight on.
if st.query_params.get("gap") == "solid" and not st.session_state.get("_url_gap_done"):
    st.session_state["gap_hl"] = True
    st.session_state["_url_gap_done"] = True
t1, t2, t3 = st.columns([1.3, 1.3, 1])
gap_hl = t1.toggle("Highlight the solid-material gap", key="gap_hl",
                   help="Labels every habitat atmosphere that has no solid-material (fabric, plastic) test near it.")
# #58: the FLEX model's P(flame goes out), drawn only where real FLEX tests support it (flex_boundary.json).
fb = data.flex_boundary() or {}
overlay_fuels = sorted({g["fuel"] for g in fb.get("grids", [])})
ml_overlay = t2.toggle("Model overlay: P(droplet flame goes out)", key="ml_overlay", disabled=not overlay_fuels,
                       help="The FLEX model's estimate at the pressures FLEX tested, with no added CO₂ or helium. "
                            "Blank where no real test supports the model.")
overlay_fuel = (t3.segmented_control("Overlay fuel", overlay_fuels, default=overlay_fuels[0], key="ml_overlay_fuel")
                if ml_overlay and overlay_fuels else None) or (overlay_fuels[0] if overlay_fuels else None)

c1, c2, c3 = st.columns([2, 1.2, 1])
inv_opts = ["(none)"] + sorted(plot["investigation"].unique())
focus = c1.selectbox("Highlight one investigation", inv_opts,
                     help="Colours stay by fuel phase; other investigations are greyed out.")
phases = c2.multiselect("Fuel phases", list(PHASE_COLOURS), default=list(PHASE_COLOURS))
show_burner = c3.toggle("Burner O₂ (inverse)", value=False, key="burner",
                        help="Inverse flames blow O₂ out of the burner into a fuel atmosphere. That O₂ is "
                             "not the cabin air, so these points are hidden unless you ask for them.")

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


no_solid_evidence = data.no_solid_evidence   # red box = no solid test near, not even at a matching O2


gap_label_done = False
for h in habs:
    if h["o2_frac"] >= 0.99:   # spacesuit: pure O2, box would be off-scale
        continue
    no_solids = no_solid_evidence(h)
    strong = gap_hl and no_solids
    fig.add_shape(type="rect", x0=h["pressure_kPa"] * 0.75, x1=h["pressure_kPa"] * 1.25,
                  y0=(h["o2_frac"] - 0.06) * 100, y1=(h["o2_frac"] + 0.06) * 100,
                  line=dict(color=ui.RED if no_solids else "rgba(160,160,160,0.6)",
                            dash="solid" if strong else "dot", width=3 if strong else 1.5),
                  fillcolor=f"rgba(227,73,72,{0.2 if strong else 0.07})" if no_solids else "rgba(0,0,0,0)",
                  layer="below")
    if strong and not gap_label_done:
        # One callout in the empty upper-right area, pointing at the first red box's top-right corner.
        gap_label_done = True
        fig.add_annotation(x=math.log10(h["pressure_kPa"] * 1.25), y=(h["o2_frac"] + 0.06) * 100,
                           xref="x", yref="y", ax=math.log10(260), ay=64, axref="x", ayref="y",
                           text="<b>Red boxes: no solid-material test</b><br>(no fabric or plastic tested here)",
                           showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=ui.RED,
                           font=dict(size=12, color="#ffffff"), bgcolor="rgba(179,38,30,0.95)",  # both themes
                           bordercolor=ui.RED, borderwidth=1, borderpad=4)

overlay_grids = []
if ml_overlay and overlay_fuel:
    # One strip per tested pressure: the row with no added diluent (x = 0) of each pressure's grid. CO2 and He grids
    # share that row, so take one grid per pressure. Cells the model may not use (no real test nearby) stay blank.
    seen = set()
    for g in fb.get("grids", []):
        if g["fuel"] != overlay_fuel or g["pressure_kPa"] in seen:
            continue
        seen.add(g["pressure_kPa"])
        overlay_grids.append(g)
    scale = ui.seq_blue()
    for g in overlay_grids:
        o2 = g["o2_frac"]
        row = g["p_extinction"][g["x_diluent"].index(min(g["x_diluent"], key=abs))]
        edges = [o2[0] - (o2[1] - o2[0]) / 2] + [(a + b) / 2 for a, b in zip(o2, o2[1:])] + \
                [o2[-1] + (o2[-1] - o2[-2]) / 2]
        p = g["pressure_kPa"]
        for v, lo, hi in zip(row, edges, edges[1:]):
            if v is None:
                continue
            fig.add_shape(type="rect", x0=p / 1.12, x1=p * 1.12, y0=lo * 100, y1=hi * 100, layer="below",
                          line_width=0, fillcolor=ui.scale_colour(scale, v), opacity=0.9)
        ok = [(x, v) for x, v in zip(o2, row) if v is not None]
        fig.add_scatter(x=[p] * len(ok), y=[x * 100 for x, _ in ok], mode="markers", name="P(goes out), model",
                        showlegend=False, marker=dict(size=14, opacity=0, color=[v for _, v in ok], colorscale=scale,
                                                      cmin=0, cmax=1, showscale=g is overlay_grids[0],
                                                      colorbar=dict(title="P(goes<br>out)", tickformat=".0%",
                                                                    len=0.55, y=0.4, thickness=12)),
                        hovertemplate=f"<b>FLEX model</b> · {overlay_fuel} droplet, {p:g} kPa<br>"
                                      "O₂ %{y:.1f} % → P(flame goes out) %{marker.color:.0%}<extra></extra>")

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
        marker=dict(size=9, color=colour, opacity=0.8, line=dict(width=1, color=TH["neutral_line"])),
        customdata=d[["investigation", "test_id", "fuel", "outcome"]].fillna("—").values,
        hovertemplate="<b>%{customdata[0]}</b><br>test %{customdata[1]} · %{customdata[2]}<br>"
                      "O₂ %{y:.1f} % · %{x:.1f} kPa<br>outcome: %{customdata[3]}<extra></extra>",
    )

if show_burner and not burner.empty:
    fig.add_scatter(
        x=burner["pressure_kPa"], y=burner["o2_frac"] * 100, mode="markers",
        name="Burner O₂ (inverse flames, not the atmosphere)",
        marker=dict(symbol="x-thin", size=8, color="rgba(150,150,150,0.9)", line=dict(width=1.5,
                    color="rgba(150,150,150,0.9)")),
        customdata=burner[["investigation", "test_id", "fuel"]].fillna("—").values,
        hovertemplate="<b>%{customdata[0]}</b><br>test %{customdata[1]} · inverse flame, fuel atmosphere "
                      "%{customdata[2]}<br>burner O₂ %{y:.1f} % (not the ambient) · %{x:.1f} kPa<extra></extra>",
    )

if habs:
    fig.add_scatter(
        x=[h["pressure_kPa"] for h in habs], y=[h["o2_frac"] * 100 for h in habs], mode="markers",
        name="Habitat atmosphere", text=[h["name"] for h in habs],
        marker=dict(symbol="diamond", size=14, color=HABITAT_COLOUR, line=dict(width=2, color=TH["neutral_line"])),
        customdata=[[h["n_tests_near"], (h.get("by_fuel_phase") or {}).get("solid", 0), h.get("source", ""),
                     h["pressure_kPa"] / PSI_TO_KPA] for h in habs],
        hovertemplate="<b>%{text}</b><br>%{y:.0f} % O₂ · %{x:.1f} kPa (= %{customdata[3]:.1f} psi)<br>"
                      "%{customdata[0]} tests nearby, %{customdata[1]} with solids<br>%{customdata[2]}<extra></extra>",
    )
    # Labels sit in empty parts of the plot with a leader line, on a solid background, so they never
    # cover test points. Offsets are in axis units (log10 kPa, O2 %).
    for h in habs:
        dx, dy = LABEL_OFFSETS.get(h["name"].split(" / ")[0], (0.12, 6))
        fig.add_annotation(
            x=math.log10(h["pressure_kPa"]), y=h["o2_frac"] * 100, xref="x", yref="y",
            ax=math.log10(h["pressure_kPa"]) + dx, ay=h["o2_frac"] * 100 + dy, axref="x", ayref="y",
            text=h["name"], showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=TH["leader"],
            font=dict(size=12, color=TH["label_fg"]), bgcolor=TH["label_bg"],
            bordercolor=TH["label_border"], borderwidth=1, borderpad=3)

fig.update_layout(
    height=620, margin=dict(l=0, r=10, t=10, b=0),
    xaxis=dict(title="Pressure (kPa, log scale)", type="log", gridcolor=ui.GRID,
               tickvals=KPA_TICKS, ticktext=[f"{v:g}" for v in KPA_TICKS],
               range=[math.log10(25), math.log10(560)]),
    yaxis=dict(title="O₂ (% by volume)", range=[0, 105], gridcolor=ui.GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
)
n_solid = int((t["fuel_phase"] == "solid").sum())
n_solid_plotted = int((plot["fuel_phase"] == "solid").sum())
# One red banner for the solid-material gap (never two stacked): the highlight view names the habitats.
not_placed = (f"The {n_solid} solid-fuel test records (BASS, BASS-II, SAFFIRE, SAME…) have no pressure column in "
              "their PSI tables, so they cannot be placed on this map.") if n_solid and n_solid_plotted == 0 else ""
if gap_hl:
    red = [h["name"] for h in habs if h["o2_frac"] < 0.99 and no_solid_evidence(h)]
    max_solid = data.key_number("max_solid_o2_pct")
    ui.callout("gap", f"**The gap:** no fabric or plastic (solid-material) test in PSI is near the atmosphere of "
                      f"{', '.join(red) or 'any habitat'} (red boxes)"
                      + (f"; the highest O₂ any solid-fuel test used is {max_solid:g} %" if max_solid else "")
                      + ". Solids are the main fire load in a habitat, so these planned atmospheres have no direct "
                      f"evidence for them. {not_placed}")
elif not_placed:
    ui.callout("gap", f"**No solid-material test appears on this map.** {not_placed} BASS-II and SAFFIRE ran near "
                      "ISS cabin conditions (~17–22 % O₂) — below every planned Moon-base atmosphere.")

left, right = st.columns([2.3, 1], gap="large")
with left:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    ui.provenance_label("computed", f"{len(plot):,} of {len(t):,} test records have both an ambient O₂ level and a pressure. "
               f"{len(burner):,} inverse-flame records (ACME Flame Design, CFI-G) are "
               f"{'shown as grey ×' if show_burner else 'left off'}: their O₂ is the burner stream, not the "
               f"atmosphere. Dotted boxes: {near_txt.split(';')[0]} around each habitat; "
               "red = no solid-material test at this O₂ level at all. Spacesuit (100 % O₂) has no box.")
    if overlay_grids:
        d0 = overlay_grids[0].get("d0_mm")
        pressures = ", ".join(format(g["pressure_kPa"], "g") for g in overlay_grids)
        ui.provenance_label("ml", f"Blue strips: the FLEX model's probability that a burning {overlay_fuel.lower()} "
                                  f"droplet ({d0:g} mm) goes out on its own, at the pressures FLEX tested "
                                  f"({pressures} kPa) with no added CO₂ "
                                  "or helium, in microgravity. Blank = no real FLEX test close enough to support the "
                                  "model. Droplets only — not fabrics or plastics. Source: FLEX (PSI-69).")

with right:
    ui.section("Data gaps")
    if cov.get("gaps"):
        for g in cov["gaps"]:
            with ui.card():
                st.markdown(f"**{g['text']}**")
                if g.get("evidence"):
                    st.caption(data.partial_g_fix(g["evidence"]))
                if "partial-gravity" in g["text"].lower():
                    with st.expander("Why 0 g and 1 g don't bound the Moon"):
                        st.markdown(data.PARTIAL_G_NOTE)
                if g.get("closest_proxy"):
                    st.caption(f"Closest proxy: {g['closest_proxy']}")
    else:
        ui.empty_state("Coverage analysis (processed/coverage.json) not available yet.")

ui.section("Tests near each habitat atmosphere")
if cov.get("habitats"):
    ui.table(pd.DataFrame([{
        "Habitat": h["name"], "O₂ %": round(h["o2_frac"] * 100, 1), "Pressure kPa": h["pressure_kPa"],
        "Tests near": h["n_tests_near"],
        **{f"{p} tests": (h.get("by_fuel_phase") or {}).get(p, 0) for p in PHASE_COLOURS},
        "O₂ match, pressure unknown": (h.get("o2_match_pressure_unknown") or {}).get("n", 0),
        "Source": h.get("source"),
    } for h in cov["habitats"]]), hide_index=True)
    st.caption("'O₂ match, pressure unknown' = tests at a similar O₂ whose PSI table has no pressure column "
               "(e.g. BASS-II); they cannot be placed on the map.")
