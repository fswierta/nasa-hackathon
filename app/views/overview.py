import pandas as pd
import streamlit as st

from app import data
from app.nav import APP_DIR

DETAIL = "views/detail.py"

st.title("🗂️ Overview")
st.caption("All Combustion Science investigations in NASA PSI. Select a row to open its detail page.")

invs = data.investigations()


def _opts(key):
    return sorted({i.get(key) for i in invs if i.get(key)})


with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    phase = c1.multiselect("Fuel phase", _opts("fuel_phase"))
    tier = c2.multiselect("Relevance tier", ["A", "B", "C"],
                          help="A = core spacecraft fire safety, B = related, C = other combustion")
    hw = c3.multiselect("Hardware family", _opts("hardware_family"))
    kind = c4.multiselect("Type", _opts("type"), help="flight = microgravity on ISS/Cygnus; "
                                                     "ground = study re-using flight data")
    c5, c6 = st.columns([3, 1])
    o2 = c5.slider("Tested O₂ range overlaps (% by volume)", 0, 100, (0, 100), step=1)
    keep_no_o2 = c6.checkbox("Include investigations with no O₂ data", value=True)
    query = st.text_input("Search title, acronym, fuels, keywords", placeholder="e.g. PMMA, droplet, smoke")


def match(i: dict) -> bool:
    if phase and i.get("fuel_phase") not in phase:
        return False
    if tier and i.get("relevance_tier") not in tier:
        return False
    if hw and i.get("hardware_family") not in hw:
        return False
    if kind and i.get("type") not in kind:
        return False
    rng = i.get("o2_range")
    if rng:
        if rng[1] * 100 < o2[0] or rng[0] * 100 > o2[1]:
            return False
    elif not keep_no_o2:
        return False
    if query:
        hay = " ".join(str(i.get(k) or "") for k in ("id", "acronym", "title", "keywords", "fuels"))
        if query.lower() not in hay.lower():
            return False
    return True


rows = [i for i in invs if match(i)]
df = pd.DataFrame([{
    "id": i["id"],
    "Investigation": i.get("acronym") or "(ground study)",
    "Title": i.get("title"),
    "Tier": i.get("relevance_tier"),
    "Type": i.get("type"),
    "Fuel phase": i.get("fuel_phase") or "—",
    "Hardware": i.get("hardware_family") or "—",
    "Fuels": ", ".join(i.get("fuels") or []) or "—",
    "O₂ %": data.fmt_range(i.get("o2_range"), scale=100, digits=1),
    "Pressure kPa": data.fmt_range(i.get("pressure_range_kPa"), digits=0),
    "Flow cm/s": data.fmt_range(i.get("flow_range_cm_s"), digits=1),
    "Table rows": data.table_rows(i["id"])[0],
    "DOI": data.doi_url(i.get("doi")),
} for i in rows])

st.markdown(f"**{len(rows)}** of {len(invs)} investigations")
if df.empty:
    st.info("No investigation matches these filters.")
    st.stop()

event = st.dataframe(
    df, hide_index=True, on_select="rerun", selection_mode="single-row", key="overview_table",
    column_config={
        "id": st.column_config.TextColumn("PSI ID", width="small"),
        "Investigation": st.column_config.TextColumn(width=230),
        "Title": st.column_config.TextColumn(width=520, help="Full title also on the detail page"),
        "Tier": st.column_config.TextColumn(width="small"),
        "Table rows": st.column_config.NumberColumn(help="Rows parsed from the PSI experimental table "
                                                    "(tests, or test days for ACME/SLICE logs)"),
        "DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)"),
    },
)
st.caption("'no data' = the PSI experimental table has no such column (not zero).")

sel = event.selection.rows if event and event.selection else []
if sel:
    pid = df.iloc[sel[0]]["id"]
    st.session_state["detail_id"] = pid
    if (APP_DIR / DETAIL).exists():
        st.switch_page(DETAIL)
