import pandas as pd
import streamlit as st

from app import data

st.title("⚖️ Compare investigations")
st.caption("Pick 2–4 investigations. Rows where they differ are marked ≠.")

invs = data.investigations()
ids = [i["id"] for i in invs]
default = [x for x in ("PSI-26", "PSI-25", "PSI-99") if x in ids]
picked = st.multiselect("Investigations", ids, default=default, max_selections=4,
                        format_func=lambda x: data.label(data.investigation(x)))
if len(picked) < 2:
    st.info("Select at least two investigations.")
    st.stop()

tests = data.tests()


def join(v):
    return ", ".join(v) if v else "—"


def outcomes(pid: str) -> str:
    o = tests.loc[tests["investigation_id"] == pid, "outcome"].dropna()
    if o.empty:
        return "—"
    return ", ".join(f"{k} ({n})" for k, n in o.value_counts().head(5).items())


ROWS = [
    ("Title", lambda i: i.get("title") or "—"),
    ("Relevance tier", lambda i: i.get("relevance_tier") or "—"),
    ("Type · platform", lambda i: f"{i.get('type') or '—'} · {i.get('platform') or '—'}"),
    ("Hardware family", lambda i: i.get("hardware_family") or "—"),
    ("Fuel phase", lambda i: i.get("fuel_phase") or "—"),
    ("Fuels / materials", lambda i: join(i.get("fuels"))),
    ("Sample geometry", lambda i: join(i.get("sample_geometry"))),
    ("O₂ (%)", lambda i: data.fmt_range(i.get("o2_range"), scale=100, digits=1)),
    ("Pressure (kPa)", lambda i: data.fmt_range(i.get("pressure_range_kPa"), digits=1)),
    ("Airflow (cm/s)", lambda i: data.fmt_range(i.get("flow_range_cm_s"), digits=1)),
    ("Diluents", lambda i: join(i.get("diluents"))),
    ("Varied between tests", lambda i: join(i.get("varied_variables"))),
    ("Held fixed", lambda i: join(i.get("fixed_variables"))),
    ("Table rows", lambda i: "{} {}".format(*data.table_rows(i["id"]))),
    ("Recorded outcomes", lambda i: outcomes(i["id"])),
    ("Publications", lambda i: str(len(i.get("publications") or []))),
    ("DOI", lambda i: i.get("doi") or "—"),
]

sel = [data.investigation(p) for p in picked]
table = {"": [], "≠": []}
for p in picked:
    table[data.label(data.investigation(p))] = []
for name, fn in ROWS:
    vals = [fn(i) for i in sel]
    table[""].append(name)
    table["≠"].append("≠" if len(set(vals)) > 1 else "")
    for p, v in zip(picked, vals):
        table[data.label(data.investigation(p))].append(v)
df = pd.DataFrame(table)

only_diff = st.toggle("Show only rows that differ")
if only_diff:
    df = df[df["≠"] == "≠"]
st.dataframe(df, hide_index=True, height=38 + 35 * len(df),
             column_config={"": st.column_config.TextColumn(width="medium"),
                            "≠": st.column_config.TextColumn(width="small")})

st.subheader("Summaries")
cols = st.columns(len(sel))
for col, i in zip(cols, sel):
    with col:
        st.markdown(f"**{data.label(i)}**")
        prof = data.profile(i["id"])
        if prof and prof.get("summary_plain"):
            st.write(prof["summary_plain"])
            st.caption("🤖 AI summary — see the detail page for sources.")
        else:
            st.write((i.get("objective") or "No objective on record.")[:600])
            st.caption("Objective from PSI metadata.")
        if i.get("doi"):
            st.markdown(f"[DOI {i['doi']}]({data.doi_url(i['doi'])})")
