"""Data explorer (#34): every row of processed/tests.csv with readable units, the corrections main's ETL
made (flags) next to the original PSI row, a CSV download, the untouched PSI table and PSI's file list."""
import json

import pandas as pd
import streamlit as st

from app import data

# (tests.csv column, label, scale) — numbers keep their type so the table sorts numerically.
COLUMNS = [
    ("test_id", "Test", None), ("row_kind", "Row is", None), ("date_iso", "Date", None),
    ("fuel", "Fuel", None), ("material", "Material (PSI text)", None), ("geometry", "Geometry", None),
    ("o2_frac", "O₂ %", 100), ("o2_frac_max", "O₂ max %", 100), ("o2_role", "O₂ is", None),
    ("pressure_kPa", "Pressure kPa", 1), ("pressure_kPa_max", "Pressure max kPa", 1),
    ("flow_cm_s", "Flow cm/s", 1), ("flow_cm_s_max", "Flow max cm/s", 1), ("diluent", "Diluent", None),
    ("outcome", "Outcome", None), ("burn_time_s", "Burn time s", 1), ("spread_rate_mm_s", "Spread rate mm/s", 1),
    ("extinction_diameter_mm", "Extinction Ø mm", 1), ("d0_mm", "Initial Ø mm", 1),
    ("burning_rate_k_mm2_s", "Burning rate K mm²/s", 1), ("thickness_mm", "Thickness mm", 1),
    ("flow_direction", "Flow direction", None), ("notes", "Notes", None), ("source_file", "Source file", None),
]
ROW_KINDS = {"test": "one test", "test_day": "test day (several tests)", "test_matrix": "planned test matrix",
             "session": "sampling session (no fire)"}
# Fix ids documented in etl/fixes.py (main's ETL round 2).
FLAG_LABELS = {
    "F1": "thickness unit cm → mm (SAFFIRE)", "F2": "missing decimal point in thickness",
    "F3": "empty merged material cell filled", "F4": "BASS material name normalised",
    "F5": "BASS-II rod material → PMMA", "F6": "Excel auto-increment undone",
    "F7": "duplicate test id numbered", "F8": "remark moved to notes",
    "F9": "odd value, kept as published", "F10": "year typo in date fixed",
    "D1": "date year taken from a PSI document", "D2": "date conflict, kept as published",
}

st.title("🧮 Data explorer")
st.markdown("Every test record we parsed from the PSI experimental tables, in common units. Rows that main's "
            "ETL **corrected or flagged** carry a ⚠; pick one to see what changed, why, and the original PSI row.")

invs = data.investigations()
by_id = {i["id"]: i for i in invs}
tests = data.tests().copy()
if "flags" not in tests:
    tests["flags"] = None
if "date_iso" not in tests:
    tests["date_iso"] = None
tests["o2_role"] = data.o2_role(tests)
tests["flags"] = tests["flags"].where(tests["flags"].fillna("").str.strip() != "")
tests["flag_ids"] = tests["flags"].fillna("").map(
    lambda s: sorted({x.split(":")[0].strip() for x in s.split(" | ") if x.strip()}))
with_rows = [i["id"] for i in invs if i["id"] in set(tests["investigation_id"])]

# ---------------------------------------------------------------- filters
with st.container(border=True):
    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("Investigations", with_rows, placeholder="All investigations with a table",
                            format_func=lambda x: data.label(by_id[x]))
    df = tests[tests["investigation_id"].isin(picked)] if picked else tests
    kinds = c2.multiselect("Row kind", [k for k in ROW_KINDS if k in set(df["row_kind"])],
                           format_func=ROW_KINDS.get, placeholder="All kinds")
    if kinds:
        df = df[df["row_kind"].isin(kinds)]

    c1, c2 = st.columns([3, 2])
    fuels = c1.multiselect("Fuel", sorted(df["fuel"].dropna().unique(), key=str.lower), placeholder="All fuels")
    if fuels:
        df = df[df["fuel"].isin(fuels)]
    dates = pd.to_datetime(tests["date_iso"], errors="coerce")
    with c2:
        if dates.notna().any():
            lo, hi = dates.min().date(), dates.max().date()
            rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
            keep_undated = st.checkbox("include rows with no date", True, key="nodate")
            if isinstance(rng, tuple) and len(rng) == 2:
                d = dates.loc[df.index]
                inside = (d >= pd.Timestamp(rng[0])) & (d <= pd.Timestamp(rng[1]))
                df = df[inside | (d.isna() & keep_undated)]

    def range_filter(rows, col, label, scale, step, key):
        """Keep rows whose [value, value_max] overlaps the slider; rows without a value only if allowed."""
        v = pd.to_numeric(tests[col], errors="coerce") * scale
        vmax = pd.to_numeric(tests[f"{col}_max"], errors="coerce") * scale
        top = pd.concat([v, vmax], axis=1).max(axis=1)
        full = (float(v.min()), float(top.max()))
        sel = st.slider(label, full[0], full[1], full, step=step, key=key)
        keep_none = st.checkbox("include rows with no data", True, key=f"{key}_none")
        lo_v, hi_v = v.loc[rows.index], top.loc[rows.index]
        return rows[((lo_v <= sel[1]) & (hi_v >= sel[0])) | (lo_v.isna() & keep_none)]

    c1, c2, c3 = st.columns(3)
    with c1:
        df = range_filter(df, "o2_frac", "O₂ (%)", 100, 0.5, "f_o2")
    with c2:
        df = range_filter(df, "pressure_kPa", "Pressure (kPa)", 1, 1.0, "f_p")
    with c3:
        df = range_filter(df, "flow_cm_s", "Airflow (cm/s)", 1, 0.5, "f_flow")

    c1, c2 = st.columns([1, 2])
    only_flagged = c1.toggle("Only rows with corrections", key="only_flagged")
    if only_flagged:
        counts = pd.Series([f for ids in df["flag_ids"] for f in ids]).value_counts()
        which = c2.multiselect("Correction types", list(counts.index), placeholder="Any correction",
                               format_func=lambda f: f"{f}: {FLAG_LABELS.get(f, '')} ({counts[f]})")
        df = df[df["flag_ids"].map(lambda ids: bool(set(ids) & set(which)) if which else bool(ids))]

st.caption(f"**{len(df):,} of {len(tests):,} rows** match.")

tab_rows, tab_raw, tab_files = st.tabs(["📋 Cleaned rows", "📄 Original PSI table", "📁 Files in PSI"])

# ---------------------------------------------------------------- cleaned rows
with tab_rows:
    show = pd.DataFrame(index=df.index)
    show["⚠"] = df["flags"].notna().map({True: "⚠", False: ""})
    show["Investigation"] = df["investigation_id"].map(lambda x: data.label(by_id[x]) if x in by_id else x)
    num_cols = []
    for col, name, scale in COLUMNS:
        if col not in df:
            continue
        if scale is None:
            show[name] = df[col].replace({"": None})
        else:
            show[name] = pd.to_numeric(df[col], errors="coerce") * scale
            num_cols.append(name)
    show["Row is"] = show["Row is"].map(lambda k: ROW_KINDS.get(k, k))
    show = show.dropna(axis=1, how="all")
    num_cols = [c for c in num_cols if c in show]

    # Sorted here, on the real numbers; cells are then shown as text so a missing value reads "—" (Streamlit
    # renders nulls as "None", and a number column cannot hold "—").
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    sort_by = c1.selectbox("Sort by", ["PSI table order"] + [c for c in show if c != "⚠"], key="sort_by")
    desc = c2.toggle("Descending", key="sort_desc")
    if sort_by != "PSI table order":
        show = show.sort_values(sort_by, ascending=not desc, na_position="last", kind="stable")
    for c in num_cols:
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.4g}")
    show = show.astype(object).where(show.notna(), "—")
    event = st.dataframe(show, hide_index=True, height=420, on_select="rerun", selection_mode="single-row",
                         key="rows_table",
                         column_config={"⚠": st.column_config.TextColumn(width=36, help="Corrected or flagged"),
                                        "Investigation": st.column_config.TextColumn(width=170),
                                        "Notes": st.column_config.TextColumn(width=260)})
    st.caption("— = the PSI table has no value there (never 0). O₂ is 'burner' for inverse flames: oxygen flowing "
               "out of the burner, not the atmosphere. Click a row to inspect it. Use 'Sort by' for numeric order "
               "(column headers sort as text).")

    # Download the filtered rows with their original column names, provenance and DOI.
    export = df.drop(columns=["flag_ids"]).copy()
    export.insert(1, "doi", export["investigation_id"].map(lambda x: by_id.get(x, {}).get("doi")))
    ids = sorted(export["investigation_id"].unique(), key=lambda s: int(s.split("-")[1]))
    fname = "psi_tests_" + ("_".join(ids) if 0 < len(ids) <= 5 else f"{len(ids)}_investigations") + ".csv"
    st.download_button(f"⬇️ Download {len(export):,} rows (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv", disabled=export.empty)

    # ---- row inspector
    if not df.empty:
        st.subheader("Inspect a row")
        options = list(df.index)
        sel_rows = (event.selection.rows if event and getattr(event, "selection", None) else []) or []
        clicked = show.index[sel_rows[0]] if sel_rows and sel_rows[0] < len(show) else None
        if clicked is not None and clicked != st.session_state.get("_last_click"):
            st.session_state["_last_click"] = clicked
            st.session_state["inspect"] = clicked
        if st.session_state.get("inspect") not in options:
            flagged = df.index[df["flags"].notna()]
            st.session_state["inspect"] = flagged[0] if len(flagged) else options[0]

        def row_label(i):
            r = tests.loc[i]
            return (f"{'⚠ ' if pd.notna(r['flags']) else ''}{by_id[r['investigation_id']].get('acronym') or r['investigation_id']}"
                    f" · test {r['test_id'] if pd.notna(r['test_id']) else '—'}"
                    f" · {r['date_iso'] if pd.notna(r['date_iso']) else 'no date'}")

        idx = st.selectbox("Row", options, key="inspect", format_func=row_label)
        r = tests.loc[idx]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Corrections and flags** (main's ETL, `etl/fixes.py`)")
            if pd.notna(r["flags"]):
                for f in r["flags"].split(" | "):
                    fid = f.split(":")[0].strip()
                    st.markdown(f"- ⚠ **{fid}** ({FLAG_LABELS.get(fid, 'flag')}): {f.split(':', 1)[-1].strip()}")
            else:
                st.write("None. This row is exactly as parsed from the PSI table.")
            inv = by_id.get(r["investigation_id"], {})
            st.caption(f"Source: {r['source_file']} · DOI "
                       f"[{inv.get('doi')}]({data.doi_url(inv.get('doi'))})")
        with c2:
            st.markdown("**Original PSI row** (verbatim)")
            try:
                raw = json.loads(r["raw_row"]) if isinstance(r["raw_row"], str) else {}
            except json.JSONDecodeError:
                raw = {}
            if raw:
                st.dataframe(pd.DataFrame({"PSI column": list(raw), "value": [str(v) for v in raw.values()]}),
                             hide_index=True, height=min(400, 38 + 35 * len(raw)))
            else:
                st.write("No original row stored.")


def one_investigation(key: str, have: list[str]) -> str | None:
    """Tabs 2 and 3 show one investigation: the picked ones first, else all that have the data."""
    opts = [p for p in (picked or with_rows) if p in have] or have
    return st.selectbox("Investigation", opts, key=key, format_func=lambda x: data.label(by_id[x])) if opts else None


# ---------------------------------------------------------------- original PSI table
with tab_raw:
    st.caption("The experimental table exactly as PSI publishes it (text cells, nothing parsed), to compare with "
               "the cleaned rows.")
    has_table = [i["id"] for i in invs if i.get("experimental_table_files")]
    pid = one_investigation("raw_inv", has_table)
    if pid:
        files = by_id[pid]["experimental_table_files"]
        name = st.radio("Table", files, horizontal=True, key=f"raw_file_{pid}") if len(files) > 1 else files[0]
        raw_df = data.raw_table(name)
        st.dataframe(raw_df, hide_index=True, height=480)
        st.caption(f"{name} · {len(raw_df):,} rows × {raw_df.shape[1]} columns · "
                   f"project-data-1-pc1/data/experimental_tables/")

# ---------------------------------------------------------------- files in PSI
with tab_files:
    st.caption("Every file PSI lists for the investigation. Raw data (videos, images, sensor logs) is far too "
               "large to copy, so it stays on PSI: open the landing page to get it.")
    pid = one_investigation("files_inv", [i["id"] for i in invs])
    if pid:
        idx_df = data.file_index(pid)
        inv = by_id[pid]
        cats = sorted(c for c in idx_df["category"].unique() if c)
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        chosen = c1.multiselect("Category", cats, placeholder="All categories", key=f"cats_{pid}")
        c2.link_button("Open on PSI ↗", inv.get("landing_page") or data.PSI_REPO + pid, width="stretch")
        view = idx_df[idx_df["category"].isin(chosen)] if chosen else idx_df
        summary = idx_df.groupby("category")["bytes"].agg(["size", "sum"]).sort_values("sum", ascending=False)
        st.caption(" · ".join(f"{c or 'other'}: {int(n):,} files, {data.human_size(b)}"
                              for c, (n, b) in summary.iterrows()))
        st.dataframe(view.rename(columns={"category": "Category", "subcategory": "Subcategory",
                                          "file_name": "File", "size": "Size"})[
                         ["Category", "Subcategory", "File", "Size"]],
                     hide_index=True, height=420, column_config={"File": st.column_config.TextColumn(width=520)})
        st.caption(f"{len(view):,} of {len(idx_df):,} files · total {data.human_size(idx_df['bytes'].sum())} · "
                   "links only, nothing is downloaded from PSI by this app.")
