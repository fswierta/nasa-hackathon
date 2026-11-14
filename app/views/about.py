import re

import pandas as pd
import streamlit as st

from app import data

st.title("ℹ️ About the data")

invs = data.investigations()
tests = data.tests()
cov = data.coverage() or {}
docs = data.doc_stats()
_raw_tb = sum(i.get("total_size_gb") or 0 for i in invs) / 1000   # PSI file sizes, all 24 investigations

st.markdown(
    "All data comes from the **NASA Physical Sciences Informatics (PSI)** repository "
    "([psi.nasa.gov](https://psi.nasa.gov)), *Combustion Science* research area: "
    f"{len(invs)} investigations, their metadata, experimental tables and science documents. "
    "PSI datasets are released under **Creative Commons Zero v1.0 (CC0)**; PSI asks that each dataset be "
    f"cited by its DOI, listed below. Raw data (videos, images, sensor files, ~{_raw_tb:.1f} TB) stays on PSI; "
    "every investigation links to it."
)
st.warning("This is an independent hackathon project for the NASA Space Apps Challenge 2026. "
           "It is **not endorsed by NASA**, and NASA's name is used only to identify the data source.",
           icon="⚠️")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Investigations", len(invs))
c2.metric("Test records", f"{len(tests):,}")
tot = cov.get("totals") or {}
c3.metric("…with O₂ and pressure", f"{tot.get('rows_with_o2_and_pressure', 0):,}")
c4.metric("Documents indexed", docs["n_documents"])

# ---------------------------------------------------------------- citations
st.subheader("Datasets and DOIs")
st.dataframe(pd.DataFrame([{
    "PSI ID": i["id"], "Investigation": i.get("acronym") or "(ground study)", "Title": i.get("title"),
    "Type": i.get("type"), "Table rows": data.table_rows(i["id"])[0],
    "Rows are": data.table_rows(i["id"])[1] if data.table_rows(i["id"])[0] else "—",
    "Documents": docs["per_investigation"].get(i["id"], 0),
    "DOI": data.doi_url(i.get("doi")), "PSI page": i.get("landing_page"),
} for i in invs]), hide_index=True, column_config={
    "DOI": st.column_config.LinkColumn(display_text=r"https://doi\.org/(.*)"),
    "PSI page": st.column_config.LinkColumn(display_text="open"),
    "Investigation": st.column_config.TextColumn(width=230),
    "Title": st.column_config.TextColumn(width=520),
})

# ---------------------------------------------------------------- data quality
st.subheader("Data issues we found in the PSI tables")
st.markdown("The experimental tables are hand-made spreadsheets. Our ETL normalizes units "
            "(O₂ → mole fraction, pressure → kPa, airflow → cm/s) and **never guesses**: a value it cannot "
            "read is left empty and logged. Grouped log:")
log = data.etl_log()
if log.empty:
    st.info("No ETL log available.")
else:
    log = log.copy()
    log["kind"] = log["issue"].map(lambda s: re.sub(r"\[[^\]]*\]", "[…]", s or "").strip())
    val_col = "value" if "value" in log.columns else "raw_value"
    groups = (log.groupby("kind")
              .agg(count=("kind", "size"),
                   files=("source_file", lambda s: ", ".join(sorted({f.split("_")[0] for f in s if f}))),
                   examples=(val_col, lambda s: " · ".join(list(dict.fromkeys(v for v in s if v))[:4])))
              .sort_values("count", ascending=False).reset_index())
    st.dataframe(groups.rename(columns={"kind": "Issue", "count": "Cells", "files": "Investigations",
                                        "examples": "Example values"}), hide_index=True)
    with st.expander(f"Full log ({len(log)} entries)"):
        st.dataframe(log.drop(columns=["kind"]), hide_index=True)

gaps = cov.get("gaps") or []
if gaps:
    st.subheader("Coverage gaps")
    for g in gaps:
        st.markdown(f"- **{g['text']}** {g.get('evidence', '')}")
    if not any("partial" in g["text"].lower() for g in gaps):
        st.markdown("- **No partial-gravity data.** PSI has no combustion tests at lunar (1/6 g) or Martian "
                    "(3/8 g) gravity; microgravity and 1 g results only bound those cases.")

# ---------------------------------------------------------------- AI use
st.subheader("How AI is used (disclosure)")
_prof = data.profile("PSI-26")
_summ = ("built by rules: the summary quotes the PSI objective/approach and the findings are sentences quoted "
         "from PSI documents (no AI-written text)" if _prof and data.written_by_rules(_prof.get("generated"))
         else "written by a language model from PSI documents and labelled as AI-written, with sources")
_ins = data.insights()
_ins_txt = ("computed from the PSI tables and backed by quoted sentences (no AI-written text)"
            if isinstance(_ins, dict) and data.written_by_rules(_ins.get("generated"))
            else "labelled with how they were produced")
st.markdown(
    "- **Scenario ranking** and the **condition map** are deterministic calculations on the PSI tables — "
    "no AI, fully reproducible.\n"
    "- **Ask the data** finds passages with local AI search (a sentence-embedding model + keyword search). "
    "Depending on configuration, the answer is (a) the passages themselves, (b) a sentence quoted by a local "
    "question-answering model, or (c) an answer written by a language model (Google Gemini free tier or "
    "Anthropic Claude) only from those passages, with uncited claims removed. Each answer is labelled with "
    "the mode that produced it; in mode (c) the question is sent to that provider.\n"
    f"- **Investigation summaries and findings** are {_summ}.\n"
    f"- **Insights** are {_ins_txt}.\n"
    "- The project code was written with help from AI coding assistants (Claude Code).\n\n"
    "Every statement links to its PSI source (investigation, DOI, file and page) so it can be checked."
)
st.caption("Built for NASA Space Apps Challenge 2026 — 'Flame in Freefall'.")
