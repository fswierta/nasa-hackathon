import pandas as pd
import streamlit as st

from app import data, ui

invs = data.investigations()
ids = [i["id"] for i in invs]

# Id from the URL (?id=PSI-26, shareable) or from the overview selection.
pid = st.query_params.get("id") or st.session_state.get("detail_id") or "PSI-26"
if pid not in ids:
    ui.callout("alert", f"Unknown investigation id {pid!r}.")
    pid = "PSI-26"
pid = st.selectbox("Investigation", ids, index=ids.index(pid),
                   format_func=lambda x: data.label(data.investigation(x)))
st.query_params["id"] = pid
st.session_state["detail_id"] = pid
inv = data.investigation(pid)
prof = data.profile(pid)

# ---------------------------------------------------------------- header
ui.page_header(inv.get("acronym") or pid, f"**{inv.get('title')}**", "🔬")
tier_txt = {"A": "A — core spacecraft fire safety", "B": "B — related", "C": "C — other combustion"}
c1, c2, c3, c4 = st.columns(4)
ui.metric_card(c1, "Relevance tier", inv.get("relevance_tier") or "—",
               help=tier_txt.get(inv.get("relevance_tier")))
ui.metric_card(c2, "Type", inv.get("type") or "—")
ui.metric_card(c3, "Platform", (inv.get("platform") or "—").replace("International Space Station (ISS)", "ISS")
                   .replace(" Spacecraft", ""), help=inv.get("platform"))
n_rows, row_kind = data.table_rows(pid)
ui.metric_card(c4, f"Table rows ({row_kind})" if n_rows else "Table rows", n_rows,
                   help="Rows parsed from the PSI experimental table")
links = []
if inv.get("doi"):
    links.append(f"DOI {ui.source_link(inv['doi'])}")
if inv.get("landing_page"):
    links.append(f"[PSI landing page / raw data]({inv['landing_page']})")
if inv.get("related_ids"):
    links.append("Related: " + ", ".join(data.label(data.investigation(r)) for r in inv["related_ids"]
                                         if data.investigation(r)))
st.markdown(" · ".join(links))

# ---------------------------------------------------------------- summary
st.subheader("Summary")
if prof:
    tab_plain, tab_expert = st.tabs(["Plain language", "Expert"])
    tab_plain.write(prof.get("summary_plain") or "—")
    tab_expert.write(prof.get("summary_expert") or "—")
    ui.provenance_label("rules" if data.written_by_rules(prof.get("generated")) else "ai",
                        data.profile_label(prof))
    val = (prof.get("generated") or {}).get("validation") or {}
    dropped = len(val.get("findings_dropped") or []) + len(val.get("expert_sentences_dropped") or [])
    if dropped or val.get("findings_added_from_rules") or val.get("plain_summary_replaced_by_rules"):
        st.caption("Number check: " + "; ".join(x for x in (
            f"{dropped} AI sentence(s) removed because their numbers are not in the cited sources" if dropped else "",
            f"{val['findings_added_from_rules']} finding(s) added as quotes picked by rules"
            if val.get("findings_added_from_rules") else "",
            "plain summary replaced by the PSI metadata text" if val.get("plain_summary_replaced_by_rules") else "",
        ) if x) + ".")
    if prof.get("key_findings"):
        st.markdown("**Key findings**")
        for n, f in enumerate(prof["key_findings"], 1):
            s = f.get("source") or {}
            where = " · ".join(x for x in (s.get("psi_id"), s.get("file", "").split("/")[-1],
                                           f"p. {s['page']}" if s.get("page") else None) if x)
            st.markdown(f"{n}. {f['text']}  \n<small>Source: {where} · "
                        f"{ui.source_link(s.get('doi'))}</small>", unsafe_allow_html=True)
    if prof.get("fire_safety_pillar"):
        st.markdown("**Fire-safety pillar:** " + ", ".join(prof["fire_safety_pillar"]))
else:
    ui.empty_state("No summary generated for this investigation yet. The investigators' own description is below.")

# ---------------------------------------------------------------- investigators' own text (#34)
# Verbatim PSI metadata written by the investigation team: shown as theirs, with none of our labels.
st.subheader("In the investigators' words")
team = "; ".join(f"{c['name']} ({c['affiliation']})" if c.get("affiliation") else c["name"]
                 for c in inv.get("contacts") or [] if c.get("name"))
doi_md = f"DOI {ui.source_link(inv['doi'])}" if inv.get("doi") else "no DOI on record"
with st.container(border=True):
    ui.provenance_label("psi", f"Written by the investigation team: {team or 'team not listed in PSI'} · "
               f"Source: PSI metadata, {doi_md} · unedited")
    words = [(t, inv.get(k)) for k, t in (("objective", "Objective"), ("hypothesis", "Hypothesis"),
                                          ("approach", "Approach"),
                                          ("research_impacts", "Research impacts / Earth benefits"))]
    words = [(t, v) for t, v in words if v]
    if words:
        for tab, (_, text) in zip(st.tabs([t for t, _ in words]), words):
            tab.markdown(text.replace("$", r"\$"))   # escape $ so it is not read as LaTeX
    else:
        st.write("PSI has no objective, hypothesis, approach or impacts text for this investigation.")

# ---------------------------------------------------------------- conditions
st.subheader("Test conditions")
c1, c2, c3 = st.columns(3)
ui.metric_card(c1, "O₂ (% by volume)", data.fmt_range(inv.get("o2_range"), scale=100, digits=1))
ui.metric_card(c2, "Pressure (kPa)", data.fmt_range(inv.get("pressure_range_kPa"), digits=1))
ui.metric_card(c3, "Airflow (cm/s)", data.fmt_range(inv.get("flow_range_cm_s"), digits=1))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Varied between tests**")
    st.write(data.pretty_vars(inv.get("varied_variables")))
    st.markdown("**Held fixed**")
    st.write(data.pretty_vars(inv.get("fixed_variables")))
with c2:
    st.markdown("**Fuels / materials**")
    st.write(", ".join(inv.get("fuels") or []) or "—")
    st.markdown("**Fuel phase · hardware**")
    st.write(f"{inv.get('fuel_phase') or '—'} · {inv.get('hardware_family') or inv.get('hardware') or '—'}")
    sensors = (prof or {}).get("measurements") or inv.get("measurements")
    if sensors:
        st.markdown("**Measurements / sensors**")
        st.write(", ".join(s if isinstance(s, str) else s.get("name", str(s)) for s in sensors))
st.caption("'no data' = the PSI experimental table has no such column, not a value of zero.")

# ---------------------------------------------------------------- tests
st.subheader("Test records")
t = data.tests()
t = t[t["investigation_id"] == pid].drop(columns=["investigation_id", "raw_row"], errors="ignore")
if t.empty:
    ui.empty_state("No experimental table for this investigation in PSI (e.g. ground/modelling studies).")
else:
    t = t.dropna(axis=1, how="all")
    st.dataframe(t, hide_index=True, height=min(420, 38 + 35 * len(t)))
    st.caption(f"{len(t)} rows from {', '.join(sorted(t['source_file'].dropna().unique()))} "
               "(values normalized: O₂ as mole fraction, pressure in kPa, flow in cm/s).")

# ---------------------------------------------------------------- neighbours (#24)
sim = data.similarity()
neigh = ((sim or {}).get("neighbors") or {}).get(pid) or []
if neigh:
    st.subheader("Most similar investigations")
    cols = st.columns(min(3, len(neigh)))
    for col, nb in zip(cols, neigh[:3]):
        other = data.investigation(nb["id"])
        if not other:
            continue
        with col.container(border=True):
            st.markdown(f"**{data.label(other)}**")
            st.caption(f"Similarity {nb.get('score', 0):.2f}" +
                       (" · shared: " + ", ".join(map(str, nb["shared"])) if nb.get("shared") else ""))
            if st.button("Open", key=f"nb_{nb['id']}", type="tertiary"):
                st.session_state["detail_id"] = nb["id"]
                st.query_params["id"] = nb["id"]
                st.rerun()
    st.caption(f"Computed from the PSI metadata text and test conditions ({data.similarity_method(sim)}); "
               "see the similarity map on the Compare page.")

# ---------------------------------------------------------------- publications
# ---------------------------------------------------------------- science documents (#34)
docs = data.documents(pid)
st.subheader(f"Science documents ({len(docs)})")
if docs:
    st.dataframe(pd.DataFrame([{
        "Title": d["title"], "Type": d["type"], "Format": d["format"], "Size": data.human_size(d["bytes"]),
        "Pages searchable in Ask": str(d["pages_indexed"] or "—"),   # one dtype for Arrow
    } for d in docs]), hide_index=True, column_config={"Title": st.column_config.TextColumn(width=520)})
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Download a document", range(len(docs)),
                        format_func=lambda n: f"{docs[n]['title']} ({docs[n]['format']}, "
                                              f"{data.human_size(docs[n]['bytes'])})")
    c2.download_button("⬇️ Download", data=data.file_bytes(docs[pick]["path"]), file_name=docs[pick]["file"],
                       key=f"doc_{pid}", width="stretch")
    st.caption("These are the PSI science documents the AI search cites (file + page). The full archive, "
               "including raw data, is on the [PSI landing page]"
               f"({inv.get('landing_page') or data.PSI_REPO + pid}).")
else:
    st.write("No science documents for this investigation in our local copy. "
             f"See the [PSI landing page]({inv.get('landing_page') or data.PSI_REPO + pid}).")

pubs = inv.get("publications") or []
st.subheader(f"Publications ({len(pubs)})")
if pubs:
    st.markdown("\n".join(
        f"- **{p.get('title') or 'Untitled'}**" + (f"  \n  {p['authors']}" if p.get("authors") else "") +
        (f" · {ui.source_link(p['doi'], 'doi.org/' + p['doi'])}" if p.get("doi") else "")
        for p in pubs))
else:
    st.write("None listed in PSI.")
st.caption(f"Cite this dataset: NASA PSI {pid}, DOI {inv.get('doi')}. Licence: {inv.get('license') or 'CC0'}.")
