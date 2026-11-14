import streamlit as st

from app import data

invs = data.investigations()
ids = [i["id"] for i in invs]

# Id from the URL (?id=PSI-26, shareable) or from the overview selection.
pid = st.query_params.get("id") or st.session_state.get("detail_id") or "PSI-26"
if pid not in ids:
    st.error(f"Unknown investigation id {pid!r}.")
    pid = "PSI-26"
pid = st.selectbox("Investigation", ids, index=ids.index(pid),
                   format_func=lambda x: data.label(data.investigation(x)))
st.query_params["id"] = pid
st.session_state["detail_id"] = pid
inv = data.investigation(pid)
prof = data.profile(pid)

# ---------------------------------------------------------------- header
st.title(f"🔬 {inv.get('acronym') or pid}")
st.markdown(f"**{inv.get('title')}**")
tier_txt = {"A": "A — core spacecraft fire safety", "B": "B — related", "C": "C — other combustion"}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Relevance tier", inv.get("relevance_tier") or "—", help=tier_txt.get(inv.get("relevance_tier")))
c2.metric("Type", inv.get("type") or "—")
c3.metric("Platform", (inv.get("platform") or "—").replace("International Space Station", "ISS"))
c4.metric("Test records", inv.get("n_tests") or 0)
links = []
if inv.get("doi"):
    links.append(f"DOI [{inv['doi']}]({data.doi_url(inv['doi'])})")
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
    gen = prof.get("generated") or {}
    st.caption(f"🤖 AI-generated ({gen.get('model', 'Claude')}, {gen.get('date', '')}) from PSI documents — "
               "check the cited sources.")
    if prof.get("key_findings"):
        st.markdown("**Key findings**")
        for n, f in enumerate(prof["key_findings"], 1):
            s = f.get("source") or {}
            where = " · ".join(x for x in (s.get("psi_id"), s.get("file", "").split("/")[-1],
                                           f"p. {s['page']}" if s.get("page") else None) if x)
            st.markdown(f"{n}. {f['text']}  \n<small>Source: {where} · "
                        f"[{s.get('doi')}]({data.doi_url(s.get('doi'))})</small>", unsafe_allow_html=True)
    if prof.get("fire_safety_pillar"):
        st.markdown("**Fire-safety pillar:** " + ", ".join(prof["fire_safety_pillar"]))
else:
    st.write(inv.get("objective") or "No objective on record.")
    st.caption("Objective from PSI metadata. The AI summary appears here once it has been generated.")

with st.expander("Hypothesis, approach and research impacts (PSI metadata)"):
    for key, title in (("objective", "Objective"), ("hypothesis", "Hypothesis"),
                       ("approach", "Approach"), ("research_impacts", "Research impacts / Earth benefits")):
        if inv.get(key):
            st.markdown(f"**{title}**")
            st.write(inv[key])

# ---------------------------------------------------------------- conditions
st.subheader("Test conditions")
c1, c2, c3 = st.columns(3)
c1.metric("O₂ (% by volume)", data.fmt_range(inv.get("o2_range"), scale=100, digits=1))
c2.metric("Pressure (kPa)", data.fmt_range(inv.get("pressure_range_kPa"), digits=1))
c3.metric("Airflow (cm/s)", data.fmt_range(inv.get("flow_range_cm_s"), digits=1))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Varied between tests**")
    st.write(", ".join(inv.get("varied_variables") or []) or "—")
    st.markdown("**Held fixed**")
    st.write(", ".join(inv.get("fixed_variables") or []) or "—")
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
    st.info("No experimental table for this investigation in PSI (e.g. ground/modelling studies).")
else:
    t = t.dropna(axis=1, how="all")
    st.dataframe(t, hide_index=True, height=min(420, 38 + 35 * len(t)))
    st.caption(f"{len(t)} rows from {', '.join(sorted(t['source_file'].dropna().unique()))} "
               "(values normalized: O₂ as mole fraction, pressure in kPa, flow in cm/s).")

# ---------------------------------------------------------------- publications
pubs = inv.get("publications") or []
st.subheader(f"Publications ({len(pubs)})")
if pubs:
    st.markdown("\n".join(
        f"- {p.get('title') or 'Untitled'}" + (f" — [{p['doi']}]({data.doi_url(p['doi'])})" if p.get("doi") else "")
        for p in pubs))
else:
    st.write("None listed in PSI.")
st.caption(f"Cite this dataset: NASA PSI {pid}, DOI {inv.get('doi')}. Licence: {inv.get('license') or 'CC0'}.")
