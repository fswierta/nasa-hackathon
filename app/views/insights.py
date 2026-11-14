import streamlit as st

from app import data

PILLARS = [
    ("prevention", "🛡️ Prevention", "Ignition control and material selection from flammability limits."),
    ("detection", "🚨 Detection", "Sensing smoke and fire early, telling smoke from dust."),
    ("suppression", "🧯 Suppression", "Putting fires out: CO₂, N₂, He, water mist."),
]

st.title("💡 Insights")
st.markdown("Cross-investigation lessons for spacecraft fire safety, grouped by the three NASA "
            "fire-safety pillars. Every insight cites the PSI investigations it rests on.")

raw = data.insights()
if isinstance(raw, list):
    items, earth = raw, []
elif isinstance(raw, dict):
    items = raw.get("insights") or [x for p, *_ in PILLARS for x in (raw.get(p) or [])]
    earth = raw.get("earth_applications") or []
else:
    items, earth = [], []


def _where(c: dict) -> str:
    """'p. 12' for documents, 'PSI metadata' for metadata fields, else the short file name."""
    if c.get("page"):
        return f"p. {c['page']}"
    f = c.get("file") or ""
    if f.startswith("metadata/"):
        return "PSI metadata"
    return f.split("/")[-1].split("#")[0][:40]


def render_citations(cits) -> str:
    out, seen = [], set()
    for c in cits or []:
        if isinstance(c, str):
            c = {"psi_id": c}
        pid = c.get("psi_id")
        inv = data.investigation(pid) if pid else None
        name = data.label(inv) if inv else ("NASA Researcher's Guide" if pid == "REF" else pid or "source")
        where = _where(c)
        text = f"{name}, {where}" if where else name
        if text in seen:
            continue
        seen.add(text)
        doi = c.get("doi") or (inv or {}).get("doi")
        quote = (c.get("quote") or "").replace('"', "'")[:200]
        title = f' "{quote}"' if quote else ""
        out.append(f"[{text}]({data.doi_url(doi)}{title})" if doi else text)
    return " · ".join(out)


if not items:
    st.info("Insights are not available yet: they appear here once the AI layer generates "
            "`processed/ai/insights.json`. Meanwhile, the data gaps below come straight from the data.",
            icon="⏳")
else:
    st.caption(data.insights_label(raw))
    tabs = st.tabs([t for _, t, _ in PILLARS] + (["Other"] if any(
        i.get("pillar") not in {p for p, *_ in PILLARS} for i in items) else []))
    for tab, (key, _, desc) in zip(tabs, PILLARS):
        with tab:
            st.caption(desc)
            group = [i for i in items if i.get("pillar") == key]
            if not group:
                st.write("No insight for this pillar yet.")
            for ins in group:
                with st.container(border=True):
                    st.markdown(f"**{ins.get('title', '')}**")
                    st.write(ins.get("text", ""))
                    if ins.get("habitats"):
                        st.caption("Relevant to: " + ", ".join(ins["habitats"]))
                    if ins.get("citations"):
                        st.caption("Sources: " + render_citations(ins["citations"]))
    if len(tabs) > len(PILLARS):
        with tabs[-1]:
            for ins in [i for i in items if i.get("pillar") not in {p for p, *_ in PILLARS}]:
                with st.container(border=True):
                    st.markdown(f"**{ins.get('title', '')}**")
                    st.write(ins.get("text", ""))
                    if ins.get("citations"):
                        st.caption("Sources: " + render_citations(ins["citations"]))

st.subheader("🌍 Benefits on Earth")
if earth:
    for e in earth:
        if isinstance(e, str):
            st.markdown(f"- {e}")
        else:
            src = render_citations(e.get("citations"))
            st.markdown(f"- **{e.get('title', '')}** {e.get('text', '')}" + (f"  \n  <small>{src}</small>" if src else ""),
                        unsafe_allow_html=True)
else:
    impacts = sorted((i for i in data.investigations() if i.get("research_impacts")),
                     key=lambda i: i.get("relevance_tier") or "Z")
    st.caption("From the 'research impacts' each investigation reports in PSI (not AI-generated):")
    for i in impacts[:8]:
        with st.expander(data.label(i)):
            st.write(i["research_impacts"])

gaps = (data.coverage() or {}).get("gaps") or []
if gaps:
    st.subheader("⚠️ Where the evidence runs out")
    for g in gaps:
        st.markdown(f"- **{g['text']}**")
