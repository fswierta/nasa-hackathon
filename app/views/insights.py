import streamlit as st

from app import data, ui

PILLARS = [
    ("prevention", "🛡️ Prevention", "Ignition control and material selection from flammability limits."),
    ("detection", "🚨 Detection", "Sensing smoke and fire early, telling smoke from dust."),
    ("suppression", "🧯 Suppression", "Putting fires out: CO₂, N₂, He, water mist."),
]

ui.page_header("Insights", ("Cross-investigation lessons for spacecraft fire safety, grouped by the three NASA "
            "fire-safety pillars. Every insight cites the PSI investigations it rests on."), "💡")

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
    ui.empty_state("Insights are not available yet: they appear here once the AI layer generates "
            "`processed/ai/insights.json`. Meanwhile, the data gaps below come straight from the data.",
            icon="⏳")
else:
    ui.provenance_label("rules" if data.written_by_rules(raw.get("generated") if isinstance(raw, dict) else None)
                    else "ai", data.insights_label(raw))
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
                    st.write(data.partial_g_fix(ins.get("text")))
                    if "partial-gravity" in (ins.get("title") or "").lower():
                        with st.expander("Why 0 g and 1 g don't bound the Moon"):
                            st.markdown(data.PARTIAL_G_NOTE)
                    if ins.get("habitats"):
                        st.caption("Relevant to: " + ", ".join(ins["habitats"]))
                    if ins.get("citations"):
                        st.caption("Sources: " + render_citations(ins["citations"]))
    if len(tabs) > len(PILLARS):
        with tabs[-1]:
            for ins in [i for i in items if i.get("pillar") not in {p for p, *_ in PILLARS}]:
                with st.container(border=True):
                    st.markdown(f"**{ins.get('title', '')}**")
                    st.write(data.partial_g_fix(ins.get("text")))
                    if "partial-gravity" in (ins.get("title") or "").lower():
                        with st.expander("Why 0 g and 1 g don't bound the Moon"):
                            st.markdown(data.PARTIAL_G_NOTE)
                    if ins.get("citations"):
                        st.caption("Sources: " + render_citations(ins["citations"]))

st.subheader("🧠 Myths that microgravity research overturned")
with st.container(border=True):
    st.markdown("NASA's early spacecraft fire safety rested on “several critical misconceptions”:\n"
                "1. “A steady fire cannot exist in a quiescent microgravity environment.”\n"
                "2. “A fire will self-extinguish if ventilation is shut off.”\n"
                "3. “Because the reduced-gravity environment is inherently safer, terrestrial fire standards and "
                "tests represent a very conservative approach to fire safety.”\n\n"
                "Research in drop towers and on orbit “demonstrates the fallacy of these assumptions.” "
                "So switching the fans off is **not** a reliable way to put out a fire.")
    st.caption(f"Quoted from NASA, A Researcher's Guide to ISS Combustion Science (2015), p. 35 — "
               f"[nasa.gov]({data.RESEARCHERS_GUIDE}); also in the PSI-20 documents.")

st.subheader("🌍 Benefits on Earth")
st.markdown("**NASA fire-safety technology named in the challenge**")
SPINOFFS = [
    ("🧯 Fine-water-mist extinguishers",
     "Developed for the ISS, with practical applications on Earth (challenge text). “…non-toxic water-mist "
     "portable fire extinguishers were designed, tested, fabricated and certified for spaceflight. The first two "
     "of the planned nine water-mist portable fire extinguishers were sent to the ISS in December 2015 to replace "
     "CO2 units that are currently onboard.” The research began with the Water Mist Fire Suppression experiment "
     "(Mist) on STS-107 in 2003.",
     f"NASA, A Researcher's Guide to ISS Combustion Science (2015), p. 26 · [nasa.gov]({data.RESEARCHERS_GUIDE})"),
    ("🚨 Smoke detectors",
     "PSI's SAME measured the smoke of spacecraft materials on the ISS (Detection tab): “The smoke detectors "
     "developed from the results of SAME can also be useful in other extreme environments on Earth, such as "
     "submarines or underwater laboratories.” An earlier shuttle smoke detector that tells smoke from dust is a "
     "NASA Spinoff.",
     "PSI-102 SAME research impacts (DOI 10.60555/gd4k-8h62) · NASA Spinoff "
     "[“Fire Detector”](https://spinoff.nasa.gov/node/9407) (outside PSI)"),
    ("🫁 Firefighters' air tanks",
     "NASA technology improved the air tanks firefighters breathe from (NASA Spinoff, 1970s).",
     "NASA Spinoff [“Fireman's Air Tanks”](https://spinoff.nasa.gov/node/9582) (outside PSI; no PSI data)"),
    ("🧥 Firefighters' protective clothing",
     "NASA technology improved firefighters' protective clothing (NASA Spinoff, 1980s).",
     "NASA Spinoff [“New Technology for the Fire Attire”](https://spinoff.nasa.gov/node/9918) "
     "(outside PSI; no PSI data)"),
]
cols = st.columns(2)
for n, (title, text, src) in enumerate(SPINOFFS):
    with cols[n % 2].container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)
        st.caption(src)
st.caption("Spinoff articles are linked from the challenge page; their descriptions here follow the challenge "
           "text, not PSI data.")

if earth:
    st.markdown("**What the PSI investigations say about Earth uses** (quoted from PSI)")
    for e in earth:
        if isinstance(e, str):
            st.markdown(f"- {e}")
        else:
            src = render_citations(e.get("citations"))
            title = f"**{e['title']}** " if e.get("title") else ""
            st.markdown(f"- {title}{e.get('text', '')}" + (f"  \n  <small>{src}</small>" if src else ""),
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
