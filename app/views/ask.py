import html
import re

import streamlit as st

from app import data, ui

EXAMPLES = [
    "Does flame spread faster in microgravity than in 1 g?",
    "Which materials were tested in Saffire?",
    "What O2 level stopped droplet burning in FLEX?",
    "How does smoke differ in microgravity (SAME)?",
    "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?",
]

ui.page_header("Ask the data", ("Ask in plain language. Answers come **only from PSI documents and metadata**, and every "
            "statement links to its source (investigation, DOI, file and page)."), "💬")

invs = data.investigations()
with st.expander("Limit the search to specific investigations"):
    limit = st.multiselect("Investigations", [i["id"] for i in invs],
                           format_func=lambda x: data.label(data.investigation(x)))


def citation_url(c: dict) -> str | None:
    return data.doi_url(c.get("doi")) or (data.PSI_REPO + c["psi_id"] if c.get("psi_id") not in (None, "REF") else None)


def render_span(res: dict) -> None:
    """Extractive QA (#22): show the exact answer, highlighted inside the passage it came from.
    The span is located by its text (offsets are only a hint), so a mismatch never mis-highlights."""
    span = res.get("answer_span")
    if not span or not span.get("text"):
        return
    passages = res.get("passages") or []
    i = span.get("passage_index")
    ctx = passages[i] if isinstance(i, int) and 0 <= i < len(passages) else {}
    snippet = ctx.get("snippet") or ctx.get("text") or ""
    with st.container(border=True):
        st.markdown(f"**Answer:** {html.escape(span['text'])}")
        at = snippet.find(span["text"])
        if at >= 0:
            lo, hi = max(0, at - 220), min(len(snippet), at + len(span["text"]) + 220)
            st.markdown(("…" if lo else "") + html.escape(snippet[lo:at]) +
                        f"<mark>{html.escape(span['text'])}</mark>" +
                        html.escape(snippet[at + len(span['text']):hi]) + ("…" if hi < len(snippet) else ""),
                        unsafe_allow_html=True)
        src = []
        if ctx.get("psi_id"):
            inv = data.investigation(ctx["psi_id"])
            src.append(data.label(inv) if inv else ctx["psi_id"])
        if ctx.get("file"):
            src.append(ctx["file"].split("/")[-1].split("#")[0] + (f", p. {ctx['page']}" if ctx.get("page") else ""))
        if span.get("score") is not None:
            src.append(f"QA confidence {span['score']:.2f}")
        if src:
            st.caption(" · ".join(src))


def render_answer(res: dict) -> None:
    cits = res.get("citations") or []
    text = (res.get("answer") or "").replace("$", r"\$")

    def link(m):
        n = int(m.group(1))
        if 1 <= n <= len(cits) and citation_url(cits[n - 1]):
            return f"[[{n}]]({citation_url(cits[n - 1])} \"{cits[n - 1].get('psi_id')}\")"
        return m.group(0)

    # Provenance first: how this answer was produced depends on ask()'s mode (#16).
    text = re.sub(r"^\((?:AI answer unavailable|No AI answer)[^)]*\)\s*", "", text)
    lab = data.answer_label(res)
    if lab:
        ui.provenance_label("ai" if lab[0] == "✨" else "psi", f"{lab[0]} {lab[1]}")
    render_span(res)
    st.markdown(re.sub(r"\[(\d+)\]", link, text))
    if cits:
        st.markdown("**Sources**")
        for n, c in enumerate(cits, 1):
            pid = c.get("psi_id")
            inv = data.investigation(pid) if pid else None
            name = data.label(inv) if inv else ("NASA Researcher's Guide" if pid == "REF" else pid or "—")
            file = (c.get("file") or "").split("/")[-1].split("#")[0]
            page = f", p. {c['page']}" if c.get("page") else ""
            doi = f" · DOI {ui.source_link(c['doi'])}" if c.get("doi") else ""
            st.caption(f"[{n}] **{name}** — {file}{page}{doi}")


if "chat" not in st.session_state:
    st.session_state.chat = []

# /ask?q=... asks a question straight away (shareable demo links, screenshots).
if st.query_params.get("q") and not st.session_state.get("_url_q_done"):
    st.session_state.pending = st.query_params["q"]
    st.session_state["_url_q_done"] = True

if not st.session_state.chat:
    st.markdown("**Try one of these:**")
    cols = st.columns(len(EXAMPLES))
    for col, q in zip(cols, EXAMPLES):
        if col.button(q, key=f"ex_{q}", width="stretch"):
            st.session_state.pending = q

for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant", avatar="🔥"):
        render_answer(turn["res"])

question = st.chat_input("Ask about the PSI combustion experiments…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="🔥"):
        data.wait_for_index()
        with st.spinner("Searching the PSI documents…"):
            try:
                from ai.qa import ask
                res = ask(question, psi_ids=limit or None)
            except Exception as e:  # keep the page alive if the AI layer breaks
                res = {"answer": f"Sorry, the question could not be answered ({type(e).__name__}).",
                       "citations": []}
        render_answer(res)
    st.session_state.chat.append({"q": question, "res": res})
    if len(st.session_state.chat) == 1:
        st.rerun()  # hide the example buttons

if st.session_state.chat and st.button("Clear conversation", type="tertiary"):
    st.session_state.chat = []
    st.rerun()

st.caption("Each answer says how it was produced: passages found by local AI search, a sentence quoted by a "
           "local QA model, or an answer written by a language model from those passages only. Always check "
           "the linked source before relying on an answer.")
