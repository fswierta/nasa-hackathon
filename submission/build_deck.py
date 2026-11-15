"""Build submission/deck.pptx (8 slides, 16:9) for IGNIS — issue #39.

Numbers are read from processed/key_numbers.json at build time, screenshots from app/screenshots/
(cropped into submission/assets/). Speaker notes on every slide. No NASA logo.

    py submission/build_deck.py            -> submission/deck.pptx
    (PDF: submission/export_deck.ps1 uses PowerPoint to write deck.pdf + slide PNGs for review)
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "deck.pptx"
ASSETS = ROOT / "submission" / "assets"
SHOTS = ROOT / "app" / "screenshots"
K = {k: v["value"] for k, v in json.loads((ROOT / "processed" / "key_numbers.json").read_text(encoding="utf-8")).items()}
FLEX_CARD = json.loads((ROOT / "processed" / "analysis" / "flex_boundary.json").read_text(encoding="utf-8"))["model_card"]
BENCH = json.loads((ROOT / "processed" / "ai" / "benchmark.json").read_text(encoding="utf-8"))
N_SLIDES = 8

# palette: deep-space navy dominates, flame orange is the single accent
NAVY, NAVY2 = RGBColor(0x0B, 0x14, 0x26), RGBColor(0x17, 0x25, 0x40)
FLAME, EMBER = RGBColor(0xF2, 0x64, 0x19), RGBColor(0xFF, 0xB0, 0x00)
WHITE, INK, MUTED = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x14, 0x1C, 0x2B), RGBColor(0x5B, 0x6B, 0x82)
TINT, ICE = RGBColor(0xF1, 0xF4, 0xF9), RGBColor(0xC9, 0xD6, 0xEA)
HEAD, BODY = "Arial", "Calibri"
W, H = 13.333, 7.5
FOOTER = "IGNIS · NASA Space Apps 2026 · Flame in Freefall"


def fmt(n):
    return f"{n:,}" if isinstance(n, int) else f"{n:g}"


# ------------------------------------------------------------------ helpers

def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing_after=0, name=None):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing_after)
        for chunk in (para if isinstance(para, list) else [(para, {})]):
            t, o = chunk if isinstance(chunk, tuple) else (chunk, {})
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, name=None, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


def badge(slide, x, y, d, label, fill=FLAME, color=WHITE, size=16, name=None):
    """Motif: number/year inside a flame-coloured circle."""
    c = box(slide, x, y, d, d, fill, shape=MSO_SHAPE.OVAL, name=name)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, color
    return c


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def title(slide, t, dark=False, sub=None):
    ph = slide.shapes.title
    ph.left, ph.top, ph.width, ph.height = Inches(0.6), Inches(0.45), Inches(W - 1.2), Inches(0.9)
    tf = ph.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = t
    r.font.name, r.font.size, r.font.bold = HEAD, Pt(34), True
    r.font.color.rgb = WHITE if dark else INK
    if sub:
        text(slide, 0.6, 1.3, W - 1.2, 0.5, sub, size=16, color=ICE if dark else MUTED)


def footer(slide, n, dark=False):
    text(slide, 0.6, H - 0.45, 8, 0.3, FOOTER, size=10, color=ICE if dark else MUTED, name="Footer")
    text(slide, W - 1.6, H - 0.45, 1.0, 0.3, f"{n} / {N_SLIDES}", size=10, color=ICE if dark else MUTED,
         align=PP_ALIGN.RIGHT, name="Slide number")


def crop(src, name, top=0, height=None, left=0, width=None):
    """Crop a screenshot (pixel box) into submission/assets and return (path, aspect w/h)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    im = Image.open(SHOTS / src)
    width = width or im.width - left
    height = height or im.height - top
    out = ASSETS / name
    im.crop((left, top, left + width, top + height)).save(out)
    return out, width / height


def picture(slide, path, x, y, w=None, h=None, aspect=None, name=None):
    if w and not h:
        h = w / aspect
    if h and not w:
        w = h * aspect
    frame = box(slide, x - 0.06, y - 0.06, w + 0.12, h + 0.12, ICE, radius=0.03, name=(name or "") + " frame")
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        pic.name = name
    return pic, frame


def bar_chart(slide, x, y, w, h, categories, series, colors, name):
    """Native PowerPoint bar chart (editable in PowerPoint); values 0-1 shown as percentages."""
    data = CategoryChartData()
    data.categories = categories
    for label, values in series:
        data.add_series(label, values)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gf.name = name
    ch = gf.chart
    ch.font.size, ch.font.name = Pt(11), BODY
    ch.font.color.rgb = INK
    ch.has_legend = True
    ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
    ch.legend.font.size = Pt(11)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0, 1
    va.has_major_gridlines = False
    va.visible = False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11)
    ca.format.line.color.rgb = ICE
    ca.reverse_order = True
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 60, -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = "0%", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(10)
    for ser, col in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    return gf


# ------------------------------------------------------------------ slides

def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    layout = prs.slide_layouts[5]  # "Title Only": real title placeholder, everything else composed
    prs.core_properties.title = "IGNIS — Intelligent Guidance for Navigating Ignition in Space"
    prs.core_properties.subject = "NASA Space Apps Challenge 2026 — Flame in Freefall"

    shot_rank, a_rank = crop("ranker.png", "ranker.png", left=380, width=1040, top=100, height=865)
    shot_ask, a_ask = crop("ask.png", "ask.png", left=380, width=1040, top=95, height=520)
    shot_map, a_map = crop("condition_map.png", "condition_map.png", left=380, width=700, top=495, height=620)
    shot_resp, a_resp = crop("response.png", "response.png", left=330, width=1100, top=1345, height=500)
    shot_ml, a_ml = crop("ml_lab_train.png", "ml_lab_train.png", left=60, width=1310, top=860, height=590)

    # 1 — hook + name (dark)
    s = prs.slides.add_slide(layout)
    background(s, NAVY)
    s.shapes.title.text_frame.text = ""
    s.shapes.title.left, s.shapes.title.top, s.shapes.title.width, s.shapes.title.height = Inches(0.6), Inches(1.2), Inches(6.6), Inches(1.4)
    tf = s.shapes.title.text_frame
    tf.margin_left = 0
    r = tf.paragraphs[0].add_run()
    r.text = "IGNIS"
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(88), True, WHITE
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    text(s, 0.6, 2.75, 6.4, 1.0, "Intelligent Guidance for Navigating Ignition in Space", size=24, color=EMBER, font=HEAD, bold=True)
    text(s, 0.6, 3.85, 6.2, 1.4,
         [f"Fire-safety insights from {K['investigations']} NASA microgravity combustion investigations — "
          "for the people designing spacecraft, Moon and Mars bases and rovers."], size=17, color=ICE)
    text(s, 0.6, 5.55, 6.2, 0.9, [[("Next stop: a Moon base at ", {}), ("≥30 % O₂, ≤10.2 psi", {"bold": True, "color": EMBER}),
                                   (" — far richer in oxygen than Earth air, so materials burn more easily.", {})]],
         size=15, color=WHITE)
    events = [("1967", "Apollo 1", "Cabin fire in pure oxygen during a ground test; three crew lost"),
              ("1970", "Apollo 13", "Oxygen tank explosion on the way to the Moon"),
              ("1997", "Mir", "Fire from an oxygen generator on the space station"),
              ("now", "Artemis II", "Smoke alarm in flight, \"80,000 miles from home\"")]
    for i, (yr, who, what) in enumerate(events):
        y = 1.25 + i * 1.3
        box(s, 7.55, y, 5.2, 1.08, NAVY2, name=f"Incident {who}")
        badge(s, 7.75, y + 0.16, 0.76, yr, size=15 if yr != "now" else 14, name=f"Year {who}")
        text(s, 8.75, y + 0.14, 3.85, 0.4, who, size=17, bold=True, color=WHITE, font=HEAD)
        text(s, 8.75, y + 0.5, 3.85, 0.55, what, size=13, color=ICE)
    footer(s, 1, dark=True)
    s.notes_slide.notes_text_frame.text = (
        "Open with the challenge's Artemis II quote: 'If you want to get somebody's attention really quick, make the fire "
        "alarm go off in your spacecraft when you're still about 80,000 miles from home.' Fire has cost crews before. "
        "In microgravity there is no buoyancy, so flames become blue spheres fed by diffusion and fan airflow. "
        "The next habitats - Moon base and pressurized rovers - will run at 30 % oxygen or more at 10.2 psi or less. "
        "IGNIS answers: which NASA experiments tell us how materials burn there, and where is there no evidence?")

    # 2 — the data problem
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "NASA's evidence is scattered across 24 archives",
          sub="Physical Sciences Informatics (PSI): every combustion investigation flown on the ISS and Cygnus, and the ground studies built on them")
    stats = [(fmt(K["investigations"]), f"investigations\n({K['flight_investigations']} flown in space)"),
             (fmt(K["psi_files"]), "files in PSI"),
             (f"{K['psi_raw_tb']:g} TB", "raw images, video, sensor data"),
             (fmt(K["publications"]), "publications")]
    for i, (big, small) in enumerate(stats):
        x = 0.6 + i * 3.08
        box(s, x, 2.1, 2.85, 1.95, TINT, name=f"Stat {small.split()[0]}")
        text(s, x + 0.25, 2.25, 2.4, 0.95, big, size=44, bold=True, color=FLAME, font=HEAD)
        text(s, x + 0.25, 3.2, 2.45, 0.8, small, size=14, color=INK)
    text(s, 0.6, 4.4, 12, 0.5, [[("22 test tables — each with its own columns and units. ", {"bold": True}),
                                 ("Real errors we found, fixed and flagged with the PSI document that proves it:", {})]],
         size=16, color=INK)
    errors = [("CH₄ → CH₅ … CH₁₄", "Spreadsheet auto-fill turned one fuel into eleven fake ones (ACME CLD)"),
              ("3.7 → 0.37 mm", "Fabric thickness under a wrong 'cm' header (Saffire; SAFFIRE document p. 3)"),
              ("2103 → 2013", "A test dated in the 22nd century; the GMT column gives the real day (BASS)")]
    for i, (big, small) in enumerate(errors):
        x = 0.6 + i * 4.1
        box(s, x, 5.05, 3.85, 1.55, WHITE, line=ICE, name=f"Error {i + 1}")
        text(s, x + 0.25, 5.2, 3.4, 0.5, big, size=20, bold=True, color=NAVY, font=HEAD)
        text(s, x + 0.25, 5.72, 3.4, 0.8, small, size=13, color=MUTED)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "The data is public but hard to use: mmHg vs atm vs bar, oxygen as percent or mole fraction, ranges inside cells. "
        f"We cleaned all {K['experiment_tables_parsed']} parseable tables into {fmt(K['test_records'])} test records. "
        f"{K['rows_with_corrections']} rows carry a correction flag naming the PSI document behind it - "
        "for example the Excel auto-fill CH4 to CH14, a fabric 'thickness 3.7 mm' that is really 0.37 mm, "
        "and a test dated 2103. Nothing is guessed: missing values stay empty.")

    # 3 — what we built (process flow)
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "What we built: from PSI archives to one decision tool",
          sub="Every number on screen traces back to a PSI table or document page")
    steps = [("1", "Clean", [f"{fmt(K['test_records'])} test records", "common units: O₂ fraction, kPa, cm/s, mm",
                             f"{K['rows_with_corrections']} rows with documented fixes"]),
             ("2", "Index", [f"{K['documents_indexed']} NASA documents", f"{fmt(K['passages'])} passages with file + page",
                             f"AI search: right source first in {K['retrieval_source_hit1_pct']} % of {K['retrieval_questions']} test questions"]),
             ("3", "Model", [f"Droplets: goes out or burns completely? {K['goes_out_accuracy_pct']:g} % vs {K['goes_out_baseline_pct']:g} % baseline",
                             f"Gas flames: self-extinguish {K['gas_self_extinguish_accuracy_pct']:g} % vs {K['gas_self_extinguish_baseline_pct']:g} %",
                             f"Similarity map: {K['similarity_clusters']} clusters, {K['similarity_related_found']} of {K['similarity_related_total']} PSI links found"]),
             ("4", "Dashboard", ["14 pages: rank, map, ask, fire scenario, response advisor, ML Lab",
                                 "runs with no API key", "optional Gemini answers, every claim cited"])]
    for i, (num, head, lines) in enumerate(steps):
        x = 0.6 + i * 3.12
        box(s, x, 2.2, 2.82, 3.7, TINT, name=f"Step {head}")
        badge(s, x + 0.25, 2.42, 0.62, num, size=18, name=f"Step number {num}")
        text(s, x + 1.02, 2.5, 1.7, 0.5, head, size=22, bold=True, color=NAVY, font=HEAD)
        text(s, x + 0.25, 3.35, 2.4, 2.45, [[("▪ ", {"color": FLAME}), (l, {})] for l in lines],
             size=14, color=INK, spacing_after=8)
        if i < 3:
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.86), Inches(3.85), Inches(0.22), Inches(0.36))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = FLAME; arrow.line.fill.background()
            arrow.name = f"Arrow {i + 1}"
    text(s, 0.6, 6.2, 12.1, 0.6, [[("Our definition of “SI-powered”: ", {"bold": True, "color": FLAME}),
                                   ("AI that shows its sources — models refuse to extrapolate beyond tested ranges, "
                                    "and generated text keeps only claims backed by a retrieved PSI passage.", {})]],
         size=15, color=INK)
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "Four layers. Clean: per-table parsers plus documented fixes. Index: NASA's PDFs and Word files split into passages, "
        "searched by a small local embedding model plus keywords - the vectors ship in the repository, so a fresh clone works. "
        f"Model: a random forest on {K['goes_out_n_train']} FLEX droplet tests predicts whether a flame goes out or the droplet "
        f"burns completely, {K['goes_out_accuracy_pct']:g} percent cross-validated versus {K['goes_out_baseline_pct']:g} for always "
        f"guessing 'goes out' - the next slide shows how we got there; a burning-rate regression; and a similarity map that rediscovers {K['similarity_related_found']} of "
        f"{K['similarity_related_total']} 'related investigation' links PSI curators wrote by hand. Dashboard: works offline; "
        "with a free Gemini key the Ask page writes answers from the retrieved passages only.")

    # 4 — how we know the models work (ML validation)
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "How we know the models work",
          sub="Cross-validated against a baseline - every number below is recomputed on each build")
    study = FLEX_CARD["imbalance_study"]
    short = {"3 classes, no weighting": "3 classes, plain",
             "3 classes, class_weight=balanced (previous model)": "3 classes, class-balanced",
             "3 classes, balanced + bigger leaves (8)": "3 classes, bigger leaves"}
    cats = [short.get(r["strategy"], "2 classes, conservative" if "tuned" in r["strategy"] else "2 classes (ours)")
            for r in study]
    n_comp = FLEX_CARD["class_counts"]["completion"]
    text(s, 0.6, 1.95, 6.2, 0.4, f"Class imbalance: {n_comp} of {FLEX_CARD['n_train']} droplets burned out",
         size=17, bold=True, color=NAVY, font=HEAD)
    bar_chart(s, 0.45, 2.35, 6.4, 3.45, cats,
              [("Complete burns found (recall)", [r["recall_completion"] for r in study]),
               ("Right when it says so (precision)", [r["precision_completion"] for r in study])],
              [FLAME, NAVY], "Chart imbalance")
    text(s, 0.6, 5.85, 6.2, 0.75,
         [[("Fix: ", {"bold": True, "color": FLAME}),
           (f"'disruption' (droplet broke up) is a test ending, not a fire outcome. Leaving it out lifts recall "
            f"{K['completion_recall_3class_pct']} % → {K['completion_recall_pct']} % at {K['completion_precision_pct']} % precision, "
            f"ROC-AUC {K['goes_out_roc_auc']:.2f}. Threshold tuned only inside training folds (nested CV).", {})]],
         size=12, color=INK)
    names = {"keyword (BM25)": "Keyword (BM25)", "vector (MiniLM embeddings)": "Embeddings (MiniLM)",
             "hybrid (as used by Ask)": "Hybrid (used by Ask)"}
    summ = BENCH["summary"]
    text(s, 7.15, 1.95, 5.6, 0.4, f"AI search: {BENCH['n_questions']} questions with a known source", size=17, bold=True,
         color=NAVY, font=HEAD)
    bar_chart(s, 7.0, 2.35, 5.85, 3.45, [names.get(m, m) for m in summ],
              [("Right source first", [v["hit@1"] for v in summ.values()]),
               ("Answer passage in top 3", [v["answer@3"] for v in summ.values()])],
              [FLAME, NAVY], "Chart retrieval")
    hyb = next(v for k, v in summ.items() if k.startswith("hybrid"))
    text(s, 7.15, 5.85, 5.6, 0.75,
         [[("Half the questions never name the experiment; ", {"bold": True, "color": FLAME}),
           (f"expected sources were fixed before the run. Hybrid search has the right source in its top 3 for "
            f"{hyb['hit@3'] * 100:.0f} % - Gemini may only write from those passages.", {})]],
         size=12, color=INK)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        f"Judges should not have to trust us, so the models are measured. Left: the droplet model. Only {n_comp} of "
        f"{FLEX_CARD['n_train']} FLEX droplets burned completely, so a classifier learns to say 'the flame goes out' and misses "
        f"most complete burns - class weights alone found only {K['completion_recall_3class_pct']} percent of them. The fix was "
        "the framing: 'disruption' means the droplet broke up or moved, a test ending rather than a fire outcome. Leaving it "
        f"out, the model finds {K['completion_recall_pct']} percent of complete burns and is right {K['completion_precision_pct']} "
        f"percent of the time it says so, ROC-AUC {K['goes_out_roc_auc']:.2f}, and it beats the always-goes-out baseline, "
        f"{K['goes_out_accuracy_pct']:g} versus {K['goes_out_baseline_pct']:g} percent. A conservative threshold finds "
        f"{K['completion_recall_conservative_pct']} percent, at lower precision - used as a 'may burn completely' warning, "
        "because saying 'it goes out' when it does not is the dangerous error. Right: the search behind Ask, tested on "
        f"{BENCH['n_questions']} questions whose source we fixed before running; the hybrid search puts the right source first "
        f"in {hyb['hit@1'] * 100:.0f} percent.")

    # 5 — demo: rank for your habitat
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "Rank the experiments that matter for your habitat")
    ph4 = 5.15
    picture(s, shot_rank, 0.6, 1.5, h=ph4, aspect=a_rank, name="Screenshot ranker")
    x4 = 0.6 + ph4 * a_rank + 0.55
    points = [("1", "Pick a habitat", "ISS, Lunar base, Mars base, pressurized rover, spacesuit — or set your own O₂, pressure, airflow, gravity and materials"),
              ("2", "See why", "Each bar is split into its score components — O₂, pressure, airflow, material, pillar, relevance, data richness — with adjustable weights"),
              ("3", "See what's missing", "Every result lists its data gaps and DOI; Moon and Mars gravity always show “no PSI test at this gravity”")]
    for i, (n, head, body) in enumerate(points):
        y = 1.55 + i * 1.7
        badge(s, x4, y, 0.55, n, size=16, name=f"Point {n}")
        text(s, x4 + 0.75, y - 0.02, W - x4 - 1.35, 0.45, head, size=18, bold=True, color=NAVY, font=HEAD)
        text(s, x4 + 0.75, y + 0.42, W - x4 - 1.35, 1.2, body, size=13, color=MUTED)
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Live demo: choose the Lunar base preset. Droplet and gas-flame studies (FLEX, CFI-G) rank high because their "
        "tested oxygen and pressure overlap the Moon-base atmosphere; the solid-material studies (BASS, Saffire) rank on "
        "materials - and every card says which part of the evidence is missing. Move a weight slider to show the ranking is "
        "transparent, not a black box.")

    # 6 — demo: ask + map
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "Ask, map, respond — and open the models")
    cw = 4.25
    cells = [(shot_ask, a_ask, "1", 0.75, 1.6), (shot_resp, a_resp, "3", 0.75, 4.35),
             (shot_ml, a_ml, "4", 0.75 + cw + 0.45, 4.35)]
    for path, asp, n, x, y in cells:
        picture(s, path, x, y, w=cw, aspect=asp, name=f"Screenshot {n}")
        badge(s, x - 0.3, y - 0.3, 0.42, n, size=13, name=f"Badge {n}")
    mh = 2.45
    picture(s, shot_map, 0.75 + cw + 0.45, 1.6, h=mh, aspect=a_map, name="Screenshot 2")
    badge(s, 0.75 + cw + 0.45 - 0.3, 1.6 - 0.3, 0.42, "2", size=13, name="Badge 2")
    caps = [("1", "Ask the data", "Gemini writes the answer only from NASA passages; every claim cites its page"),
            ("2", "Condition map", "Every test by O₂ and pressure against planned habitat atmospheres"),
            ("3", "Response advisor", f"With CO₂ in O₂-rich air, {K['o2_rich_went_out_co2_pct']} % of droplet flames went out vs {K['o2_rich_went_out_n2_only_pct']} % without"),
            ("4", "ML Lab", "Explore the data, choose features, train with cross-validation, compare with the baseline")]
    cx = 0.75 + 2 * cw + 0.85
    for i, (n, head, body) in enumerate(caps):
        y = 1.55 + i * 1.33
        badge(s, cx, y, 0.42, n, size=13, name=f"Caption badge {n}")
        text(s, cx + 0.55, y - 0.03, W - cx - 1.15, 0.35, head, size=14, bold=True, color=NAVY, font=HEAD)
        text(s, cx + 0.55, y + 0.33, W - cx - 1.15, 0.95, body, size=11, color=MUTED)
    footer(s, 6)
    s.notes_slide.notes_text_frame.text = (
        "Four pages in one minute. Ask: a question about Moon-base atmospheres; Gemini answers only from retrieved NASA passages "
        "and each claim links to its page. Condition map: every test by oxygen and pressure against the planned habitats - "
        "burner oxygen of inverse flames is kept off, it is not cabin air. Response advisor: what the research says about "
        "fighting a fire in that atmosphere - suppressant curves from the FLEX model, the measured table, and NASA's own list "
        "of misconceptions, always with the note that this is not a crew procedure. ML Lab: the models are open - data, "
        "features, cross-validation and the baseline are all on screen. Then point at the empty region around the Moon-base "
        "marker on the map - that is the next slide.")

    # 7 — key insight (dark)
    s = prs.slides.add_slide(layout)
    background(s, NAVY)
    title(s, "The Moon-base data gap", dark=True)
    text(s, 0.6, 1.55, 3.0, 2.6, str(K["solid_tests_near_lunar_base"]), size=200, bold=True, color=FLAME, font=HEAD)
    text(s, 0.6, 4.4, 5.2, 1.6,
         [[("real fabric or plastic samples", {"bold": True, "color": WHITE, "size": 24})],
          [("near the planned Moon-base atmosphere (≥30 % O₂, ≤70 kPa) — out of "
            f"{fmt(K['test_records'])} test records. Fabrics and plastics are a habitat's main fire load.", {"color": ICE})]],
         size=16, spacing_after=6)
    facts = [(f"{K['max_solid_o2_pct']:g} %", "highest O₂ any solid material was tested at — the Moon base needs ≥30 %"),
             ("1/6 g", "lunar gravity may be “nearly the most flammable condition” (Ferkul et al., LPSC 2026) — "
                             "and PSI has no partial-gravity data"),
             ("34 % / 8.2 psia", "a burner emulating PMMA burned steadily here (ACME BRE, Final Report 2022): "
                             "“a common plastic could steadily burn”"),
             ("Next", "test solids at 30–34 % O₂ and in partial gravity — NASA's planned FM2 experiment")]
    for i, (big, small) in enumerate(facts):
        y = 1.45 + i * 1.32
        box(s, 6.3, y, 6.45, 1.15, NAVY2, name=f"Fact {i + 1}")
        text(s, 6.55, y + 0.17, 1.95, 0.8, big, size=22 if len(big) < 8 else 18, bold=True, color=EMBER, font=HEAD,
             anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.55, y + 0.13, 4.0, 0.95, small, size=13, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, 7, dark=True)
    s.notes_slide.notes_text_frame.text = (
        f"This is the headline, and it is computed, not typed: zero solid-material tests near the Moon-base atmosphere. "
        f"Solids were tested only up to {K['max_solid_o2_pct']:g} percent oxygen, and no solid test even records its pressure. "
        "The challenge says 0 g and 1 g results 'could bound' partial gravity - but microgravity flame spread is most vigorous "
        "at low flow speeds (Olson and Ferkul's 'Goldilocks' zone, documented with PSI-62), exactly the weak buoyant flow at "
        "lunar gravity - NASA's newer analysis calls it nearly the most flammable condition (Ferkul et al., LPSC 2026). "
        "The closest evidence: the ACME Burning Rate Emulator, a gas burner built to mimic a PMMA-like solid, burned "
        "steadily at 34 percent oxygen and 8.2 psia and at 26.5 percent and 10.2 psia - its final report says a common "
        "plastic could steadily burn in these NASA proposed atmospheres. IGNIS tells mission planners what to test next.")

    # 8 — impact & team
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "Who it helps")
    cols = [("Researchers & students", "Find, compare and cite PSI studies in minutes — with the investigators' own words, documents and DOIs"),
            ("Mission planners", "See which habitat conditions are backed by flight data — and which are not, before materials are chosen"),
            ("Earth", "The same research led to water-mist extinguishers, smoke-vs-dust detectors and better firefighter gear (NASA spinoffs)")]
    for i, (head, body) in enumerate(cols):
        x = 0.6 + i * 4.1
        box(s, x, 1.65, 3.85, 2.25, TINT, name=f"Audience {i + 1}")
        badge(s, x + 0.25, 1.88, 0.55, str(i + 1), size=16, name=f"Audience number {i + 1}")
        text(s, x + 0.95, 1.92, 2.75, 0.5, head, size=18, bold=True, color=NAVY, font=HEAD)
        text(s, x + 0.25, 2.7, 3.35, 1.55, body, size=14, color=INK)
    box(s, 0.6, 4.3, 12.15, 1.6, NAVY, name="Team panel")
    text(s, 0.95, 4.52, 11.4, 1.3,
         [[("Team: ", {"bold": True, "color": EMBER}), ("[names and roles]", {})],
          [("Live demo: ", {"bold": True, "color": EMBER}), ("[Streamlit URL]", {}),
           ("     Code: ", {"bold": True, "color": EMBER}), ("github.com/fswierta/nasa-hackathon", {})],
          [("Data: NASA PSI, CC0 · every number cites its PSI DOI · AI use disclosed · not endorsed by NASA", {"color": ICE, "size": 13})]],
         size=16, color=WHITE, spacing_after=8)
    footer(s, 8)
    s.notes_slide.notes_text_frame.text = (
        "Three audiences: researchers and students who need to find and cite PSI studies; mission planners who need to know "
        "which habitat conditions are backed by data; and the public on Earth, where this research already led to better "
        "extinguishers, detectors and firefighter gear. Close with the challenge's words: help make the next giant leap a safer one.")

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print("wrote", build().relative_to(ROOT))
