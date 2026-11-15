# 7-slide presentation — IGNIS

**IGNIS — Intelligent Guidance for Navigating Ignition in Space** (title on slide 1, footer on every slide)

Format: one idea per slide, big numbers, a screenshot on slides 4–5. Speaker notes under each
slide (~30–40 s each). `[SCREENSHOT: …]` = pc3 fills in from the running app. Do not use the
NASA logo (Space Apps rules); NASA mission photos only if their licence allows and credited.

---

## 1 · The problem — fire is still the fastest way to lose a crew

**Title:** Fire behaves differently in space — and we are about to raise the oxygen.

- Apollo 1 (1967) · Apollo 13 (1970) · Mir (1997) · Artemis II smoke alarm
- No buoyancy → flames turn into blue spheres fed only by diffusion and airflow
- Planned Moon base & rovers: **≥30 % O2 at ≤10.2 psi** — "materials much more flammable"

*Notes:* Open with the Artemis II quote from the challenge text ("…make the fire alarm go off in
your spacecraft when you're still about 80,000 miles from home"). Then: the next habitats will
use more oxygen than Earth air, and the data that tells us how materials burn is hard to use.

---

## 2 · The data — a trove nobody can read in a weekend

- **24** combustion investigations in NASA PSI · **19** flown on ISS / Cygnus
- **66,567** files · **~13.5 TB** (raw data stays on PSI) · **352** publications
- 22 experiment tables — **every one with different columns and units**

*Notes:* mmHg vs atm vs bar, O2 as percent vs mole fraction, ranges inside cells. We found and
logged real errors in the source tables (spreadsheet auto-fill turned CH4 into CH5…CH14, "3400 %"
oxygen, a 0 mmHg pressure) — we never guessed a value; every fix is in `processed/etl_log.csv`.

---

## 3 · What we built

Pipeline: **PSI tables + documents → clean dataset → AI layer → dashboard**

- 1,947 test records in common units (O2 fraction · kPa · cm/s · mm)
- 4,631 document passages, each with file + page → retrieval for the AI
- Explainable ranking · condition map · cited Q&A
- **Two ML models on FLEX:** outcome classifier (76.5 % accuracy vs 68.6 % baseline, 261 tests; class imbalance handled by a
  goes-out-vs-burns-completely model: completion recall 30 % → 47 %, ROC-AUC 0.82) and
  burning-rate regression — both refuse to extrapolate beyond tested ranges
- **Computed similarity map** of all 24 studies: 7 clusters, labels from the data, recovers 14 of 18
  "related investigation" links that PSI curators wrote by hand

*Notes:* "SI-powered" in our app means: a neural embedding model + keyword search find the PSI
passages that answer a question, and we show them verbatim with file and page — no generated text
that could invent a number. The ranking is deterministic and shows its reasons; the prediction
model is fitted on FLEX data and refuses to extrapolate. (An LLM answer layer is built in and can
be switched on with an API key — it is off in this submission.)

---

## 4 · How we know the models work

- **Class imbalance (FLEX droplets):** only 27 of 261 burned completely; class weights alone found 30 % of them.
  'Disruption' (droplet broke up) is a test ending, not a fire outcome — the goes-out-vs-burns-completely model
  finds 47 % at 63 % precision (ROC-AUC 0.82), accuracy 88.5 % (baseline 86.9 %); a conservative threshold
  (tuned only inside training folds, nested CV) finds 62 % and drives a "may burn completely" warning.
- **AI search benchmark:** 40 questions with the source fixed before the run, half not naming the experiment.
  Hybrid search: right source first 88 %, in top 3 95 %, answer passage in top 3 90 %.
- Charts are native PowerPoint charts filled from `processed/analysis/flex_boundary.json` and
  `processed/ai/benchmark.json` at build time.

## 5 · Demo — rank, map, respond, ask

*Ranker:*
[SCREENSHOT: Scenario ranker, "Lunar base" preset, top 5 with score bars and "why"]

- Pick a habitat or type your own O2 / pressure / airflow / materials
- Score = O2 + pressure + airflow + material + pillar + relevance + data richness — weights adjustable
- Every result: why it ranks, what data is missing, DOI link

*Notes:* Show the Moon base preset: droplet and gas-flame studies (FLEX, CFI-G) rank high on
conditions, solids (BASS, Saffire) on materials — and the app says which part is missing.

*Map, response advisor, Ask:*

[SCREENSHOT: Condition map, O2 vs pressure, habitat markers]  [SCREENSHOT: chat answer with citations]

- Condition map: all tests vs ISS / Moon base / exploration atmosphere / spacesuit
- Ask: "Does flame spread faster in microgravity than in 1 g?" → best matching passages [1][2] → PDF page
- Similarity map: Saffire · BASS/MSG duct · droplets/MDCA · ACME gas flames · jet/coflow diffusion flames · smoke detection · ground models — found by the algorithm, not drawn by us

*Notes:* Click a citation live to show it opens the source file and page. On the similarity map,
hover BASS → nearest neighbours BASS-II, SLICE (same flow duct) with the concrete shared facts.

---

## 6 · Key insight — the Moon-base data gap

**0 solid-material tests near any planned exploration atmosphere.**

- Solids (fabrics, plastics) tested only up to **22.2 % O2**; Moon base needs **≥30 %**
- No solid test even records a pressure; no partial-gravity data in PSI at all
- **And the Moon may be worse than both 0 g and 1 g:** the challenge says 0 g / 1 g results "could bound"
  partial gravity, but microgravity flame spread peaks at low flow speeds (Olson & Ferkul's "Goldilocks"
  zone, PSI-62) — the kind of weak buoyant flow lunar gravity produces. NASA's FM2 (Flammability of
  Materials on the Moon) is planned to measure it. Source: Ferkul, Kralik, Liao & Johnston, "Material Flammability at Lunar and Martian Gravity", 57th LPSC 2026, NASA NTRS 20260001966, slide 17: lunar gravity "nearly the most flammable condition".
- (Inverse flames reach 85 % O2 — but that is burner gas, not cabin air; highest *atmosphere* O2 tested: 40 %)
- Closest evidence: **ACME BRE** — a gas burner emulating a PMMA-like solid — **burned steadily at 34 % O2 / 8.2 psia and 26.5 % O2 / 10.2 psia** (BRE Final Report 2022, pp. 25–26: "a common plastic could steadily burn under these NASA proposed atmospheric conditions")
- FLEX droplet data stops at ~70 kPa → our model refuses to predict for 56.5 kPa

*Notes:* This is the headline. Numbers come from `processed/coverage.json`, computed from the
PSI tables, not typed by hand. Frame it as: the dashboard tells mission planners what to test next
(e.g. future LUCI / FM2 experiments).

---

## 7 · Impact & team

- **Researchers / students:** find, compare and cite PSI studies in minutes
- **Mission planners:** see which habitat conditions are backed by data — and which are not
- **Earth:** water-mist extinguishers, smoke vs dust detection, firefighter gear (NASA spinoffs)
- Open data (CC0) · every number cites its PSI DOI · AI use disclosed · not NASA-endorsed

**Team:** [names / roles] · **Live demo:** [URL] · **Code:** https://github.com/fswierta/nasa-hackathon

*Notes:* End on: "make the next giant leap a safer one" (challenge text).
