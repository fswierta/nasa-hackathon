# 7-slide presentation

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
- Explainable ranking · condition map · cited Q&A · FLEX prediction model

*Notes:* "SI-powered" in our app means: a neural embedding model + keyword search find the PSI
passages that answer a question, and we show them verbatim with file and page — no generated text
that could invent a number. The ranking is deterministic and shows its reasons; the prediction
model is fitted on FLEX data and refuses to extrapolate. (An LLM answer layer is built in and can
be switched on with an API key — it is off in this submission.)

---

## 4 · Demo — rank investigations for *your* habitat

[SCREENSHOT: Scenario ranker, "Lunar base" preset, top 5 with score bars and "why"]

- Pick a habitat or type your own O2 / pressure / airflow / materials
- Score = O2 + pressure + airflow + material + pillar + relevance + data richness — weights adjustable
- Every result: why it ranks, what data is missing, DOI link

*Notes:* Show the Moon base preset: droplet and gas-flame studies (FLEX, CFI-G) rank high on
conditions, solids (BASS, Saffire) on materials — and the app says which part is missing.

---

## 5 · Demo — ask the data, see the gaps

[SCREENSHOT: Condition map, O2 vs pressure, habitat markers]  [SCREENSHOT: chat answer with citations]

- Condition map: all tests vs ISS / Moon base / exploration atmosphere / spacesuit
- Ask: "Does flame spread faster in microgravity than in 1 g?" → best matching passages [1][2] → PDF page

*Notes:* Click a citation live to show it opens the source file and page.

---

## 6 · Key insight — the Moon-base data gap

**0 solid-material tests near any planned exploration atmosphere.**

- Solids (fabrics, plastics) tested only up to **22.2 % O2**; Moon base needs **≥30 %**
- No solid test even records a pressure; no partial-gravity data in PSI at all
- Closest proxy: **ACME BRE** — a gas burner that emulates burning solids — at 21–40 % O2
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
