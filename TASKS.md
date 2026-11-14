# Tasks

Status: `[ ]` todo · `[~]` in progress · `[x]` done. **Edit only your own section.**
Contracts and folder ownership: see `CLAUDE.md`. Requirement codes (R1, S2, C3…) refer to
`project-data-1-pc1/AGENT_BRIEF.txt` §1.

Milestones: **M1** stubs everywhere so the app runs end-to-end with fake data ·
**M2** real data + AI in the app (MVP) · **M3** polish + submission.

---

## main (pc1) — Data / ETL  → `etl/`, `processed/`

- [x] **M1** `etl/build_investigations.py`: build a first `processed/investigations.json` from
      `investigations_catalog.json` + `metadata/*.json` (id, acronym, title, doi, landing_page,
      objective, publications, relevance tier from brief §4; other fields `null`). Push early — pc2/pc3 depend on it.
      *Done: 24 investigations, all with objective/hypothesis/approach/impacts/publications/DOI.
      Extra fields beyond the contract: hardware, sub_research_area, start, end, keywords, hypothesis,
      approach, research_impacts, contacts, license, total_size_gb, n_files, experimental_table_files.
      Rebuild: `py etl/parse_tests.py && py etl/build_investigations.py`.*
- [x] **M2** Per-table parsers for the 22 `experimental_tables/*.csv` → `processed/tests.csv`
      (normalize O2 → mole fraction, pressure → kPa, flow → cm/s; handle ranges, BOMs, footnote rows,
      typos — brief §5). Start with tier-A: BASS, BASS-II, SAFFIRE I–III, FLEX, SAME, SAME-R.
      *Done: `etl/parse_tests.py`, 21 tables → 1947 rows (PSI-142 skipped: it lists datasets, not tests).
      Data errors found in the PSI files and handled (all in etl_log.csv): CLD fuel column Excel auto-fill
      (CH5…CH14, C2H11 → fuel empty), s-Flame "N3", BRE "3400%" (=34 %), FLEX cp1252 + "0 mmHg",
      SAFFIRE-II "037 mm" (ambiguous → empty), duplicate rows in SAFFIRE-III/SPICE/DAFT.*
- [x] **M2** Log unparsed cells to `processed/etl_log.csv`; unit tests in `etl/tests/` (8 passing).
- [x] **M2** Fill o2/pressure/flow ranges, fuels, fuel_phase, varied vs fixed variables, n_tests in
      `investigations.json` from the parsed tests (C2). *`measurements` (sensors) still null — not in
      the tables; pc2 can extract them from the PDFs into the AI profiles.*
- [ ] **M3** Data-coverage stats for the "About the data" page (tests per investigation, gaps such as "no partial-g data").

## pc2 — AI layer  → `ai/`, `processed/ai/`

- [x] **M1** `ai/habitats.py` with presets (ISS 21%/101.3 kPa, Lunar base 30%/70.3 kPa,
      Exploration atm 34%/56.5 kPa, Mars base, Pressurized rover, Spacesuit 100%/29.6 kPa) — brief §2.
- [x] **M1** `rank()` and `ask()` — skipped stubs, shipped real versions directly (see `ai/README.md`).
- [~] **M2** Text extraction + chunking + embeddings of `data/documents/**` and the Researcher's Guide
      (keep file + page for every chunk) → `processed/ai/index/`.
- [~] **M2** `ask()` with RAG (code done; untested with a real key); every answer cites PSI-ID + DOI + file/page (R2, R5).
- [~] **M2** LLM-generated `processed/ai/profiles/PSI-XX.json` for all 24 (summaries, findings with
      sources, pillar, habitat relevance) — cache, don't regenerate on every run (R3).
- [x] **M2** Real `rank()`: explainable weighted score (brief §6.2) using `processed/investigations.json` (R4, S1).
- [ ] **M3** Cross-investigation insights for prevention / detection / suppression (S3); optional
      simple predictive model on FLEX or BASS (S2).

## pc3 — Dashboard + submission  → `app/`, `submission/`

- [ ] **M1** `app/main.py` Streamlit skeleton with pages; use `investigations_catalog.json` directly
      until `processed/investigations.json` exists (R1).
- [ ] **M2** Overview (cards/table + filters) and Investigation detail (summary, conditions,
      varied vs fixed, sensors, documents, DOI, link to PSI) (R3, C2).
- [ ] **M2** Scenario ranker page: habitat presets + custom inputs + weight sliders, show score
      breakdown and "why" (R4, S1).
- [ ] **M2** Condition map: scatter of all tests O2 vs pressure (flow as size), habitat atmospheres overlaid (S2).
- [ ] **M2** "Ask the data" chat calling `ai.qa.ask()` with clickable citations (R2).
- [ ] **M3** Compare view (2–4 side by side, C3), Insights, Earth benefits (C1), About-the-data page (PSI attribution, CC0, DOIs).
- [ ] **M3** `submission/`: project summary, 7-slide deck or ≤30 s video script, AI-use disclosure,
      deployment (Streamlit Cloud / HF Spaces). Check rules at spaceappschallenge.org/resources.

---

## Requests between PCs

Write here when you need something from another PC's folder (who → who: what).

- pc2 → main: `rank()` needs `o2_range`, `pressure_range_kPa`, `flow_range_cm_s`, `fuels` in `processed/investigations.json`; until then it uses the AGENT_BRIEF §4 seed in `ai/catalog.py` (many ranges are null).
  **main: done (13:01)** — ranges/fuels are filled for every investigation with a table; checked
  that `rank(HABITATS["Lunar base"])` uses them. Ranges stay null only where the PSI table has no such column.
- 
