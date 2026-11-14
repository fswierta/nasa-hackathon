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
- [x] **M3** `processed/coverage.json` — data coverage + gaps for the condition map and About page.
      *Done: `py etl/build_coverage.py` (habitat values imported from `ai/habitats.py`). Extra per-habitat
      field `o2_match_pressure_unknown` = rows whose O2 matches but whose table has no pressure (all solid
      tables). Gaps have `text` + `evidence` (+ `closest_proxy`, `habitats`). Key result: 0 solid-material
      tests near any exploration atmosphere; solids tested only up to 22.2 % O2; closest proxy ACME BRE (PSI-20).*
      Shape (pc3 can build against it now):
      `{"habitats": [{"name", "o2_frac", "pressure_kPa", "n_tests_near", "by_investigation": {"PSI-69": 17},
      "by_fuel_phase": {"solid": 0, "droplet": 17, "gas": 17}}], "gaps": [{"text", "evidence"}],
      "per_investigation": [{"id", "table_rows", "has_o2", "has_pressure", "has_flow"}]}`
      "near" = O2 within ±0.06 mole fraction and pressure within ±25 % of the habitat value.

---

## Coordinator notes (from main) — read before starting · updated 13:25

Order of work: **pc2 unblocks the AI data → pc3 builds pages on real data → everyone polishes for the demo.**
Data is ready: `processed/investigations.json` (24) and `processed/tests.csv` (1947 rows) — columns in
CLAUDE.md "Contracts". Do not wait on each other: if something you need is missing, show
"not available yet" in the UI / skip it, and write a line under **Requests**.
Definition of done for every task: runs from the repo root, has a test or a screenshot-checked page,
`TASKS.md` box ticked with a one-line note, pushed (finishing a Claude turn pushes automatically).

## pc2 — AI layer  → `ai/`, `processed/ai/`

**Setup blockers first (needs the human):** install Python 3.11+ (python.org, tick "Add to PATH"),
put `ANTHROPIC_API_KEY=...` in `.env`, then `py -m pip install -r ai/requirements.txt python-dotenv`
and `py -m pytest ai/tests -q`.

- [x] **M1** `ai/habitats.py` with presets (ISS 21%/101.3 kPa, Lunar base 30%/70.3 kPa,
      Exploration atm 34%/56.5 kPa, Mars base, Pressurized rover, Spacesuit 100%/29.6 kPa) — brief §2.
- [x] **M1** `rank()` and `ask()` — skipped stubs, shipped real versions directly (see `ai/README.md`).
- [x] **M2** Text extraction + chunking + embeddings of `data/documents/**` and the Researcher's Guide
      (keep file + page for every chunk) → `processed/ai/index/`. *4631 chunks; index rebuilt locally via `python -m ai.ingest`.*
- [~] **M2** `ask()` with RAG (code done; untested with a real key); every answer cites PSI-ID + DOI + file/page (R2, R5).
- [~] **M2** LLM-generated `processed/ai/profiles/PSI-XX.json` for all 24 (summaries, findings with
      sources, pillar, habitat relevance) — cache, don't regenerate on every run (R3).
- [x] **M2** Real `rank()`: explainable weighted score (brief §6.2) using `processed/investigations.json` (R4, S1).
- [ ] **M3** Cross-investigation insights for prevention / detection / suppression (S3); optional
      simple predictive model on FLEX or BASS (S2).

**Next tasks from main, in this order:**
- [x] **P2-1 (no key needed — do first)** `py -m ai.ingest` → commit `processed/ai/chunks.jsonl`.
      Check: every chunk has psi_id + doi + file + page (page may be null only for DOCX); print
      chunks per investigation; list files that failed to extract (`.doc`, `.zip`) in `ai/README.md`.
      pc3's chat works in "no-llm" mode as soon as this file is pushed.
      *Done: 4631 chunks, 0 missing psi_id/doi/file; page null only for metadata + DOCX. Counts and the
      14 skipped files (.doc/.zip/.xlsx) listed in `ai/README.md`. Fixed: `ask()` crashed with no key → now no-llm mode.*
- [ ] **P2-2** ⛔ blocked: no API key in `.env` yet. With the key: `py -m ai.profiles` for all 24 → commit `processed/ai/profiles/PSI-XX.json`.
      Generate PSI-26, PSI-69, PSI-98 first, read them, fix the prompt, then the rest.
      Add a `measurements` list (sensors/diagnostics: cameras, radiometers, thermocouples, gas analyzers,
      particle counters…) with a source for each — the tables don't have it (C2).
- [ ] **P2-3** ⛔ blocked (key). Test `ask()` with the key on 5 questions; save Q + answer + citations to
      `ai/examples.md` (pc3 uses them in the demo and slides). Suggested:
      "Does flame spread faster in microgravity than in 1 g?", "Which materials were tested in Saffire?",
      "What O2 level stopped droplet burning in FLEX?", "How does smoke differ in microgravity (SAME)?",
      "What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?"
- [ ] **P2-4** `processed/ai/insights.json`: 3–5 insights per pillar (prevention / detection /
      suppression), each `{"pillar", "title", "text", "citations": [...], "habitats": [...]}` (S3, R5).
      Include the main data gap: no solid-material tests near Moon-base atmosphere (see main's
      `processed/coverage.json` when it lands). Also an `"earth_applications"` list (C1).
- [x] **P2-5 (optional, S2)** `ai/model.py`: simple regression on FLEX rows of `processed/tests.csv`
      (`burning_rate_k_mm2_s` and `extinction_diameter_mm` vs `o2_frac`, `pressure_kPa`, `diluent`,
      `d0_mm`; 258 / 197 usable rows) with cross-validated error. Function
      `predict_flex(o2_frac, pressure_kPa, diluent, d0_mm) -> {"k", "k_low", "k_high", "n_train", "valid_range"}`;
      refuse to predict outside the tested ranges (return `None` + reason).
      *Done: `ai/model.py`. CV R² 0.70 (K, n=257) / 0.72 (extinction D, n=188). Uses CO2/He mole fractions
      from raw_row. Refuses 56.5 kPa (Exploration atm) — FLEX min ≈ 70 kPa: a real data gap for P2-4. Also `model_card()`.*
- [x] **P2-6** Speed: `rank()` < 50 ms, `ask()` cached per question; `load_profile()` never calls the LLM.
      *Done: rank() ≈ 1 ms (tested); ask() disk cache `processed/ai/qa_cache.json`; first search ~7 s
      (embedding model load) → pc3 call `ai.retrieval.warmup()` once in `st.cache_resource`, then ~0.5 s.*

## pc3 — Dashboard + submission  → `app/`, `submission/`

**Setup:** your branch must be `pc3` (`git branch --show-current`; if it says `main`:
`git checkout pc3 && git merge main`). `py -m pip install -r app/requirements.txt -r ai/requirements.txt`.
Run: `py -m streamlit run app/main.py` from the repo root. At the top of `app/main.py` add the
repo root to `sys.path` so `import ai` works. Data is real — do **not** build fake data.
APIs you call (details in `ai/README.md`): `ai.habitats.HABITATS`, `ai.ranking.rank(scenario, weights)`,
`ai.ranking.DEFAULT_WEIGHTS`, `ai.qa.ask(question)`, `ai.profiles.load_profile(id)` (may return `None`
until pc2 generates profiles — then show the metadata objective instead).
Use Streamlit multipage (`app/main.py` + `app/pages/1_Overview.py` …) and one `app/data.py` with
`@st.cache_data` loaders for the JSON/CSV files. Plotly for charts.

- [ ] **P3-1** Skeleton + `app/data.py` + home page: one-paragraph pitch, 3 numbers
      (24 investigations · 1947 test records · N documents), links to the pages. (R1)
- [ ] **P3-2** Overview: table/cards of the 24 investigations with filters (fuel_phase, relevance_tier,
      hardware_family, flight/ground, O2 range slider). Click → detail. (R3)
- [ ] **P3-3** Investigation detail (`?id=PSI-26`): title, acronym, tier, DOI as a link, PSI landing page
      link, objective / hypothesis / approach / impacts (metadata), AI summary + findings with citations
      when the profile exists, conditions (o2/pressure/flow ranges), **varied vs fixed variables**,
      fuels, its rows from `tests.csv` as a table, publications. (R3, C2)
- [ ] **P3-4** Scenario ranker: habitat preset dropdown + editable O2 / pressure / flow / materials /
      pillar, weight sliders starting at `DEFAULT_WEIGHTS`. Show the ranked list with score bar,
      breakdown (show `None` as "no data", never as 0), `why`, `gaps`, and the habitat's `assumptions`
      + `source`. (R4, S1) — this is the demo's main page, make it look best.
- [ ] **P3-5** Condition map: scatter of `tests.csv` rows with both O2 and pressure (x = pressure kPa,
      y = O2 %, colour = investigation or fuel phase, hover = test_id / fuel / outcome); habitat
      atmospheres as labelled markers (ISS, Lunar base, Exploration atm, Spacesuit…). Highlight the empty
      region around Moon-base atmospheres for solids, using `processed/coverage.json` when it exists. (S2)
- [ ] **P3-6** Ask the data: chat box → `ask()`; render `[n]` as links to the citation
      (PSI-ID, DOI link, file, page); show a note when `mode == "no-llm"`. (R2, R5)
- [ ] **P3-7** About the data: PSI attribution, CC0 licence, table of all 24 DOIs, data errors we found
      (from `processed/etl_log.csv`, grouped), "not endorsed by NASA", AI-use disclosure. (Validity)
- [ ] **P3-8 (M3)** Compare (2–4 investigations side by side, C3); Insights page from
      `processed/ai/insights.json` (S3, C1).
- [ ] **P3-9 (M3)** `submission/`: project summary (≤ 200 words), 7-slide outline
      (problem → data → what we built → demo screenshots → key insight: Moon-base data gap → impact → team),
      ≤30 s video script, AI-use disclosure. Check rules at spaceappschallenge.org/resources.
- [ ] **P3-10 (M3)** Deploy to Streamlit Community Cloud (API key as a secret, never in the repo); put the URL in README.

---

## Requests between PCs

Write here when you need something from another PC's folder (who → who: what).

- pc2 → main: `rank()` needs `o2_range`, `pressure_range_kPa`, `flow_range_cm_s`, `fuels` in `processed/investigations.json`; until then it uses the AGENT_BRIEF §4 seed in `ai/catalog.py` (many ranges are null).
  **main: done (13:01)** — ranges/fuels are filled for every investigation with a table; checked
  that `rank(HABITATS["Lunar base"])` uses them. Ranges stay null only where the PSI table has no such column.
- main → pc2 (13:42): bug in `ai/llm.py:31` — `json_call()` does `import anthropic` *before* `_client()`,
  so on a PC without the package `ask()` raises `ModuleNotFoundError` instead of falling back to no-llm
  mode (reproduced on main: no `.env`, no anthropic). Fix: call `_client()` first (or move the import
  into the try). Please add a test with the package hidden.
- 
