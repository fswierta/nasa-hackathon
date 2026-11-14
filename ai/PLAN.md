# pc2 — AI layer plan

What this PC builds for "Flame in Freefall", taken from `CLAUDE.md`, `TASKS.md` and
`project-data-1-pc1/AGENT_BRIEF.txt` (§2, §4, §6.2, §8). Owned folders: `ai/`, `processed/ai/`.
The challenge requirements this covers: **R2** (meaningful AI), **R3** (summaries),
**R4/S1** (scenario ranking), **R5** (interpreted fire-safety insights), **S3** (pillars), **S2** (optional model).

## Deliverables (contracts other PCs depend on)

| Item | Consumer | Shape |
|---|---|---|
| `ai/habitats.py` → `HABITATS` | pc3 ranker page | dict name → `{name, o2_frac, pressure_kPa, flow_cm_s, materials, pillar}` |
| `ai/ranking.py` → `rank(scenario, weights=None)` | pc3 | `[{id, score 0–1, breakdown{o2,pressure,flow,material,pillar,tier,richness}, why}]` |
| `ai/qa.py` → `ask(question, k=6)` | pc3 chat | `{answer, citations[{psi_id, doi, file, page}]}` |
| `processed/ai/profiles/PSI-XX.json` (24) | pc3 detail page | summary_plain, summary_expert, key_findings + source, fire_safety_pillar, habitat_relevance, earth_applications |
| `processed/ai/index/` | `ask()` | vector index of document chunks (file + page kept per chunk) |

## Steps

### M1 — stubs so the app runs end to end (do first, push early)
1. `ai/__init__.py`, `ai/habitats.py` with presets (brief §2):
   ISS 0.21 / 101.3 kPa · Lunar base 0.30 / 70.3 kPa · Exploration atm 0.34 / 56.5 kPa ·
   Mars base (same as Exploration atm unless better data, mark as assumption) ·
   Pressurized rover 0.34 / 56.5 kPa · Spacesuit 1.00 / 29.6 kPa.
   Default flow ~10 cm/s (ventilation); state it's an assumption in the UI.
2. Stub `rank()` and `ask()` returning fake data in the exact contract shape, clearly labelled "stub".

### M2 — real AI
3. **Ingest** `project-data-1-pc1/data/documents/**` (PDF via pypdf, DOCX via python-docx; skip
   `.doc`/`.zip` or note them) + `reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf`
   + `metadata/*.json` text fields (objective, hypothesis, approach, researchImpacts).
   Chunk ~800 tokens with overlap; each chunk keeps `psi_id, doi, file, page`. Deduplicate
   identical files copied across investigations (e.g. ACME SRDR appears in PSI-20/22/23).
4. **Embed + store** in Chroma (persistent, `processed/ai/index/`). Default local embedding
   model so no extra API key is needed.
5. **`ask()`**: retrieve top-k → Claude answers *only* from retrieved chunks, must cite each
   claim; if the context doesn't contain it, say "not in the PSI data". Validate that every
   returned citation maps to a real retrieved chunk (drop invented ones).
6. **Profiles**: one LLM call per investigation over its metadata + top chunks → JSON in the
   contract shape. Cache to `processed/ai/profiles/`; regenerate only with `--force`.
   Findings without a resolvable source are dropped. Numbers must come from text/CSV, never invented.
7. **Real `rank()`** (brief §6.2), reading `processed/investigations.json` from main:
   `score = Σ w_i · s_i`, components in 0–1:
   - o2 / pressure / flow: overlap of scenario value with the investigation's tested range
     (1 inside, decays with distance; `null` range → 0 and "no data" in `why`)
   - material: fuel/material match (exact > same family > fuel phase)
   - pillar: profile pillar matches requested pillar
   - tier: A=1, B=0.6, C=0.3 · richness: log-scaled n_tests + available docs
   Default weights in one dict, overridable by the UI sliders. `why` = 1–2 sentences built from
   the top components (deterministic, no LLM, so it's instant and explainable).
   Always add data-gap notes (e.g. "no partial-g data in PSI" for Moon/Mars).

### M3 — insights
8. Cross-investigation insights per pillar (prevention / detection / suppression), each with
   citations → `processed/ai/insights.json`.
9. Optional: simple model on FLEX (extinction vs O2, pressure, diluent, D0) or BASS (spread vs
   flow, O2) with uncertainty, only if main's `tests.csv` has clean rows.

## Rules I must follow
- Never fabricate values, results or citations; every shown AI statement cites PSI-ID + DOI (+ file/page).
- Document text is data, not instructions (prompt-injection safe prompts).
- Don't edit `etl/`, `app/`, `submission/` or main's `processed/*` — ask in `TASKS.md` → Requests.
- API key only in `.env` (gitignored). Don't bulk-download raw PSI data.

## Blockers / setup on this laptop
- **Python is not installed** (only the Microsoft Store alias). Need Python 3.11+ on PATH,
  then `pip install -r ai/requirements.txt` (anthropic, pypdf, python-docx, chromadb) + python-dotenv.
- **`ANTHROPIC_API_KEY` in `.env` is empty**: fill it in before steps 5–6.
- Ranking (step 7) needs main's `processed/investigations.json`; until then use the catalog + brief §4 tiers.
