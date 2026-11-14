# ai/ — owned by pc2

RAG index, LLM profiles, scenario ranking and Q&A. Plan: [PLAN.md](PLAN.md). Contracts: `CLAUDE.md`.
Run everything from the repo root.

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -r ai/requirements.txt   # Windows: .venv\Scripts\...
# optional LLM key in .env (gitignored): GEMINI_API_KEY=... (free tier) or ANTHROPIC_API_KEY=sk-ant-...
python -m ai.ingest          # extract PDFs/DOCX -> processed/ai/chunks.jsonl + embeddings.npy (only when docs change)
python -m ai.vectors --check # vector-search status
python -m ai.profiles        # profiles -> processed/ai/profiles/PSI-XX.json (cached; --force to redo)
python -m pytest ai/tests -q # no key needed
```

Everything search needs is committed, so a **fresh clone works**: `chunks.jsonl`, and the chunk embeddings
`embeddings.npy` (float16, 3660 unique texts × 384, 2.8 MB) + `embeddings_meta.json` (model, chunk→row map,
fingerprint of chunks.jsonl). Vector search is numpy cosine (#21); the only download is the ~80 MB ONNX
MiniLM query encoder on first use. Identical texts (one PDF filed under several investigations) share one
vector and come back as one hit with `also_in: [other PSI-IDs]`. If the embeddings are missing/stale or the
encoder can't load, `ai.retrieval.status()["vector"]["warning"]` says so and search uses BM25 only.
The local Chroma index is optional (`python -m ai.ingest --chroma`, gitignored).

## API for pc3 (dashboard)

```python
from ai.habitats import HABITATS        # ISS, Lunar base, Exploration atmosphere, Mars base, Pressurized rover, Spacesuit
from ai.ranking import rank, DEFAULT_WEIGHTS
from ai.qa import ask
from ai.profiles import load_profile

rank(HABITATS["Lunar base"])                       # 24 results, best first
rank({**HABITATS["ISS"], "pillar": "detection"}, weights={"tier": 0.3})
# -> [{"id", "acronym", "doi", "score", "breakdown": {o2,pressure,flow,material,pillar,tier,richness,gravity}, "why", "gaps": [...]}]
#    breakdown values are 0-1, or None = "no data on record" (show it as such, don't show 0)
#    gravity (#37): scenario["gravity"] micro -> flight 1.0 / ground 0.5; earth -> ground 1.0 / flight 0.3;
#    lunar / mars -> None for every investigation (PSI has no partial-g data; 0 g / 1 g may not bound it)

ask("Does flame spread faster in microgravity than in 1 g?")
# -> {"answer": "... [1] ... [2]", "citations": [{"psi_id","doi","file","page"}], "mode": "llm"|"no-llm"}
#    [n] in the answer = citations[n-1]; "no-llm" = no API key, answer is raw excerpts

load_profile("PSI-26")   # contract 3 dict, or None if not generated yet
```

```python
import ai.retrieval; ai.retrieval.warmup()   # call once at start-up (st.cache_resource): ~7 s, then ask() ~0.5 s
from ai.model import predict_flex, model_card
predict_flex(o2_frac=0.21, pressure_kPa=101.3, diluent="N2", d0_mm=3.0, fuel="heptane")
# -> {"k": 0.534, "k_low": 0.386, "k_high": 0.681, "unit": "mm²/s", "n_train": 257, "cv_r2": 0.70,
#     "valid_range": {...}, "reason": None, "source": {"psi_id": "PSI-69", "doi": ...}}
predict_flex(0.34, 56.5, "N2", 3.0, "heptane")   # Exploration atm -> k=None, reason="pressure ... outside ..."
predict_flex(..., target="extinction_d")         # extinction diameter (mm) instead of K
model_card()                                     # coefficients + CV error for an "about the model" box
```

**No-llm answers** (no API key): `answer` is Markdown — `**Best match [1]** — <best sentence>` with question
terms in bold, then `**Supporting passages:**` `- [n] <sentence> — *PSI-ID · file, p.*`, then an italic note
that it's verbatim PSI text. `ask()` also returns `passages: [{n, sentence, snippet, psi_id, doi, file, page}]`
(same order as `citations`) if you'd rather render them yourself. Sentences are picked from up to 18 retrieved
passages by rare-term overlap + result wording; citations, captions, procedure steps and hypotheses are skipped.
Worked examples with citation checks: [examples.md](examples.md) (`python -m ai.examples`).

`ask()` answers from the LLM are cached in `processed/ai/qa_cache.json` (committed), so demo
questions work instantly and without a key once they've been asked once.

Habitat entries carry `assumptions` (fields not from PSI data, e.g. default 10 cm/s airflow) and
`source`. Show both in the UI.

## ML Lab (`ai/mllab.py` + page `app/views/ml_lab.py`, issue #47)

Classical, cross-validated ML on `analysis.datasets` (FLEX droplets, CFI-G gaseous flames) — no LLM.
`run(config)` → metrics (mean ± sd over folds) next to a baseline evaluated in the same folds, confusion
matrix (first repeat, each test once), per-class recall, permutation importance (batched: all shuffled
copies of a fold predicted in one call), warnings; `learning_curve(config)`; `predict(config, inputs)` →
probabilities / value + CV error + 5 nearest real tests, refusing inputs outside the tested min–max;
`mutual_information`, `correlations`, `default_config`, `shipped_config` (= `analysis.flex_classifier`).
Leaking features (`post_test` or catalog `leak` contains the target) raise `FeatureRefused` unless
`leakage_demo=True` (result flagged, page shows a red LEAKAGE DEMO banner).

**Precomputed:** `python -m ai.mllab --precompute` → `processed/ai/mllab_runs.json` (default config of every
dataset × target + the shipped model, with learning curves). Keys = config (features in catalog order) +
a fingerprint of the dataset rows, so an ETL change makes them unreachable instead of stale; incremental.
The page opens instantly; changed settings train live (time shown). Warnings are recomputed on every call.

| config (5-fold × 3 CV, random forest, 50 trees, balanced) | score | baseline |
|---|---|---|
| cfig `self_extinguished` | accuracy 0.866 ± 0.035 | 0.665 |
| cfig `hot_ignition` | accuracy 0.832 ± 0.029 | 0.814 |
| flex `outcome` | accuracy 0.706 ± 0.071 (balanced 0.609) | 0.686 |
| flex `went_out` | accuracy 0.833 ± 0.053 — **below** the 0.869 baseline (page says so) | 0.869 |
| flex `burning_rate_k_mm2_s` / `extinction_diameter_mm` | R² 0.664 / 0.745 | ≤ 0 (mean) |
| shipped model (flex `outcome`, 300 trees, 5 × 5) | 0.703 ± 0.057 (flex_classifier: 0.707) | 0.686 |

Timing on this laptop (measured slow: a 100-tree forest fit on 200 rows takes 0.55 s): a live default run
takes ~8–15 s; precomputed ones load instantly. Forest threads (`n_jobs`) and fold threads were measured
and did not help at this data size (GIL-bound per-tree overhead), so they are off.

## Data cards (`processed/ai/datacards.jsonl`, `python -m ai.datacards`, issue #19)

One synthetic chunk per investigation, written as plain sentences from `processed/investigations.json` +
`processed/tests.csv`: test count, fuels/materials, geometries, O₂ levels (percent + mole fraction),
pressure (kPa + atm), flow speed, diluents, varied vs fixed variables, outcomes, the O₂ range of tests that
ended in extinction and the lowest O₂ that still burned to completion, measured ranges (droplet diameter,
spread rate, burn time…). Source label `processed/investigations.json (PSI experimental table)` + the
investigation's DOI. `load_chunks()` merges them into BM25 and the vectors, so factual questions hit real
values ("How many tests did BASS-II run?" → *"BASS-II (PSI-25) ran 129 tests…"*).

**Re-run after any ETL change:** `python -m ai.datacards && python -m ai.vectors` (the vector fingerprint
covers the cards; `test_datacards_are_current_and_match_etl` fails if the cards are stale).
Factual lookups ("which…", "how many…", "what O₂/pressure/material…" — not "what is known about…") get a
small boost for metadata + data-card chunks in retrieval and for data-card sentences in the extractive answer.
LLM profiles get the card as excerpt 1, so they state *tested* conditions, not plans or facility limits.

Page evidence (`python -m ai.screenshot <url> <png> --wait-for <text>`, headless Edge/Chrome over DevTools):
`ai/screenshots/ask_saffire_materials.png` (Gemini, three distinct sources incl. the data card),
`ask_nollm_bass_ii_tests.png` (no key: extractive, data card leads), `ask_nokey_flex_o2_stop_cached.png`
(no key: the committed Gemini answer is served from cache and labelled as such).

## LLM providers (`ai/llm.py`, issue #26)

| `FIF_LLM` | used when | model (`FIF_MODEL` overrides) |
|---|---|---|
| `gemini` | default if `GEMINI_API_KEY` is set | `gemini-3.8-flash`, falls back to `gemini-3.5-flash-lite` on quota (429) or 503 "high demand" |
| `anthropic` | default if only `ANTHROPIC_API_KEY` is set | `claude-opus-5-5` |
| `none` | no key, or forced | everything runs extractive / rule-based, exactly as without #26 |

`ai.llm.provider()` / `model_name()` tell the app what is active. `json_call()` raises `LLMUnavailable` on
quota, server, network or schema problems and callers fall back — nothing crashes. `ask()` returns
`mode` = provider name and `model`; generated answers are cached in `processed/ai/qa_cache.json`
(`python -m ai.prewarm` pre-warms the 5 demo questions → [examples_llm.md](examples_llm.md)).
**Gemini free tier:** questions + retrieved passages are sent to Google and may be used to improve its
products — the app and `submission/ai_disclosure.md` must say so (pc3 issue). A Gemini timeout is 25 s per
model with one retry, so a Google-side outage costs at most ~1 min before the extractive answer appears.
Live test: `test_gemini_live_smoke` runs one tiny call when a key exists (skips on transient outages).

## Profiles (contract 3) — `load_profile(pid)` picks the best available

| folder | built by | provenance `generated.by` |
|---|---|---|
| `processed/ai/profiles/` | `python -m ai.profiles --llm --force` (Gemini free tier via `ai/llm.py`) | `Google Gemini (free tier)` |
| `processed/ai/profiles_rules/` | `python -m ai.profiles --force` (no key) | `rules (no LLM)` |

`load_profile()` returns the LLM file when present, else the rule-based one — same keys, so the app only
reads `generated["by"]` to label provenance. Both are committed; the demo never calls the API live.

**LLM profiles (#30):** the model sees ~20 retrieved excerpts for the investigation. Guards, all recorded
in `generated.validation`: a finding is dropped if its excerpt number is invalid **or it contains a number
absent from the cited excerpt**; `summary_expert` sentences with numbers absent from every excerpt are
dropped; `summary_plain` falls back to the rule-based one if it invents a number; values from plans /
requirements documents must be called "planned". `measurements` (sensors) always come from the rule-based
detector (`measurements_source`), which proved more reliable. `habitat_relevance` is the model's judgement;
`habitat_relevance_rules` keeps the `rank()`-based score next to it.

**Rule-based profiles** (`ai/profile_rules.py`) — contract-3 fields plus: `summary_source`,
`key_findings_note`, `habitat_relevance_note`, `earth_applications_source`, `measurements`, and `score` on each finding.
- `summary_plain` / `summary_expert`: quoted from metadata `objective` (+ `approach`).
- `key_findings`: verbatim result sentences, scored by `textutil.finding_score` (rewards "results showed",
  "was found", numbers with units; penalises captions, prior work, hardware, data-portal text). Sources in
  order: own documents → Researcher's Guide passages about the investigation (`psi_id: "REF"`) →
  metadata researchImpacts → strong statements in metadata approach. Fewer than 3 →
  `key_findings_note` explains why (9 investigations: metadata-only or no result statements in PSI docs).
- `measurements`: keyword dictionary (22 instrument types) → `{name, type, matched, source}`.
- `fire_safety_pillar`: regex rules over sub_research_area + keywords + title (`modeling` if none match).
- `habitat_relevance`: `rank()` score / best score for that habitat (`spacecraft` = Exploration atmosphere).

## Insights (`processed/ai/insights.json`, `python -m ai.insights`)

No LLM, no hand-written claims. 18 insights + 12 Earth applications, shape:
`{"insights": [{"id", "pillar": "prevention|detection|suppression", "kind": "evidence|data|data_gap",
"title", "text", "habitats": [...], "numbers": {...}, "citations": [{"psi_id", "doi", "file", "page", "quote"?}]}],
"earth_applications": [{"text", "citations"}]}`.
- `title` / `text` are templates (in `SPECS`) filled with numbers computed at build time from
  `processed/investigations.json`, `tests.csv`, `coverage.json`, the profiles and `ai.model` (`numbers` holds them).
- Every insight is backed by 1–3 verbatim sentences picked from `search()` results for a fixed query
  (citations with `quote`; render them as the evidence). Literature-review sentences ("their results…"),
  citations, plans and hypotheses are excluded; optional `must` regex keeps quotes on topic. Each quote is
  re-verified against `chunks.jsonl`; the build fails if a quote doesn't match or an insight has no quote.
- Citations without `quote` point at the computed-data file behind the numbers.
- `data_gap` (7): main's `coverage.json` gaps + the FLEX pressure gap (no data below ~70 kPa).
- Earth applications: verbatim `researchImpacts` sentences (`ACRONYM: “quote”`).
The Researcher's Guide citation has `psi_id: "REF"` and `doi: null` (it is not a PSI dataset).

## Knowledge base coverage (`processed/ai/chunks.jsonl`)

4631 chunks, all with `psi_id`, `doi`, `file`; `page` is null only for metadata fields and DOCX.
Sources: 85 PDFs, 9 DOCX, metadata text of all 24 investigations, the 2015 Researcher's Guide (`psi_id = "REF"`).
Chunks per investigation: PSI-20 1350 · PSI-22 764 · PSI-23 641 · PSI-10 496 · PSI-117 237 · PSI-101 171 ·
PSI-102 165 · PSI-68 161 · PSI-107 148 · PSI-159 122 · PSI-39 107 · REF 56 · PSI-115 39 · PSI-25 27 ·
PSI-26 26 · PSI-47 24 · PSI-69 22 · PSI-99 19 · PSI-98 18 · PSI-100 18 · PSI-21/60/62/106/142 4 each (metadata only).

Not extracted (format not supported):
- `.doc`: PSI-107 SPICE_SRD, PSI-47 DAFT_1YearReport, PSI-47 DAFT_FinalReport
- `.zip`: PSI-20 ACME BRE Data interpretation, PSI-22 ACME Data interpretation, PSI-22 ACME Gas flow
- `.xlsx` (tabular; the test tables are already in `processed/tests.csv`): SAFFIRE-I/II/III MergeResult,
  BASS As-Run / Planned test matrix, BASS-II Test.Matrices, CFI DataSummary

## FLEX model (`ai/model.py`, S2)

Linear least squares on 257 (K) / 188 (extinction diameter) FLEX droplet tests:
`target ~ fuel + O2 + ln(pressure) + x_CO2 + x_He + D0`. 5-fold CV: K RMSE 0.075 mm²/s (R² 0.70),
extinction diameter RMSE 0.52 mm (R² 0.72). Band = ±1.96·CV-RMSE. Inputs outside the per-fuel tested
box are refused, so it can't extrapolate to low-pressure exploration atmospheres (FLEX min ≈ 70 kPa).
Ranges are checked per variable, not jointly; treat combinations far from tested points with care.

## Data sources and honesty rules

- Investigations come from `processed/investigations.json` (main's ETL) when it exists; otherwise
  from the raw catalog + metadata + the `SEED` table in `catalog.py` (values copied from
  AGENT_BRIEF §4 only; everything else `None`). ETL values override the seed.
- Every `ask()` citation points to a chunk that was actually retrieved; citations the model
  invents are dropped. Profile findings without a valid excerpt are dropped.
- Model: `claude-opus-5-5` (override with env `FIF_MODEL`), server-side refusal fallback enabled.

## Misconception guard (`ai/guards.py`, issue #37)

Every generated text (LLM answers in `ask()`, LLM profiles, briefings) passes `remove_myths()`: sentences
stating one of the three spacecraft-fire misconceptions listed in NASA's Researcher's Guide p. 35 as fact —
*a fire self-extinguishes if ventilation is shut off*, *no steady fire in quiescent microgravity*,
*reduced gravity is inherently safer / 1 g standards are very conservative* — are dropped (recorded in
`guard_removed` / `generated.validation.myths_removed`, with the source). Sentences that present them as a
misconception, or that report the opposite (BASS: "…sustains the flame even when the ambient airflow is
turned off"), are kept. A test scans all shipped generated text.
