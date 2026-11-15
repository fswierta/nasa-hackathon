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
# -> [{"id", "acronym", "doi", "score", "breakdown": {atmosphere,flow,material,pillar,tier,richness,gravity}, "why", "gaps": [...], "near"}]
#    breakdown values are 0-1, or None = "no data on record" (show it as such, don't show 0)
#    atmosphere (#64) = ONE joint O2 + pressure score from the test rows (coverage.json's "near": O2 ±0.06 and
#    pressure ±25 %, ambient rows): >= 0.5 when tests are near (near.n > 0), < 0.5 otherwise; near = {n, d_min, nearest}
#    weights: "o2" + "pressure" together weight it (the existing slider keys), or pass weights={"atmosphere": w}
#    gravity (#37/#54): 1.0 only with test data AT scenario["gravity"], else None ("no data"): micro -> every
#    flight investigation; earth -> analysis.scenario.gravity_evidence() (Saffire-II 1 g comparison);
#    lunar / mars -> None for all + analysis.scenario.GRAVITY[g]["statement"] in gaps (PSI-22: proxy note)

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

**Shipped models first (#59).** `_shipped_table()` maps the registry ids (`analysis.registry.specs()`, the
Models hub) to Lab configs: `flex_outcome` = the 2-class P(goes out) model (Lab target `went_out`, disruptions
left out), `flex_outcome:three_class` = the 3-class ending model behind the class bars (target `outcome`,
registry `metrics.three_class`), `cfig_ignition`, `cfig_self_extinguish` (`rows={"flame_type": "normal"}`).
Features are read from `analysis.flex_classifier.FEATURES` / `analysis.scenario.GAS_FEATURES`. The Lab's CV
reproduces the registry exactly — accuracy vs baseline 0.885 / 0.869, 0.765 / 0.686, 0.808 / 0.737,
0.869 / 0.773 (tested). A shipped feature set is trained in the shipped column order (forests depend on it),
others in catalog order. `start_config(ds, target)` = shipped config if one exists, else `default_config`;
the page opens on `START_TARGET` (FLEX `went_out`, the hub's headline; CFI-G `self_extinguished`).

**Workbench (#59):** `leaderboard(cfg, progress)` (every algorithm, same features / rows / folds, sorted by
balanced accuracy or R², baseline row, CV time, the shipped model and the selection rule's pick marked);
`ablation(cfg)` (raw only / raw + engineered / engineered only, same rows via `require`);
`partial_dependence(cfg)` (top-4 numeric features with the training rug + 2-D O₂ × CO₂ and O₂ × pressure);
`diagnostics(cfg)` (out-of-fold reliability bins as in the registry, Brier vs base-rate baseline, calibration
error + verdict, 10 worst-predicted tests). `peek_*` variants never train. `encode_config` / `decode_config`
= the page's `?config=` links (decode validates the untrusted token: known keys only, bounded numbers);
`share_query(model_id)` for `st.page_link(..., query_params=...)`; `run_record` = the downloadable JSON.

**Precomputed:** `python -m ai.mllab --precompute` → `processed/ai/mllab_runs.json`: for every start + default
config the run, learning curve, diagnostics and partial dependence; for every start config also the
leaderboard and ablation runs. Keys = config + a fingerprint of the dataset rows + the scikit-learn version,
so an ETL change makes them unreachable instead of stale; runs are reused incrementally. The page opens
instantly on the shipped model; changed settings train live (time shown). Warnings are recomputed on every call.

| config (5-fold × 3 CV, random forest, 50 trees, balanced; pinned scikit-learn 1.8.0) | score | baseline |
|---|---|---|
| cfig `self_extinguished` | accuracy 0.854 ± 0.039 | 0.665 |
| cfig `hot_ignition` | accuracy 0.853 ± 0.038 | 0.814 |
| flex `outcome` | accuracy 0.762 ± 0.049 | 0.686 |
| flex `went_out` | accuracy 0.896 ± 0.031 | 0.869 |
| flex `burning_rate_k_mm2_s` / `extinction_diameter_mm` | R² 0.664 / 0.745 | ≤ 0 (mean) |
| shipped model (flex `outcome`, 300 trees, 5 × 5) | 0.757 ± 0.036 (flex_classifier README: 0.765) | 0.686 |

Results depend on the scikit-learn version (1.9.1 gave flex `outcome` 0.70), so the cache key includes it
and versions are pinned (#56). Timing on this laptop: a live default run takes ~4–7 s with the pinned versions (8–15 s before); precomputed ones load instantly. Forest threads (`n_jobs`) and fold threads were measured
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
products — the app and `submission/ai_disclosure.md` must say so (pc3 issue).

**Demo-readiness (#52).** *Time budget:* a live answer gets 19 s in total — the primary model, then one
fallback (`gemini-3.5-flash-lite`), each with its own timeout and no SDK retries; 429 / timeout / no network →
the passages view with a neutral note, never an exception (tested with the SDK mocked). Batch jobs (profiles)
keep the long timeouts. *Number check:* a generated sentence whose numbers are not in the excerpts it cites
is dropped (an uncited sentence is checked against the excerpts the model listed as used and then cited with
them); if nothing verifiable is left, the passages view is shown. *Pre-warmed cache:* `python -m ai.prewarm
--fixed` asks every question the app sends by itself — 5 demo, the 15 round-1 benchmark questions, the Home
tour question, the Response advisor's question for each of its 18 fuels / materials (the question does not
depend on the habitat preset) — so they are answered on a fresh clone with no key; every cached answer
carries `mode`, `model` and `date` (39 entries, all generated by `gemini-3.5-flash-lite` on 2026-11-15).

*Answer-level evaluation* (`python -m ai.answer_eval` → `processed/ai/answer_eval.json`; `--markdown` prints
this table): the 15 round-1 benchmark questions through `ask()`.

| answers | expected source cited | citations resolve | answer term present | claims guard passes |
|---|---|---|---|---|
| 15/15 generated (gemini-3.5-flash-lite) | 93% | 93% | 87% | 93% |

Misses: *ACME E-FIELD goal* — correct answer ("ion-driven convection", "electrical manipulation of
sooting") without the benchmark's literal term "electric field"; *NASA's misconceptions* — correct, but the
claims guard does not yet accept the plural "misconceptions" (#70; this answer is left out of the committed
cache until then); *cotton-fiberglass fabric thickness* — retrieval misses the Saffire passage, the answer
says the documents do not cover it (retrieval tuning: #53).
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

## Deploy-ready runtime (issue #51)

- **Query encoder without chromadb:** `ai/encoder.py` runs the same all-MiniLM-L6-v2 ONNX model with
  `onnxruntime` + `tokenizers` (same tokenizer, mean pooling, L2 norm). Equivalence: cosine **1.000** vs
  chromadb's encoder on 20 queries (`ai/tests/test_encoder.py`); `py -m ai.benchmark` hit / answer rates
  unchanged, query time 0.37 → 0.01 s (vector) and 0.50 → 0.06 s (hybrid) — no padding to 256 tokens.
- **Requirements:** `ai/requirements.txt` is runtime-only (anthropic, google-genai, python-dotenv, numpy,
  onnxruntime, tokenizers). `ai/requirements-ingest.txt` adds pypdf, python-docx and chromadb for rebuilding
  the knowledge base (`py -m ai.ingest`, optional `--chroma`).
- **Offline:** the model (~80 MB) downloads once (SHA-256 checked) to the cache folder chromadb used; if
  that fails, `ai.retrieval.status()` says "Semantic search is off — … Keyword search still works." and
  the failure is remembered (no download retry per query).
- **Measured on this laptop:** AI-layer cold path ≈ 3 s (BM25 1.1 s, vectors 0.8 s, encoder 1.0 s, first
  `ask()` 0.2 s); app peak RSS after Ask → Scenario ranker → Response advisor → ML Lab **472 MB**.

## Persisted models (`ai/model_store.py`, issue #64 item 8)

`@persisted(name)` (under `@lru_cache`) loads `processed/models/<name>.joblib` when its fingerprint matches
(SHA-256 of `processed/tests.csv`, `analysis/datasets.py`, the builder's module file, sklearn / numpy / joblib
versions), else trains as before, so nothing is served stale. `py -m ai.model_store --build` writes
`flex_fitted`, `flex_burnout`, `gas_models` (1.2 MB); `--check` lists current / stale / missing. Cold start:
~40–45 s of training → 2.6 s of loading, identical predictions. Wiring the decorator into
`analysis/flex_classifier.py` / `analysis/scenario.py` is main's (#73).
