# ai/ — owned by pc2

RAG index, LLM profiles, scenario ranking and Q&A. Plan: [PLAN.md](PLAN.md). Contracts: `CLAUDE.md`.
Run everything from the repo root.

## Setup

```bash
python -m venv .venv && .venv/Scripts/pip install -r ai/requirements.txt   # Windows: .venv\Scripts\...
# put your key in .env (gitignored):  ANTHROPIC_API_KEY=sk-ant-...
python -m ai.ingest          # extract PDFs/DOCX -> processed/ai/chunks.jsonl, embed -> processed/ai/index/
python -m ai.profiles        # AI profiles -> processed/ai/profiles/PSI-XX.json (cached; --force to redo)
python -m pytest ai/tests -q # no key or index needed
```

`chunks.jsonl` and the profiles are committed. The Chroma index (`processed/ai/index/`) is not;
without it `ask()` falls back to keyword (BM25) search over `chunks.jsonl`.

## API for pc3 (dashboard)

```python
from ai.habitats import HABITATS        # ISS, Lunar base, Exploration atmosphere, Mars base, Pressurized rover, Spacesuit
from ai.ranking import rank, DEFAULT_WEIGHTS
from ai.qa import ask
from ai.profiles import load_profile

rank(HABITATS["Lunar base"])                       # 24 results, best first
rank({**HABITATS["ISS"], "pillar": "detection"}, weights={"tier": 0.3})
# -> [{"id", "acronym", "doi", "score", "breakdown": {o2,pressure,flow,material,pillar,tier,richness}, "why", "gaps": [...]}]
#    breakdown values are 0-1, or None = "no data on record" (show it as such, don't show 0)

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

`ask()` answers from the LLM are cached in `processed/ai/qa_cache.json` (committed), so demo
questions work instantly and without a key once they've been asked once.

Habitat entries carry `assumptions` (fields not from PSI data, e.g. default 10 cm/s airflow) and
`source`. Show both in the UI.

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
