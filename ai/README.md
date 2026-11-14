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

Habitat entries carry `assumptions` (fields not from PSI data, e.g. default 10 cm/s airflow) and
`source`. Show both in the UI.

## Data sources and honesty rules

- Investigations come from `processed/investigations.json` (main's ETL) when it exists; otherwise
  from the raw catalog + metadata + the `SEED` table in `catalog.py` (values copied from
  AGENT_BRIEF §4 only; everything else `None`). ETL values override the seed.
- Every `ask()` citation points to a chunk that was actually retrieved; citations the model
  invents are dropped. Profile findings without a valid excerpt are dropped.
- Model: `claude-opus-5-5` (override with env `FIF_MODEL`), server-side refusal fallback enabled.
