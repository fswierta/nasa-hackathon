# app/ — owned by pc3

Streamlit dashboard. Reads `processed/` (main's ETL, pc2's AI files) and imports `ai/`.

## Run locally (from the repo root)

```bash
pip install -r requirements.txt
streamlit run app/main.py          # http://localhost:8501
python -m pytest -q app/tests      # renders every page against the real data
```

No key is needed: without one, *Ask the data* shows the most relevant PSI passages (local search).
With `GEMINI_API_KEY` (free tier, https://aistudio.google.com) or `ANTHROPIC_API_KEY` in `.env`, it also
writes a cited answer from those passages; `FIF_LLM=gemini|anthropic|none` picks the provider
(see `ai/README.md`). Every answer is labelled with how it was produced.

## Pages

Listed in [nav.py](nav.py); files in [views/](views/) (not `pages/`: a folder named `pages/` makes
Streamlit bypass `st.navigation`). Data loaders with caching: [data.py](data.py).

| URL | Page | Reads |
|---|---|---|
| `/` | Home | counts, top data gap |
| `/overview` | Filterable table of the 24 investigations | investigations.json |
| `/detail?id=PSI-26` | One investigation | investigations.json, tests.csv, AI profile |
| `/ranker` | Scenario ranker | `ai.ranking.rank`, `ai.habitats` |
| `/condition_map` | Tests vs habitat atmospheres | tests.csv, coverage.json |
| `/predict` | Will a burning droplet go out? (FLEX model) | `analysis.flex_classifier`, analysis/flex_boundary.json |
| `/ask` | Cited Q&A | `ai.qa.ask` |
| `/compare` | 2–4 investigations side by side + similarity map | investigations.json, tests.csv, analysis/similarity.json |
| `/insights` | Insights by fire-safety pillar | processed/ai/insights.json (when it exists) |
| `/about` | Sources, DOIs, data issues, AI disclosure | etl_log.csv, coverage.json |

Screenshots of every page: [screenshots/](screenshots/).

## Deploy to Streamlit Community Cloud

Someone with access to the GitHub repo has to do this once:

1. Sign in at https://share.streamlit.io with GitHub and click **Create app → Deploy a public app from GitHub**.
2. Repository `fswierta/nasa-hackathon`, branch `main`, main file path `app/main.py`.
   Under **Advanced settings**, choose Python 3.11 or newer.
3. Under **Advanced settings → Secrets**, paste (all optional — the app works without keys):
   ```toml
   GEMINI_API_KEY = "..."        # free tier from https://aistudio.google.com
   # FIF_LLM = "gemini"          # gemini | anthropic | none
   # ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Never commit a key. `.streamlit/secrets.toml` and `.env` are gitignored, and `main.py` copies
   `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `FIF_LLM` and `FIF_MODEL` into the environment for `ai/`.
4. Deploy. The first start installs the requirements (a few minutes). Then put the app URL in the root `README.md`.

Search: the document embeddings ship in `processed/ai/embeddings.npy` (pc2, #21); the query encoder
comes from `chromadb`'s local ONNX model. If that package is missing the app falls back to keyword
search and still works. The first start also trains the FLEX model for `/predict` (~15 s, once).
