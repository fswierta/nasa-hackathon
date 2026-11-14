# app/ — owned by pc3

Streamlit dashboard. Reads `processed/` (main's ETL, pc2's AI files) and imports `ai/`.

## Run locally (from the repo root)

```bash
pip install -r requirements.txt
streamlit run app/main.py          # http://localhost:8501
python -m pytest -q app/tests      # renders every page against the real data
```

Without `ANTHROPIC_API_KEY` in `.env`, everything works except AI-written answers: *Ask the data*
then shows the most relevant document passages instead.

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
| `/ask` | Cited Q&A | `ai.qa.ask` |
| `/compare` | 2–4 investigations side by side | investigations.json, tests.csv |
| `/insights` | Insights by fire-safety pillar | processed/ai/insights.json (when it exists) |
| `/about` | Sources, DOIs, data issues, AI disclosure | etl_log.csv, coverage.json |

Screenshots of every page: [screenshots/](screenshots/).

## Deploy to Streamlit Community Cloud

Someone with access to the GitHub repo has to do this once:

1. Sign in at https://share.streamlit.io with GitHub and click **Create app → Deploy a public app from GitHub**.
2. Repository `fswierta/nasa-hackathon`, branch `main`, main file path `app/main.py`.
   Under **Advanced settings**, choose Python 3.11 or newer.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Never commit the key. `.streamlit/secrets.toml` and `.env` are gitignored, and `main.py` copies
   the secret into the environment for `ai/`.
4. Deploy. The first start installs the requirements (a few minutes). Then put the app URL in the root `README.md`.

The Chroma vector index (`processed/ai/index/`) is not in git, so the deployed app uses keyword
(BM25) search over `processed/ai/chunks.jsonl`. It works without the index.
