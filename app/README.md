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
| `/` | Home: habitat picker → gap banner, map preview, top-5 evidence; 3-step demo tour (ranker → gap → Ask) | key_numbers.json, coverage.json, `ai.ranking.rank` |
| `/overview` | Filterable table of the 24 investigations | investigations.json |
| `/detail?id=PSI-26` | One investigation: our summary, the investigators' own words (verbatim), tests, documents (download), publications | investigations.json, tests.csv, AI profile, data/documents/ |
| `/explorer` | Data explorer: every test row with filters, ⚠ corrections + original PSI row, CSV download, raw PSI table, PSI file list (links only) | tests.csv, data/experimental_tables/, data/file_index/ |
| `/ranker?habitat=Mars+base` | Scenario ranker (preset from the URL) | `ai.ranking.rank`, `ai.habitats` |
| `/scenario` | Fire scenario: what would a fire do in this atmosphere? Prediction where the data allows, otherwise the reason and the evidence | `analysis.scenario.assess` (via `app/fire.py`) |
| `/response` | Suppression evidence (renamed in #65): will it go out, counted FLEX shares with CO₂ / He, model curves at a pre-set diluent fraction, NIOSH caveat, NASA's list of misconceptions, related passages | `analysis.scenario`, `ai.qa.ask` |
| `/condition_map?gap=solid` | Tests vs habitat atmospheres (`gap=solid` highlights the solid-material gap) | tests.csv, coverage.json |
| `/predict` | Will a burning droplet go out? (FLEX model) | `analysis.flex_classifier`, analysis/flex_boundary.json |
| `/ask?q=...` | Cited Q&A (asks `q` straight away) | `ai.qa.ask` |
| `/compare` | 2–4 investigations side by side + similarity map | investigations.json, tests.csv, analysis/similarity.json |
| `/insights` | Insights by fire-safety pillar | processed/ai/insights.json (when it exists) |
| `/about` | Sources, DOIs, data issues, AI disclosure | etl_log.csv, coverage.json |

Screenshots of every page: [screenshots/](screenshots/).

### Look and feel (#48, #55)

- **Layout:** dashboard-first. Home has a habitat picker that drives the gap banner, the map preview and the
  top-5 evidence list, then the 3-step tour. The Scenario ranker has the scenario in one bar on top, four
  headline numbers, then the ranking beside the reasons; selecting a row shows its score breakdown.
- **Theme:** `.streamlit/config.toml` (repo root) has the IGNIS palette from the deck in both modes:
  navy `#0B1426`/`#172540`, flame `#F26419` accent (light mode `#C2410C` for contrast), ember `#FFB000` links.
  Callouts (error / warning / info) use palette tints. Viewers get their system setting and can switch in
  ⋮ → Settings.
- **UI kit:** `app/ui.py`. Every page uses it, so the look changes in one file:
  `page_header` (h1) → `section` (h3) → bold text in a `card`; `metric_card` (bordered); `table` (one style:
  "—" for missing, 4 significant figures; `verbatim=True` for tables quoted from PSI); `provenance_label` /
  `badge` (PSI verbatim · Computed · Rule-based · AI-written · ML model); `callout` (the solid-material gap is
  always red); `source_link`; `empty_state`; `chart_theme` (chart labels for light / dark); `inject_css`
  (prose width cap, navy text on flame buttons in dark mode).
- **Icons:** Material icons (`ui.ICONS`) in navigation, page titles, buttons and callouts.
- **Data colours:** `ui.PHASE_COLOURS`, `ui.OUTCOME_COLOURS`, `ui.AGENT_COLOURS` (blue / magenta / aqua, checked
  for colour-blind separation). Orange is the UI accent only, never a data colour.
- **Sidebar sections:** `SECTIONS` in `app/nav.py` (Explore · Decide · AI & ML · Learn).
- **Tests:** `app/tests/conftest.py` holds a shim for an AppTest bug in Streamlit ≤ 1.51 (single-select
  segmented controls); newer versions do not need it.

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
