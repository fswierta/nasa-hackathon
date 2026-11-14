# Flame in Freefall — AI fire-safety dashboard

NASA Space Apps Challenge 2026 project. An interactive, AI-powered dashboard that summarizes,
ranks and interprets NASA microgravity combustion experiments from the
[Physical Sciences Informatics (PSI)](https://psi.nasa.gov) repository to surface fire-safety
insights for spacecraft, space stations, lunar/Martian bases and pressurized rovers.

- Team guide, roles and data contracts: [CLAUDE.md](CLAUDE.md)
- Task board: [TASKS.md](TASKS.md)
- Full challenge brief: [project-data-1-pc1/AGENT_BRIEF.txt](project-data-1-pc1/AGENT_BRIEF.txt)

## Run

```bash
# Windows: some NASA document names are very long -> enable long paths or git refuses to check out
git clone -c core.longpaths=true https://github.com/fswierta/nasa-hackathon
cd nasa-hackathon
python -m pip install -r requirements.txt      # Windows without python on PATH: py -m pip ...
python -m streamlit run app/main.py            # opens http://localhost:8501
```

No API key is needed: search, ranking, the prediction models and the similarity map run locally.
Optional: copy `.env.example` to `.env` and add a Google Gemini key (free tier) for generated
chat answers. The first search downloads an 80 MB embedding model once.

## Data

All PSI data is CC0-1.0. Each investigation is cited by its DOI (10.60555/…) in the app.
This project is not endorsed by NASA.
