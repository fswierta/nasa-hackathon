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
pip install -r requirements.txt
cp .env.example .env        # add your API key
streamlit run app/main.py
```

## Data

All PSI data is CC0-1.0. Each investigation is cited by its DOI (10.60555/…) in the app.
This project is not endorsed by NASA.
