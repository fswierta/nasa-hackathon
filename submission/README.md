# submission/ — owned by main (pc1)

| File | What |
|---|---|
| [summary.md](summary.md) | Project summary for the submission form (~190 words) |
| [slides.md](slides.md) | 7-slide outline with speaker notes and screenshot slots |
| [video_script.md](video_script.md) | ≤30 s demo video script, shot by shot |
| [ai_disclosure.md](ai_disclosure.md) | AI-use disclosure (product + development) |

## Before submitting — checklist (humans)

The official rules pages could not be fetched from the dev machine (TLS error), so **verify these
on https://www.spaceappschallenge.org/resources/** (Participant Guide / submission page):

- [ ] Summary length limit, required fields, team member list
- [ ] Video vs slides requirement, max length (we assumed ≤30 s video or 7 slides), public link needed?
- [ ] AI-use disclosure format
- [ ] Judging criteria (typically Impact, Creativity, Validity, Relevance, Presentation) — slides map to them
- [ ] Submission deadline / time zone
- [ ] No NASA logo anywhere (app, slides, video); no implied endorsement

## Placeholders to fill

- [x] Project name: **IGNIS — Intelligent Guidance for Navigating Ignition in Space** (summary, README, video done; slides + app in #39 / #27)
- [ ] Screenshots on slides 4–5 (pc3 has `app/screenshots/`; add the Predict extinction page and the similarity map)
- [ ] Live demo URL (pc3, #10) and team names (slide 7)
- [ ] Re-check numbers if the data is rebuilt: `py etl/build_coverage.py` prints them

All numbers in these files come from `processed/investigations.json`, `processed/tests.csv`,
`processed/coverage.json`, `processed/ai/chunks.jsonl`, `ai.model.model_card()`,
`analysis.flex_classifier.model_card()` and `processed/analysis/similarity.json` (15:45).
