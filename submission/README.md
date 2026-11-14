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

- [ ] Project name (summary, slides 1/7, video title card) — team decision
- [ ] Screenshots on slides 4–5 (pc3, once #4 / #5 / #6 are done)
- [ ] Live demo URL (pc3, #10) and team names (slide 7)
- [ ] Re-check numbers if the data is rebuilt: `py etl/build_coverage.py` prints them

All numbers in these files come from `processed/investigations.json`, `processed/tests.csv`,
`processed/coverage.json`, `processed/ai/chunks.jsonl` and `ai.model.model_card()` (14:00).
