# Project summary

> Paste into the Space Apps submission form. ~215 words. Project name is a placeholder —
> pick one as a team. Check the form's length limit before pasting.

**[PROJECT NAME — suggestion: FlameScope]** — an AI-powered fire-safety dashboard for NASA microgravity combustion data

NASA has run decades of combustion experiments in microgravity, but the results are spread over
24 investigations, 66,000+ files and hundreds of documents in the Physical Sciences Informatics
(PSI) repository. FlameScope turns them into answers for spacecraft and habitat fire safety.

We cleaned all 21 experiment tables into one dataset of 1,947 test records with common units
(O2 fraction, kPa, cm/s), and indexed 93 PDF/Word documents (4,631 passages with file and page).
On top of that the dashboard lets users:

- **rank** investigations for a habitat (ISS, Moon base, Mars base, rover, spacesuit) with an
  explainable score;
- **map** every test by oxygen and pressure against planned habitat atmospheres;
- **ask** questions in plain language — AI search finds the answering passages in NASA's own
  documents, shown with file and page;
- **predict** how a droplet flame ends (extinction or burn-out) and its burning rate from FLEX
  data — models that refuse to extrapolate beyond tested ranges;
- **compare** all 24 investigations on a computed similarity map (clusters and nearest neighbours,
  checked against PSI's own related-study links).

Key finding: **no solid material — the main fire load in a habitat — was tested near the planned
Moon-base atmosphere (≥30 % O2, ≤70 kPa)**; solids were tested only up to 22.2 % O2.

Every value links to its PSI DOI. Not endorsed by NASA.
