# Project summary

> Paste into the Space Apps submission form. ~250 words. Check the form's length limit before pasting.

**IGNIS — Intelligent Guidance for Navigating Ignition in Space** — an AI-powered fire-safety dashboard for NASA microgravity combustion data · Live demo: [Streamlit URL]

NASA has run decades of combustion experiments in microgravity, but the results are spread over
24 investigations, 66,000+ files and hundreds of documents in the Physical Sciences Informatics
(PSI) repository. IGNIS turns them into answers for spacecraft and habitat fire safety.

We cleaned the 22 PSI tables (21 with test data) into one dataset of 1,947 test records with common
units (O2 fraction, kPa, cm/s), and indexed 93 PDF/Word documents (4,631 passages with file and page).
On top of that the dashboard lets users:

- **rank** investigations for a habitat (ISS, Moon base, Mars base, rover, spacesuit) with an
  explainable score;
- **map** every test by oxygen and pressure against planned habitat atmospheres;
- **ask** questions in plain language — AI search finds the answering passages in NASA's own
  documents, shown with file and page;
- **predict** with machine-learning models trained on the PSI tables whether a burning fuel droplet
  goes out and how fast it burns (FLEX), and whether a gas flame ignites or goes out by itself
  (ACME CFI-G) — and say plainly where NASA has no data: solid materials, Moon and Mars gravity;
- **review suppression evidence**: what the research says about CO₂ / helium and NASA's list of
  misconceptions — research support, not a crew procedure;
- **compare** all 24 investigations on a computed similarity map.

Key finding: **no real fabric or plastic sample was burned near the planned Moon-base atmosphere
(≥30 % O2, ≤70 kPa)** — solids were tested only up to 22.2 % O2. The closest evidence: ACME BRE's
gas burner emulating a PMMA-like solid burned steadily at 34 % O2 / 8.2 psia and 26.5 % O2 /
10.2 psia (BRE Final Report 2022). And lunar gravity may be nearly the most flammable condition
(Ferkul et al., LPSC 2026) — there is no partial-gravity data in PSI.

Every value links to its PSI DOI. Not endorsed by NASA.
