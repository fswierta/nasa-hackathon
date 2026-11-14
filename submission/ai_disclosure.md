# Use of AI tools — disclosure

> Space Apps asks teams to disclose AI use. Verify the required wording/format on
> https://www.spaceappschallenge.org/resources/ and adapt this text.

**In the product (what users see)**

- **Question answering and investigation summaries:** Anthropic Claude (`claude-opus-5-5`, via the
  Anthropic API). The model receives only passages retrieved from PSI documents and metadata and is
  instructed to answer only from them and cite each claim. Citations that do not match a retrieved
  passage are removed. Without an API key the app shows the retrieved passages instead.
- **Document search:** Chroma's default local sentence-embedding model plus keyword (BM25) search
  over 4,631 passages extracted from PSI PDFs/DOCX, each linked to file and page.
- **Not AI:** the scenario ranking is a transparent weighted formula (weights shown and adjustable);
  the FLEX burning-rate model is an ordinary linear regression (cross-validated R² ≈ 0.70, n = 257)
  that refuses inputs outside the tested ranges; the data cleaning is rule-based code with every
  judgement call logged.

**In building the project**

- The team used **Claude Code** (Anthropic) as a coding assistant to write and test the data
  pipeline, AI layer, dashboard and these documents. All outputs were run and reviewed by the team;
  data values come from the PSI files, not from the AI.

**Data**

- All data: NASA Physical Sciences Informatics (PSI), CC0 1.0. Each investigation is cited by its
  PSI DOI (10.60555/…) in the app. This project is not endorsed by NASA and does not use the NASA logo.
