# Use of AI tools — disclosure

> Space Apps asks teams to disclose AI use. Verify the required wording/format on
> https://www.spaceappschallenge.org/resources/ and adapt this text.

**In the product (what users see)**

- **Document search / question answering:** Chroma's default local sentence-embedding model (a
  small neural network running on our machine) plus keyword (BM25) search over 4,631 passages
  extracted from PSI PDFs/DOCX. Answers are the retrieved passages shown verbatim with file and page.
- **Investigation profiles:** built by rules from PSI metadata and retrieved passages (quoted, with
  sources) — no generated text.
- **Insights page:** the insight titles and texts are templates filled with numbers computed from the PSI
  tables (`processed/coverage.json`, `processed/tests.csv`, the FLEX models), each backed by verbatim cited
  PSI passages — no AI-written text (`processed/ai/insights.json`, `generated.by = "rules (no LLM)"`).
- **Generative layer — Google Gemini free tier** (`gemini-3.8-flash`, fallback `gemini-3.5-flash-lite`),
  used on the Ask page when a `GEMINI_API_KEY` is configured: the model sees only retrieved PSI passages,
  every statement must cite one (uncited claims are removed) and the page names the provider. **User
  questions and the retrieved passages are sent to Google, and free-tier content may be used by Google to
  improve its products** (the PSI data itself is CC0). Without a key the app runs fully offline on the
  layers below. (An Anthropic Claude backend is also implemented but not used.)
- **Classical machine learning, no generated text:** the scenario ranking is a transparent weighted
  formula (weights shown and adjustable); the FLEX outcome classifier is a class-balanced random
  forest (261 tests, cross-validated accuracy 0.77 vs 0.69 for always predicting "extinction") and
  the FLEX burning-rate model an ordinary linear regression (R² ≈ 0.70, n = 257) — both refuse inputs
  outside the tested ranges; the investigation similarity map combines MiniLM text embeddings with
  tested-condition features (MDS projection, hierarchical clustering, labels computed from the data;
  it recovers 14 of 18 related-study links written by PSI curators); the data cleaning is rule-based
  code with every judgement call logged.

**In building the project**

- The team used **Claude Code** (Anthropic) as a coding assistant to write and test the data
  pipeline, AI layer, dashboard and these documents. All outputs were run and reviewed by the team;
  data values come from the PSI files, not from the AI.

**Data**

- All data: NASA Physical Sciences Informatics (PSI), CC0 1.0. Each investigation is cited by its
  PSI DOI (10.60555/…) in the app. This project is not endorsed by NASA and does not use the NASA logo.
