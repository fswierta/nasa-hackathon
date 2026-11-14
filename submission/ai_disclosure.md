# Use of AI tools — disclosure

> Space Apps asks teams to disclose AI use. Verify the required wording/format on
> https://www.spaceappschallenge.org/resources/ and adapt this text.

**In the product (what users see)**

- **Document search / question answering:** Chroma's default local sentence-embedding model (a
  small neural network running on our machine) plus keyword (BM25) search over 4,631 passages
  extracted from PSI PDFs/DOCX. Answers are the retrieved passages shown verbatim with file and page.
- **Investigation profiles:** built by rules from PSI metadata and retrieved passages (quoted, with
  sources) — no generated text.
- **Insights page:** the 19 insight texts were written by an AI coding agent (Claude, via Claude Code)
  from retrieved PSI passages; every quoted passage (18) was checked word-for-word against the
  extracted documents, and the data-gap insights are computed from the PSI tables. The page labels them
  as AI-written.
- **Optional, switched off in this submission:** an answer/summary layer using Anthropic Claude
  (`claude-opus-5-5`) is implemented (`ai/qa.py`, `ai/profiles.py --llm`); it would only see retrieved
  PSI passages and its uncited claims are removed. It needs a paid API key, which we did not use.
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
