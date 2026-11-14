# Ask-the-data examples (no-LLM mode)

Generated 2026-11-14 by `python -m ai.examples` from `ai.qa.ask()` **without an API key**: answers are verbatim sentences from the PSI documents, ranked by keyword relevance (BM25 + embeddings, domain query expansion, scoping by investigation name). Bold = question terms. Each question has a reviewer verdict and an automatic citation check.

## 1. Does flame spread faster in microgravity than in 1 g?

**Best match [1]** — For the opposed flow configuration, the **flame** quickly reached steady **spread** for each flow speed, and the **spread** rate was fastest at an intermediate value of flow speed. — *PSI-26 · PSI-26_Science Documents_FlammabilityAspectsofFabric_USNationalCombustion.2013.docx*

**Supporting passages:**
- [2] The results also show that the electric body force can produce **flame** oscillations similar to those driven by buoyancy in normal gravity, and **microgravity** **flame** shapes with an electric field can mimic normal- — *PSI-22 · PSI-22_Reports_2018 - E-FIELD Flames 30-day Operation Accomplishments Report for ISS Increments 55 and 56 20181115.pdf, p. 8*
- [3] However, there is some indication from other experiments and models that at lower flow speeds, the opposed-flow **flame** will actually **spread** **faster**. — *PSI-26 · PSI-26_Science Documents_FlammabilityAspectsofFabric_USNationalCombustion.2013.docx*
- [4] In **microgravity**, we have a good understanding of opposed flow **flame** **spread**, and our knowledge of concurrent **spread** is also in good shape, as we have a good understanding of how items burn with a specified ambient flow. — *PSI-20 · PSI-20_Reports_Dissertation-PD-revised_.pdf, p. 18*

_Verbatim passages from the PSI documents, ranked by keyword relevance — not an AI-written answer (no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env))._

| # | PSI-ID | DOI | File | Page | Check |
|---|---|---|---|---|---|
| 1 | [PSI-26](https://psi.nasa.gov/physci/repo/data/investigations/PSI-26) | [10.60555/58h4-dy80](https://doi.org/10.60555/58h4-dy80) | `documents/PSI-26/PSI-26_Science Documents_FlammabilityAspectsofFabric_USNationalCombustion.2013.docx` | — | ok |
| 2 | [PSI-22](https://psi.nasa.gov/physci/repo/data/investigations/PSI-22) | [10.60555/se5a-8c73](https://doi.org/10.60555/se5a-8c73) | `documents/PSI-22/PSI-22_Reports_2018 - E-FIELD Flames 30-day Operation Accomplishments Report for ISS Increments 55 and 56 20181115.pdf` | 8 | ok (p. 8 of 15) |
| 3 | [PSI-26](https://psi.nasa.gov/physci/repo/data/investigations/PSI-26) | [10.60555/58h4-dy80](https://doi.org/10.60555/58h4-dy80) | `documents/PSI-26/PSI-26_Science Documents_FlammabilityAspectsofFabric_USNationalCombustion.2013.docx` | — | ok |
| 4 | [PSI-20](https://psi.nasa.gov/physci/repo/data/investigations/PSI-20) | [10.60555/vb93-3y72](https://doi.org/10.60555/vb93-3y72) | `documents/PSI-20/PSI-20_Reports_Dissertation-PD-revised_.pdf` | 18 | ok (p. 18 of 171) |

**Reviewer verdict:** Partly answered. BASS passages show spread rate depends on flow speed (fastest at an intermediate speed; opposed-flow flames may spread faster at low flow) — i.e. 'it depends on ventilation', not a simple yes/no. The E-FIELD passage is only loosely related.

## 2. Which materials were tested in Saffire?

**Best match [1]** — The fuels and sample **materials** **tested** in **SAFFIRE**-II (PSI-99) were: PMMA, PMMA + Nomex, SIBAL, silicone (solid fuels). — *PSI-99 · investigations.json (PSI experimental table)*

**Supporting passages:**
- [2] The fuels and sample **materials** **tested** in **SAFFIRE**-I (PSI-98) were: SIBAL (solid fuels). — *PSI-98 · investigations.json (PSI experimental table)*
- [3] The fuels and sample **materials** **tested** in **SAFFIRE**-III (PSI-100) were: SIBAL (solid fuels). — *PSI-100 · investigations.json (PSI experimental table)*
- [4] The sample material for **Saffire**-I is a composite fabric called Sibal consisting of cotton on a fiberglass substrate (75% cotton, 25% fiberglass by weight). — *PSI-98 · PSI-98.json#approach*

_Verbatim passages from the PSI documents, ranked by keyword relevance — not an AI-written answer (no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env))._

| # | PSI-ID | DOI | File | Page | Check |
|---|---|---|---|---|---|
| 1 | [PSI-99](https://psi.nasa.gov/physci/repo/data/investigations/PSI-99) | [10.60555/2chp-m469](https://doi.org/10.60555/2chp-m469) | `processed/investigations.json (PSI experimental table)` | — | ok (ETL table) |
| 2 | [PSI-98](https://psi.nasa.gov/physci/repo/data/investigations/PSI-98) | [10.60555/0t15-1z43](https://doi.org/10.60555/0t15-1z43) | `processed/investigations.json (PSI experimental table)` | — | ok (ETL table) |
| 3 | [PSI-100](https://psi.nasa.gov/physci/repo/data/investigations/PSI-100) | [10.60555/040q-4h36](https://doi.org/10.60555/040q-4h36) | `processed/investigations.json (PSI experimental table)` | — | ok (ETL table) |
| 4 | [PSI-98](https://psi.nasa.gov/physci/repo/data/investigations/PSI-98) | [10.60555/0t15-1z43](https://doi.org/10.60555/0t15-1z43) | `metadata/PSI-98.json#approach` | — | ok |

**Reviewer verdict:** Answered. Saffire-I/III: SIBAL cotton-fiberglass fabric; Saffire-II: SIBAL, silicone, structural PMMA, flat PMMA, PMMA with Nomex.

## 3. What O2 level stopped droplet burning in FLEX?

**Best match [1]** — In **FLEX** (PSI-69), the flame was extinguished in 186 tests run at oxygen (**O2**) concentrations of 12 to 30 percent. — *PSI-69 · investigations.json (PSI experimental table)*

**Supporting passages:**
- [2] Inexplicably after hot flame extinction, however, the **droplet** continued to vaporize at the same rate for an extended period and then suddenly **stopped** vaporizing at a finite **droplet** size. — *REF · NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf, p. 33*
- [3] To date, these experiments have identified both the radiative and diffusive extinction limits for heptane and methanol **droplets** and identified an LOI that is both significantly lower than in normal gravity and lower than previously predicted and anticipated. — *REF · NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf, p. 36*
- [4] This oxygen concentration is called the Limiting Oxygen Index (LOI ). — *PSI-69 · PSI-69_Science Documents_FLEX_Test_Plan.pdf, p. 2*

_Verbatim passages from the PSI documents, ranked by keyword relevance — not an AI-written answer (no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env))._

| # | PSI-ID | DOI | File | Page | Check |
|---|---|---|---|---|---|
| 1 | [PSI-69](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69) | [10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451) | `processed/investigations.json (PSI experimental table)` | — | ok (ETL table) |
| 2 | REF | — (Researcher's Guide) | `reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf` | 33 | ok (p. 33 of 56) |
| 3 | REF | — (Researcher's Guide) | `reference/NASA_Researchers_Guide_ISS_Combustion_Science_2015.pdf` | 36 | ok (p. 36 of 56) |
| 4 | [PSI-69](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69) | [10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451) | `documents/PSI-69/PSI-69_Science Documents_FLEX_Test_Plan.pdf` | 2 | ok (p. 2 of 15) |

**Reviewer verdict:** Qualitatively answered: passage [2] says FLEX found a Limiting Oxygen Index significantly lower than in 1 g. The lead sentence is about cool-flame vaporisation stopping — related but not the O2 level. No single number appears in the documents; see the data check.

**Data check** (computed from `processed/tests.csv`, FLEX PSI-69, all pressures and diluents):

| Fuel | Tests | Lowest O2 that still burned to completion | O2 range where flames extinguished |
|---|---|---|---|
| Heptane | 117 | 0.14 | 0.13–0.24 |
| Methanol | 157 | 0.18 | 0.12–0.30 |

Extinction depends on droplet size, pressure and diluent as well as O2 — see `ai.model.predict_flex`.

## 4. How does smoke differ in microgravity (SAME)?

**Best match [1]** — The **smoke** properties were measured using particulate aerosol diagnostics that measure **different** moments of the size distribut ion. — *PSI-102 · PSI-102_Reports_SAME_Final Science Report.pdf, p. 51*

**Supporting passages:**
- [2] The experimenta l results show that, fo r the materials tested, a substantial portion of the **smoke** particles are below 500 nm in diameter. — *PSI-102 · PSI-102_Reports_SAME_Final Science Report.pdf, p. 59*
- [3] The International Space Station (ISS) **smoke** detectors use near-IR forward scattering, rendering them most sensitive to particles larger than a micrometer, outside of the range of sensitivity of the shuttle detector. — *PSI-102 · PSI-102_Reports_SAME_Final Science Report.pdf, p. 36*
- [4] The Comparative Soot Diagnostics (CSD) experiment that flew in the Glovebox on STS-75 demonstrated that the **smokes** with solid particulates created in **microgravity** formed longer chains, and aggregated into larger particles than those created in normal Earth gravity. — *PSI-102 · PSI-102_Science Documents_SAME_Baseline Concept Description Doc.pdf, p. 7*

_Verbatim passages from the PSI documents, ranked by keyword relevance — not an AI-written answer (no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env))._

| # | PSI-ID | DOI | File | Page | Check |
|---|---|---|---|---|---|
| 1 | [PSI-102](https://psi.nasa.gov/physci/repo/data/investigations/PSI-102) | [10.60555/gd4k-8h62](https://doi.org/10.60555/gd4k-8h62) | `documents/PSI-102/PSI-102_Reports_SAME_Final Science Report.pdf` | 51 | ok (p. 51 of 68) |
| 2 | [PSI-102](https://psi.nasa.gov/physci/repo/data/investigations/PSI-102) | [10.60555/gd4k-8h62](https://doi.org/10.60555/gd4k-8h62) | `documents/PSI-102/PSI-102_Reports_SAME_Final Science Report.pdf` | 59 | ok (p. 59 of 68) |
| 3 | [PSI-102](https://psi.nasa.gov/physci/repo/data/investigations/PSI-102) | [10.60555/gd4k-8h62](https://doi.org/10.60555/gd4k-8h62) | `documents/PSI-102/PSI-102_Reports_SAME_Final Science Report.pdf` | 36 | ok (p. 36 of 68) |
| 4 | [PSI-102](https://psi.nasa.gov/physci/repo/data/investigations/PSI-102) | [10.60555/gd4k-8h62](https://doi.org/10.60555/gd4k-8h62) | `documents/PSI-102/PSI-102_Science Documents_SAME_Baseline Concept Description Doc.pdf` | 7 | ok (p. 7 of 15) |

**Reviewer verdict:** Answered with results: most SAME smoke particles were below 500 nm, and the ISS forward-scattering detectors are most sensitive above ~1 µm — a detector mismatch. Earlier CSD flight: solid smoke particles formed longer chains/larger aggregates in microgravity.

## 5. What is known about fires at Moon-base atmospheres (30–34 % O2, ~56–70 kPa)?

**Best match [1]** — This is a profound result indicating a common plastic could steadily burn under these NASA proposed atmospheric conditions. — *PSI-20 · PSI-20_Reports_Final_20220201.pdf, p. 74*

**Supporting passages:**
- [2] At an atmospheric **O2** concentration (21%) with 5% O3 (both v/v), sub-millimeter sized n-C7H16 droplet attains direct initiation of cool flame upon ignition energy initialization. — *PSI-117 · PSI-117_Reports_Dissertation_Combustion Behavior Of Sub-Millimeter Sized Oxygenated And N-Alka.pdf, p. 210*
- [3] Generally, it has been found that oxygen levels of 26 % and above can allow steady flames. — *PSI-20 · PSI-20_Reports_Dissertation-PD-revised_.pdf, p. 31*
- [4] The results even show burning can exist at 21 % oxygen for very low mass flux values. — *PSI-20 · PSI-20_Reports_Dissertation-PD-revised_.pdf, p. 80*

_Verbatim passages from the PSI documents, ranked by keyword relevance — not an AI-written answer (no LLM configured (set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env))._

| # | PSI-ID | DOI | File | Page | Check |
|---|---|---|---|---|---|
| 1 | [PSI-20](https://psi.nasa.gov/physci/repo/data/investigations/PSI-20) | [10.60555/vb93-3y72](https://doi.org/10.60555/vb93-3y72) | `documents/PSI-20/PSI-20_Reports_Final_20220201.pdf` | 74 | ok (p. 74 of 181) |
| 2 | [PSI-117](https://psi.nasa.gov/physci/repo/data/investigations/PSI-117) | [10.60555/gz0q-b577](https://doi.org/10.60555/gz0q-b577) | `documents/PSI-117/PSI-117_Reports_Dissertation_Combustion Behavior Of Sub-Millimeter Sized Oxygenated And N-Alka.pdf` | 210 | ok (p. 210 of 227) |
| 3 | [PSI-20](https://psi.nasa.gov/physci/repo/data/investigations/PSI-20) | [10.60555/vb93-3y72](https://doi.org/10.60555/vb93-3y72) | `documents/PSI-20/PSI-20_Reports_Dissertation-PD-revised_.pdf` | 31 | ok (p. 31 of 171) |
| 4 | [PSI-20](https://psi.nasa.gov/physci/repo/data/investigations/PSI-20) | [10.60555/vb93-3y72](https://doi.org/10.60555/vb93-3y72) | `documents/PSI-20/PSI-20_Reports_Dissertation-PD-revised_.pdf` | 80 | ok (p. 80 of 171) |

**Reviewer verdict:** Answered via the closest proxy: ACME BRE (gas burner emulating solids) reports a common plastic could steadily burn under NASA's proposed exploration atmospheres. No solid-material test exists at these conditions (see processed/coverage.json gaps).
