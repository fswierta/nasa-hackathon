# analysis/ — ML on the PSI data (owned by main)

Run from the repo root. CPU only; no GPU or API needed.

## FLEX outcome classifier — `analysis/flex_classifier.py` (#23)

```python
from analysis.flex_classifier import predict_outcome, model_card
predict_outcome("methanol", o2_frac=0.18, pressure_kPa=101.3, x_co2=0.1, x_he=0.0, d0_mm=2.5)
# {"probs": {"extinction": .., "disruption": .., "completion": ..}, "predicted": .., "reason": None | why refused,
#  "valid_range": {...}, "n_train": 261, "cv_accuracy": .., "baseline_accuracy": .., "source": {"psi_id": "PSI-69", "doi": ..}}
```
`py -m analysis.flex_classifier` retrains (~15 s) and writes `processed/analysis/flex_boundary.json`
(P(extinction) grids: O₂ × CO₂ and O₂ × He, at 70.9 and 101.3 kPa, per fuel, + the real test points).

**Model card**

| | |
|---|---|
| Data | FLEX (PSI-69, DOI 10.60555/mbq8-0451), 261 of 274 tests (13 lack D0/pressure) |
| Target | how the test ended: extinction 186 · disruption 51–61 · completion 27 (FLEX "Test end") |
| Features | fuel (heptane/methanol), O₂ mole fraction, ln(pressure), CO₂ and He mole fractions, initial droplet diameter |
| Model | random forest, 300 trees, min 3 samples/leaf, class-balanced (chosen over logistic regression: higher balanced accuracy and beats the baseline) |
| Validation | stratified 5-fold CV repeated 5× |
| Accuracy | **0.765 ± 0.035** vs **0.686** for "always extinction" |
| Balanced accuracy | **0.606** (chance = 0.33) |
| Recall | extinction 0.87 · disruption 0.64 · completion 0.30 (only 27 examples) |

**Limits — say these in the app:** liquid fuel droplets only (not fabrics/plastics); microgravity
(ISS); refuses inputs outside the per-fuel tested ranges (e.g. 56.5 kPa exploration atmosphere —
FLEX starts at ~70 kPa); "disruption" is a test ending, not a fire-safety outcome; grid cells
without a real test nearby are `null` and must not be drawn.

## Investigation similarity map — `analysis/similarity.py` (#24)

`py -m analysis.similarity` → `processed/analysis/similarity.json` (~5 s; first run downloads the
80 MB MiniLM model that the document search also uses).

```json
{"method": {...},
 "points":   [{"id", "acronym", "x", "y", "cluster", "fuel_phase", "hardware_family", "relevance_tier", "type", "doi"}],
 "clusters": [{"id", "label", "members", "dominant_fuel_phase", "dominant_hardware", "keywords"}],
 "cluster_quality": {"k", "silhouette"},
 "neighbors": {"PSI-26": [{"id", "acronym", "score", "text_score", "condition_score", "shared": ["..."]}]},
 "validation": {"psi_related_pairs", "found_in_top5_neighbours", "share", "missed_pairs"},
 "similarity_matrix": {"ids", "combined"}}
```

How it is computed (nothing hand-written):

| Step | Method |
|---|---|
| Text similarity | cosine of all-MiniLM-L6-v2 embeddings of title + objective + approach + keywords (`ai.vectors.embed`) |
| Condition similarity | cosine of z-scored ETL features: fuel phase, hardware family, flight/ground, tier, O₂ / ln(pressure) / airflow midpoints + spans, missing flags |
| Combined | 0.6 · text + 0.4 · conditions (fixed a priori) |
| Map (x, y) | metric MDS on 1 − similarity → distance on the map = dissimilarity |
| Clusters | average-linkage hierarchical clustering; k ∈ 3..7 by silhouette |
| Labels | dominant fuel phase + hardware of the members, keywords over-represented vs the rest |
| Neighbours | top 5 by combined similarity; `shared` lists concrete overlaps (PSI related link, phase, hardware, fuels, O₂/pressure/flow overlap, keywords) |

**Validation:** of the "related investigations" links PSI curators put in the metadata, the share
recovered in our top-5 neighbours is written to `validation.share` (misses listed in `missed_pairs`).
Use it on the About page; it is the honest quality number for this map.

## ML-ready datasets + feature catalog — `analysis/datasets.py` (for the ML Lab panel)

```python
from analysis.datasets import DATASETS, load, feature_catalog, default_features, summary
df = load("flex")      # 274 FLEX droplet tests   | load("cfig") -> 333 CFI-G gaseous flame tests
feature_catalog("flex")            # [{name, kind, unit, formula, why, leak}]
default_features("cfig", "self_extinguished")   # numeric, non-leaking features for that target
```

| Dataset | Targets | Reference (random forest, class-balanced, 5-fold ×3 CV, default features) |
|---|---|---|
| `flex` (PSI-69) | `outcome` (3 classes) · `went_out` · `burning_rate_k_mm2_s` · `extinction_diameter_mm` | `went_out`: accuracy 0.89 vs 0.87 baseline, balanced 0.69 |
| `cfig` (PSI-159) | `hot_ignition` · `self_extinguished` · `burn_time_s` (censored) | `self_extinguished`: 0.85 vs 0.67, balanced 0.83 · `hot_ignition`: 0.85 vs 0.81, balanced 0.79 |

Feature kinds: `raw` (set before the test) · `engineered` (computed here, each with a physical reason,
e.g. `o2_partial_kPa = o2_frac × pressure_kPa`) · `pi_derived` (published by the investigators: `tad_K`,
`zst`) · `post_test` (measured during/after the test — **target leakage**, never a default feature).

## Fire-scenario engine + suppression evidence — `analysis/scenario.py`

```python
from analysis.scenario import assess, response_brief, suppression_evidence, suppressant_scan, gravity_evidence, gas_model_card
assess({"fuel_class": "droplet"|"gas"|"solid", "fuel": ..., "o2_frac": ..., "pressure_kPa": ...,
        "x_co2": 0, "x_he": 0, "d0_mm": 2.5, "fuel_frac": 1.0, "hc_rate_mg_s": 1.0, "gravity": "micro"|"earth"|"lunar"|"mars"})
# -> {prediction | None, why_no_prediction, valid_range, gravity{label,data,statement}, limits[], sources[], disclaimer, ...}
response_brief(scenario)   # assess + suppression table + CO2/He scans (droplets) + NASA's list of misconceptions
```

| Fuel class | What you get | Source |
|---|---|---|
| `droplet` (heptane, methanol) | ML: P(each ending), `p_goes_out`, burning rate K with interval | FLEX, PSI-69 |
| `gas` (ethane, propane, butane) | ML: P(hot ignition) 0.81 vs 0.74 baseline, P(self-extinguish) 0.87 vs 0.77 — normal flames only, 18–40 % O₂, 51–304 kPa | ACME CFI-G, PSI-159 |
| `solid` (SIBAL, PMMA, silicone, Nomex…) | **no prediction** (no outcomes in PSI tables): tests of that material + O₂ range, Saffire-II 0 g vs 1 g rows, the gap | BASS / BASS-II / Saffire |
| gravity `lunar` / `mars` | **never a number** — the statement why (no partial-g data; Goldilocks zone, PSI-62) | — |

`suppression_evidence()`: share of FLEX droplet flames that went out, by O₂ band × agent (N₂ only / CO₂ / He),
with n. `suppressant_scan(fuel, o2, p, "CO2"|"He", mode="discharge"|"replace_n2")`: model P(goes out) vs agent
fraction (discharge dilutes O₂ too); null outside FLEX's ranges. `processed/analysis/fire_response.json` holds
the static parts (suppression table, Saffire-II gravity rows, gas model card, misconceptions quoted verbatim
from the Researcher's Guide p. 35, disclaimer). **Every page using this must show `disclaimer`.**
