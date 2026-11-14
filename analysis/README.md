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
