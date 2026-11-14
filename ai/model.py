"""Simple, honest predictive model on FLEX droplet tests (PSI-69) — task P2-5, brief S2.

    from ai.model import predict_flex, model_card
    predict_flex(o2_frac=0.18, pressure_kPa=101.3, diluent="N2+CO2", d0_mm=3.0, fuel="Heptane", co2_frac=0.1)

Linear least squares (numpy only) on rows of processed/tests.csv:
    target ~ fuel + o2_frac + ln(pressure) + x_CO2 + x_He + d0_mm
Error = 5-fold cross-validated RMSE; the band is prediction ± 1.96·RMSE.
Refuses to extrapolate: every input must lie inside the range tested *for that fuel*.
This is a screening aid built from 274 ISS tests, not a validated flammability model.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from .config import PROCESSED_DIR

FLEX_ID = "PSI-69"
FLEX_DOI = "10.60555/mbq8-0451"
TARGETS = {"k": "burning_rate_k_mm2_s", "extinction_d": "extinction_diameter_mm"}
FEATURES = ["o2_frac", "ln_pressure", "x_co2", "x_he", "d0_mm"]
RAW_CO2 = "CO initial ambient composition; mole fraction"   # 'CO2' with the subscript lost in PSI's header
RAW_HE = "He initial ambient composition; mole fraction"


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    import pandas as pd

    t = pd.read_csv(PROCESSED_DIR / "tests.csv")
    t = t[t["investigation_id"] == FLEX_ID]
    rows = []
    for _, r in t.iterrows():
        raw = json.loads(r["raw_row"]) if isinstance(r["raw_row"], str) else {}
        row = {
            "test_id": r["test_id"], "fuel": str(r["fuel"]).strip().lower(),
            "o2_frac": _num(r["o2_frac"]), "pressure_kPa": _num(r["pressure_kPa"]),
            "x_co2": _num(raw.get(RAW_CO2)) or 0.0, "x_he": _num(raw.get(RAW_HE)) or 0.0,
            "d0_mm": _num(r["d0_mm"]),
            "k": _num(r["burning_rate_k_mm2_s"]), "extinction_d": _num(r["extinction_diameter_mm"]),
        }
        row["ln_pressure"] = math.log(row["pressure_kPa"]) if row["pressure_kPa"] else None
        rows.append(row)
    return rows


def _design(rows, fuels):
    X = []
    for r in rows:
        X.append([1.0 if r["fuel"] == f else 0.0 for f in fuels] + [r[c] for c in FEATURES])
    return np.array(X, dtype=float)


def _fit(X, y, ridge=1e-6):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


@lru_cache(maxsize=None)
def _train(target: str) -> dict:
    col = target  # row key; TARGETS maps it to the tests.csv column
    rows = [r for r in _rows() if r[col] is not None and all(r[c] is not None for c in FEATURES)]
    # extinction_d of 0 means no measurable extinction diameter (burned out) — not a data point
    if target == "extinction_d":
        rows = [r for r in rows if r[col] > 0]
    fuels = sorted({r["fuel"] for r in rows})
    X, y = _design(rows, fuels), np.array([r[col] for r in rows])

    # 5-fold CV (deterministic shuffle)
    idx = np.random.default_rng(0).permutation(len(y))
    folds = np.array_split(idx, 5)
    errs = []
    for f in folds:
        train = np.setdiff1d(idx, f)
        w = _fit(X[train], y[train])
        errs.extend((X[f] @ w - y[f]).tolist())
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    baseline = float(np.std(y))
    w = _fit(X, y)

    valid = {}
    for fuel in fuels:
        fr = [r for r in rows if r["fuel"] == fuel]
        valid[fuel] = {c: [min(r[c] for r in fr), max(r[c] for r in fr)]
                       for c in ("o2_frac", "pressure_kPa", "x_co2", "x_he", "d0_mm")}
    return {"w": w, "fuels": fuels, "rmse": rmse, "baseline_std": baseline,
            "r2_cv": 1 - rmse ** 2 / baseline ** 2, "n_train": len(y), "valid_range": valid}


def _check(m, fuel, x):
    if fuel not in m["fuels"]:
        return f"fuel '{fuel}' was not tested in FLEX (tested: {', '.join(m['fuels'])})"
    for c, (lo, hi) in m["valid_range"][fuel].items():
        v = x[c]
        if v < lo - 1e-9 or v > hi + 1e-9:
            return f"{c}={v:g} is outside the FLEX range for {fuel} ({lo:g}–{hi:g})"
    return None


def predict_flex(o2_frac: float, pressure_kPa: float, diluent: str = "N2", d0_mm: float = 3.0,
                 fuel: str = "heptane", co2_frac: float | None = None, he_frac: float | None = None,
                 target: str = "k") -> dict:
    """Predict FLEX burning-rate constant K (mm²/s) or extinction diameter (mm).

    `diluent` ("N2", "N2+CO2", "N2+He") selects which extra diluent is present; give its mole
    fraction via co2_frac / he_frac (required when that diluent is selected).
    Returns {"k", "k_low", "k_high", "n_train", "valid_range", ...} or {"k": None, "reason": ...}.
    """
    m = _train(target)
    fuel = fuel.strip().lower()
    d = (diluent or "N2").upper()
    x_co2 = (co2_frac or 0.0) if "CO2" in d else 0.0
    x_he = (he_frac or 0.0) if "HE" in d else 0.0
    out = {"target": target, "unit": "mm²/s" if target == "k" else "mm", "n_train": m["n_train"],
           "cv_rmse": round(m["rmse"], 4), "cv_r2": round(m["r2_cv"], 3),
           "valid_range": m["valid_range"].get(fuel), "source": {"psi_id": FLEX_ID, "doi": FLEX_DOI}}
    if ("CO2" in d and co2_frac is None) or ("HE" in d and he_frac is None):
        return {**out, "k": None, "k_low": None, "k_high": None,
                "reason": "give the diluent mole fraction (co2_frac / he_frac)"}
    x = {"o2_frac": o2_frac, "pressure_kPa": pressure_kPa, "x_co2": x_co2, "x_he": x_he, "d0_mm": d0_mm}
    reason = _check(m, fuel, x)
    if reason:
        return {**out, "k": None, "k_low": None, "k_high": None, "reason": reason + " — not extrapolating."}
    row = {**x, "fuel": fuel, "ln_pressure": math.log(pressure_kPa)}
    pred = float(_design([row], m["fuels"])[0] @ m["w"])
    half = 1.96 * m["rmse"]
    return {**out, "k": round(pred, 4), "k_low": round(max(0.0, pred - half), 4), "k_high": round(pred + half, 4),
            "reason": None}


def model_card() -> dict:
    """Coefficients and CV quality for display on the dashboard."""
    card = {}
    for target in TARGETS:
        m = _train(target)
        names = [f"fuel={f}" for f in m["fuels"]] + FEATURES
        card[target] = {"n_train": m["n_train"], "cv_rmse": round(m["rmse"], 4),
                        "baseline_std": round(m["baseline_std"], 4), "cv_r2": round(m["r2_cv"], 3),
                        "coefficients": {n: round(float(c), 4) for n, c in zip(names, m["w"])},
                        "valid_range": m["valid_range"]}
    return card


if __name__ == "__main__":
    print(json.dumps(model_card(), indent=2))
    print(predict_flex(0.21, 101.3, "N2", 3.0, "heptane"))
    print(predict_flex(0.34, 56.5, "N2", 3.0, "heptane"))
