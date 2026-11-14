"""ML-ready datasets + feature catalog for the ML Lab panel (main owns the data layer; pc2 builds the panel).

Two supervised datasets, each one row per test, built from processed/tests.csv (so every ETL fix and
flag carries over):

    flex  - FLEX droplet combustion on the ISS (PSI-69), 274 tests
    cfig  - ACME CFI-G gaseous cool-flame tests (PSI-159), 333 tests

    from analysis.datasets import DATASETS, load, feature_catalog, default_features
    df = load("flex")                         # pandas DataFrame: meta + raw + engineered + target columns
    feature_catalog("flex")                   # [{name, kind, unit, formula, why, leak}, ...]
    default_features("flex", "outcome")       # safe feature list for that target (no leakage)

Feature kinds:
    raw         - a condition set before the test (as published by PSI)
    engineered  - computed here from raw columns, with the physical reason in `why`
    pi_derived  - computed by the investigators and published in the table (e.g. adiabatic flame temp.)
    post_test   - measured DURING/AFTER the test: using it to predict the outcome is target leakage.
                  `leak` lists the targets it must not be used for.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "processed" / "tests.csv"


def _f(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return math.nan


def _raw(row, *names):
    for n in names:
        for k, v in row.items():
            if " ".join(k.split()).lower() == n.lower():
                return v
    return None


# ------------------------------------------------------------------ dataset definitions

DATASETS = {
    "flex": {
        "title": "FLEX — burning fuel droplets in microgravity",
        "source": {"psi_id": "PSI-69", "doi": "10.60555/mbq8-0451"},
        "scope": "Liquid fuel droplets (methanol, heptane) on the ISS. Not solid materials.",
        "targets": {
            "outcome": {"type": "classification", "classes": ["extinction", "disruption", "completion"],
                        "note": "How the test ended. 'disruption' is a test ending (droplet lost), not a combustion result."},
            "went_out": {"type": "classification", "classes": [0, 1],
                         "note": "1 = flame went out with fuel left (extinction), 0 = droplet burned completely; "
                                 "disrupted tests excluded (NaN)."},
            "burning_rate_k_mm2_s": {"type": "regression", "unit": "mm²/s", "note": "d²-law burning-rate constant K."},
            "extinction_diameter_mm": {"type": "regression", "unit": "mm", "note": "Droplet diameter when the visible flame went out."},
        },
        "features": [
            {"name": "fuel", "kind": "raw", "unit": "category", "formula": "Fuel", "why": "Methanol vs heptane: different volatility and sooting."},
            {"name": "o2_frac", "kind": "raw", "unit": "mole fraction", "formula": "O₂ initial ambient composition", "why": "Main driver of flammability."},
            {"name": "pressure_kPa", "kind": "raw", "unit": "kPa", "formula": "mmHg × 0.133322", "why": "Sets gas density and reaction rates."},
            {"name": "x_co2", "kind": "raw", "unit": "mole fraction", "formula": "CO₂ initial ambient composition", "why": "Suppressant: high heat capacity, radiates."},
            {"name": "x_he", "kind": "raw", "unit": "mole fraction", "formula": "He initial ambient composition", "why": "Suppressant: high thermal conductivity."},
            {"name": "x_n2", "kind": "raw", "unit": "mole fraction", "formula": "N₂ initial ambient composition", "why": "Balance gas (collinear with O₂ + CO₂ + He = 1 − N₂)."},
            {"name": "d0_mm", "kind": "raw", "unit": "mm", "formula": "Droplet initial diameter", "why": "Large droplets lose more heat by radiation → radiative extinction."},
            {"name": "ln_pressure", "kind": "engineered", "unit": "ln(kPa)", "formula": "ln(pressure_kPa)", "why": "Pressure acts multiplicatively on rates; log makes the effect linear."},
            {"name": "o2_partial_kPa", "kind": "engineered", "unit": "kPa", "formula": "o2_frac × pressure_kPa", "why": "Oxygen actually available per volume — what habitat designers specify (e.g. 30 % at 70 kPa = 21 kPa)."},
            {"name": "co2_per_o2", "kind": "engineered", "unit": "ratio", "formula": "x_co2 / o2_frac", "why": "Suppressant per unit oxidizer."},
            {"name": "he_per_o2", "kind": "engineered", "unit": "ratio", "formula": "x_he / o2_frac", "why": "Suppressant per unit oxidizer."},
            {"name": "inert_frac", "kind": "engineered", "unit": "mole fraction", "formula": "1 − o2_frac", "why": "Total dilution of the oxidizer."},
            {"name": "d0_sq_mm2", "kind": "engineered", "unit": "mm²", "formula": "d0_mm²", "why": "d²-law: burn time scales with D₀²."},
            {"name": "is_heptane", "kind": "engineered", "unit": "0/1", "formula": "fuel == Heptane", "why": "One-hot of fuel for linear models."},
            {"name": "burn_time_s", "kind": "post_test", "unit": "s", "formula": "Burn time", "why": "Measured during the test.",
             "leak": ["outcome", "went_out", "burning_rate_k_mm2_s", "extinction_diameter_mm"]},
            {"name": "extinction_diameter_mm", "kind": "post_test", "unit": "mm", "formula": "Visible flame extinction diameter",
             "why": "Exists only if the flame went out.", "leak": ["outcome", "went_out", "burning_rate_k_mm2_s"]},
            {"name": "burning_rate_k_mm2_s", "kind": "post_test", "unit": "mm²/s", "formula": "Burning rate",
             "why": "Measured during the test.", "leak": ["outcome", "went_out", "extinction_diameter_mm"]},
        ],
    },
    "cfig": {
        "title": "ACME CFI-G — gaseous cool flames on a spherical burner",
        "source": {"psi_id": "PSI-159", "doi": "10.60555/cyt1-kh64"},
        "scope": "Ethane / propane / butane flames on the ISS, normal and inverse configuration. "
                 "In inverse flames the O₂ value is the burner stream, not an atmosphere (o2_role = burner).",
        "targets": {
            "hot_ignition": {"type": "classification", "classes": [0, 1], "note": "Did a hot flame ignite ('Hot ignition' column)."},
            "self_extinguished": {"type": "classification", "classes": [0, 1],
                                  "note": "Among ignited tests: 1 = flame went out by itself (SE), 0 = ended by stopping the flow (FT)."},
            "burn_time_s": {"type": "regression", "unit": "s",
                            "note": "Censored for flow-terminated tests (the operator stopped them) — treat with care."},
        },
        "features": [
            {"name": "flame_type", "kind": "raw", "unit": "category", "formula": "Flame Type", "why": "Normal: fuel from the burner into O₂/N₂. Inverse: O₂/N₂ from the burner into fuel/N₂."},
            {"name": "fuel_species", "kind": "raw", "unit": "category", "formula": "Fuel (ethane / propane / butane)", "why": "Cool-flame chemistry strengthens with chain length."},
            {"name": "fuel_frac", "kind": "raw", "unit": "mole fraction", "formula": "Xf (normal: burner stream; inverse: ambient before test)", "why": "Fuel dilution."},
            {"name": "o2_frac", "kind": "raw", "unit": "mole fraction", "formula": "XO2 before test (normal: ambient) / XO2 (inverse: burner)", "why": "Oxidizer strength; see o2_role."},
            {"name": "pressure_kPa", "kind": "raw", "unit": "kPa", "formula": "approx p (bar) × 100", "why": "Cool flames are favoured at higher pressure."},
            {"name": "hc_rate_mg_s", "kind": "raw", "unit": "mg/s", "formula": "mHC / HC burn rate", "why": "Fuel mass flow sets flame size and residence time."},
            {"name": "tad_K", "kind": "pi_derived", "unit": "K", "formula": "Tad (K), computed by the investigators", "why": "Adiabatic flame temperature of the mixture."},
            {"name": "zst", "kind": "pi_derived", "unit": "–", "formula": "Zst, computed by the investigators", "why": "Stoichiometric mixture fraction: where the flame sits between fuel and oxidizer."},
            {"name": "ln_pressure", "kind": "engineered", "unit": "ln(kPa)", "formula": "ln(pressure_kPa)", "why": "Log-linear pressure effect."},
            {"name": "o2_partial_kPa", "kind": "engineered", "unit": "kPa", "formula": "o2_frac × pressure_kPa", "why": "Oxygen available per volume."},
            {"name": "is_inverse", "kind": "engineered", "unit": "0/1", "formula": "flame_type == inverse", "why": "One-hot of configuration."},
            {"name": "carbon_number", "kind": "engineered", "unit": "count", "formula": "ethane 2 · propane 3 · butane 4", "why": "Ordinal encoding of fuel chain length."},
            {"name": "burn_time_s", "kind": "post_test", "unit": "s", "formula": "burn time (sec)", "why": "Measured during the test.",
             "leak": ["hot_ignition", "self_extinguished"]},
            {"name": "peak_burner_temp_C", "kind": "post_test", "unit": "°C", "formula": "Peak burner T", "why": "Measured during the test.",
             "leak": ["hot_ignition", "self_extinguished", "burn_time_s"]},
        ],
    },
}
META = ["test_id", "date_iso", "flags", "source_file"]


# ------------------------------------------------------------------ builders

def _flex(tests: pd.DataFrame) -> pd.DataFrame:
    t = tests[tests["investigation_id"] == "PSI-69"].copy()
    raws = t["raw_row"].map(json.loads)
    df = pd.DataFrame({
        "test_id": t["test_id"], "date_iso": t["date_iso"], "flags": t["flags"], "source_file": t["source_file"],
        "fuel": t["fuel"],
        "o2_frac": t["o2_frac"].map(_f), "pressure_kPa": t["pressure_kPa"].map(_f),
        "x_co2": raws.map(lambda r: _f(_raw(r, "CO initial ambient composition; mole fraction"))),
        "x_he": raws.map(lambda r: _f(_raw(r, "He initial ambient composition; mole fraction"))),
        "x_n2": raws.map(lambda r: _f(_raw(r, "N initial ambient composition; mole fraction"))),
        "d0_mm": t["d0_mm"].map(_f),
        "burn_time_s": t["burn_time_s"].map(_f),
        "extinction_diameter_mm": t["extinction_diameter_mm"].map(_f),
        "burning_rate_k_mm2_s": t["burning_rate_k_mm2_s"].map(_f),
        "outcome": t["outcome"],
    }).reset_index(drop=True)
    df["went_out"] = df["outcome"].map({"extinction": 1.0, "completion": 0.0})
    df["ln_pressure"] = df["pressure_kPa"].map(lambda p: math.log(p) if p == p and p > 0 else math.nan)
    df["o2_partial_kPa"] = df["o2_frac"] * df["pressure_kPa"]
    df["co2_per_o2"] = df["x_co2"] / df["o2_frac"]
    df["he_per_o2"] = df["x_he"] / df["o2_frac"]
    df["inert_frac"] = 1 - df["o2_frac"]
    df["d0_sq_mm2"] = df["d0_mm"] ** 2
    df["is_heptane"] = (df["fuel"] == "Heptane").astype(float)
    return df


def _cfig(tests: pd.DataFrame) -> pd.DataFrame:
    t = tests[tests["investigation_id"] == "PSI-159"].copy()
    raws = t["raw_row"].map(json.loads)
    species = t["fuel"].str.extract(r"(ethane|propane|butane)", expand=False)
    hot = raws.map(lambda r: str(_raw(r, "Hot ignition") or "").strip().lower())
    term = raws.map(lambda r: str(_raw(r, "Self-ext or flow term", "Self -ext or flow term") or "").strip().upper())
    df = pd.DataFrame({
        "test_id": t["test_id"], "date_iso": t["date_iso"], "flags": t["flags"], "source_file": t["source_file"],
        "flame_type": t["geometry"].map(lambda g: "inverse" if "inverse" in g else "normal"),
        "o2_role": t["o2_role"],
        "fuel_species": species,
        "fuel_frac": raws.map(lambda r: _f(_raw(r, "Xf", "Xf before test"))),
        "o2_frac": t["o2_frac"].map(_f), "pressure_kPa": t["pressure_kPa"].map(_f),
        "hc_rate_mg_s": raws.map(lambda r: _f(_raw(r, "mHC (mg/s)", "HC burnrate (mg/s)"))),
        "tad_K": raws.map(lambda r: _f(_raw(r, "Tad (K)"))),
        "zst": raws.map(lambda r: _f(_raw(r, "Zst"))),
        "burn_time_s": t["burn_time_s"].map(_f),
        "peak_burner_temp_C": raws.map(lambda r: _f(_raw(r, "Peak burner T (C )", "Peak burner Temp (C)"))),
        "outcome_text": t["outcome"],
    }).reset_index(drop=True)
    hot, term = hot.reset_index(drop=True), term.reset_index(drop=True)
    df["hot_ignition"] = hot.map({"yes": 1.0, "no": 0.0})          # 'flash' etc. -> NaN (ambiguous)
    df["self_extinguished"] = term.map({"SE": 1.0, "FT": 0.0}).where(df["hot_ignition"] == 1.0)
    df["ln_pressure"] = df["pressure_kPa"].map(lambda p: math.log(p) if p == p and p > 0 else math.nan)
    df["o2_partial_kPa"] = df["o2_frac"] * df["pressure_kPa"]
    df["is_inverse"] = (df["flame_type"] == "inverse").astype(float)
    df["carbon_number"] = df["fuel_species"].map({"ethane": 2.0, "propane": 3.0, "butane": 4.0})
    return df


_BUILDERS = {"flex": _flex, "cfig": _cfig}


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    tests = pd.read_csv(TESTS, dtype=str, keep_default_na=False)
    return _BUILDERS[name](tests)


def load(name: str) -> pd.DataFrame:
    """One row per test: META + features (raw, engineered, pi_derived, post_test) + target columns."""
    return _load(name).copy()


def feature_catalog(name: str) -> list[dict]:
    return [dict(f, leak=f.get("leak", [])) for f in DATASETS[name]["features"]]


def default_features(name: str, target: str) -> list[str]:
    """Numeric, non-leaking features safe for `target` (categoricals are covered by their one-hot/ordinal twins)."""
    df = _load(name)
    out = []
    for f in feature_catalog(name):
        if f["kind"] == "post_test" or target in f["leak"] or f["name"] == target:
            continue
        if f["unit"] == "category" or f["name"] not in df.columns:
            continue
        out.append(f["name"])
    return out


def summary(name: str) -> dict:
    """Numbers for the data-exploration tab: rows, missing values, target balance."""
    df, spec = _load(name), DATASETS[name]
    cols = [f["name"] for f in spec["features"] if f["name"] in df.columns]
    targets = {}
    for t, meta in spec["targets"].items():
        s = df[t]
        if meta["type"] == "classification":
            targets[t] = {"n": int(s.notna().sum()), "classes": {str(k): int(v) for k, v in s.value_counts().items()}}
        else:
            targets[t] = {"n": int(s.notna().sum()), "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}
    corrected = df["flags"].map(lambda f: any(p.startswith("F") for p in f.split(" | ") if p))
    return {"rows": len(df), "rows_with_corrections": int(corrected.sum()),
            "missing": {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}, "targets": targets}


if __name__ == "__main__":
    for n in DATASETS:
        print(n, json.dumps(summary(n), indent=1))
        print("  default features per target:", {t: default_features(n, t) for t in DATASETS[n]["targets"]})
