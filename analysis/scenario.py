"""Fire-scenario engine + suppression evidence (for the "Fire scenario" / "Response advisor" pages).

    from analysis.scenario import assess, suppression_evidence, suppressant_scan, gravity_evidence
    assess({"fuel_class": "droplet", "fuel": "heptane", "o2_frac": 0.30, "pressure_kPa": 101.3,
            "x_co2": 0.0, "x_he": 0.0, "d0_mm": 2.5, "gravity": "micro"})

What it does, per fuel class - and what it refuses to do:
  droplet  liquid fuel droplets  -> ML prediction from FLEX (PSI-69): how the burn ends + burning rate
  gas      gaseous fuel flames   -> ML prediction from ACME CFI-G normal flames (PSI-159): ignites? goes out?
  solid    fabrics / plastics    -> NO prediction (PSI tables record no outcomes for solids). Returns the
                                    evidence that exists: the Saffire-II 0 g vs 1 g results per material,
                                    the O2 levels solids were tested at, and the data gap.
  gravity  micro = the data; earth = only Saffire-II comparisons; lunar / mars = no data in PSI ->
           never a number, always the reason.

Every block carries its PSI source. This is research decision support, NOT an operational procedure.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
P = ROOT / "processed"
SEED = 0

DISCLAIMER = ("Research decision support built from NASA PSI experiment data. It is not a crew procedure: "
              "real spacecraft fire response follows the vehicle's own flight rules and trained procedures.")
SRC = {
    "flex": {"psi_id": "PSI-69", "name": "FLEX", "doi": "10.60555/mbq8-0451"},
    "cfig": {"psi_id": "PSI-159", "name": "ACME CFI-G", "doi": "10.60555/cyt1-kh64"},
    "saffire2": {"psi_id": "PSI-99", "name": "Saffire-II", "doi": "10.60555/2chp-m469"},
    "bass2": {"psi_id": "PSI-25", "name": "BASS-II", "doi": "10.60555/4qc4-de67"},
    "goldilocks": {"psi_id": "PSI-62", "name": "BASS flame-spread modelling (Olson & Ferkul 'Goldilocks' zone)", "doi": "10.60555/e9xp-ex83"},
    "guide": {"psi_id": None, "name": "NASA Researcher's Guide to ISS Combustion Science (2015), p. 35", "doi": None},
}
GRAVITY = {
    "micro": {"label": "Microgravity (ISS, transit)", "data": True,
              "statement": "All PSI flight tests ran in microgravity - predictions below apply here."},
    "earth": {"label": "Earth gravity (1 g)", "data": "partial",
              "statement": "PSI holds 1 g results only as the ground comparison in Saffire-II (9 samples). "
                           "The ML models are trained on microgravity tests and are not valid at 1 g."},
    "lunar": {"label": "Moon (1/6 g)", "data": False,
              "statement": "No partial-gravity combustion data exists in PSI, so no prediction is made. 0 g and 1 g results "
                           "may not bound this case: microgravity flame spread peaks at low flow speeds (Olson & Ferkul's "
                           "'Goldilocks' zone, PSI-62), the kind of weak buoyant flow lunar gravity produces."},
    "mars": {"label": "Mars (3/8 g)", "data": False,
             "statement": "No partial-gravity combustion data exists in PSI, so no prediction is made. 0 g and 1 g results "
                          "may not bound this case (see the 'Goldilocks' flammability zone, PSI-62)."},
}


def _tests() -> list[dict]:
    with (P / "tests.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ gas flames: CFI-G normal-flame models

GAS_FEATURES = ["carbon_number", "fuel_frac", "o2_frac", "ln_pressure", "hc_rate_mg_s"]
GAS_FUELS = {"ethane": 2.0, "propane": 3.0, "butane": 4.0}


@lru_cache(maxsize=1)
def _gas_models():
    """Random forests on CFI-G *normal* flames (fuel from the burner into an O2/N2 atmosphere), using only
    inputs a user can set. Inverse flames are excluded: their O2 is the burner stream, not an atmosphere."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

    from analysis.datasets import load

    df = load("cfig")
    df = df[df["flame_type"] == "normal"]
    out = {}
    for target in ("hot_ignition", "self_extinguished"):
        d = df.dropna(subset=[target] + GAS_FEATURES)
        X, y = d[GAS_FEATURES].to_numpy(), d[target].astype(int).to_numpy()
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED)
        cv = cross_validate(model, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED),
                            scoring=["accuracy", "balanced_accuracy"])
        model.fit(X, y)
        out[target] = {"model": model, "n_train": int(len(d)),
                       "cv_accuracy": round(float(cv["test_accuracy"].mean()), 3),
                       "cv_balanced_accuracy": round(float(cv["test_balanced_accuracy"].mean()), 3),
                       "baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 3),
                       "rows": d}
    ranges = {k: [round(float(df[k].min()), 4), round(float(df[k].max()), 4)] for k in ("o2_frac", "pressure_kPa", "fuel_frac", "hc_rate_mg_s")}
    return out, ranges


def gas_model_card() -> dict:
    models, ranges = _gas_models()
    return {"data": "ACME CFI-G normal flames (PSI-159)", "features": GAS_FEATURES, "valid_range": ranges,
            "models": {t: {k: v for k, v in m.items() if k not in ("model", "rows")} for t, m in models.items()},
            "source": SRC["cfig"], "scope": "Ethane / propane / butane flames on a spherical burner in microgravity."}


def _gas(s: dict) -> dict:
    models, ranges = _gas_models()
    fuel = (s.get("fuel") or "").lower()
    if fuel not in GAS_FUELS:
        return {"prediction": None, "why_no_prediction": f"fuel '{s.get('fuel')}' not tested in CFI-G (ethane, propane, butane)"}
    x = {"carbon_number": GAS_FUELS[fuel], "fuel_frac": s.get("fuel_frac", 1.0), "o2_frac": s["o2_frac"],
         "pressure_kPa": s["pressure_kPa"], "hc_rate_mg_s": s.get("hc_rate_mg_s", 1.0)}
    for k in ("o2_frac", "pressure_kPa", "fuel_frac", "hc_rate_mg_s"):
        lo, hi = ranges[k]
        if not lo - 1e-9 <= x[k] <= hi + 1e-9:
            return {"prediction": None, "valid_range": ranges,
                    "why_no_prediction": f"{k}={x[k]} is outside the CFI-G tested range ({lo}–{hi}) — not extrapolating."}
    row = np.array([[x["carbon_number"], x["fuel_frac"], x["o2_frac"], math.log(x["pressure_kPa"]), x["hc_rate_mg_s"]]])
    p_ign = float(models["hot_ignition"]["model"].predict_proba(row)[0][1])
    p_se = float(models["self_extinguished"]["model"].predict_proba(row)[0][1])
    return {"prediction": {"p_hot_ignition": round(p_ign, 3), "p_self_extinguish_if_ignited": round(p_se, 3)},
            "model": gas_model_card()["models"], "valid_range": ranges, "why_no_prediction": None}


# ------------------------------------------------------------------ droplets: FLEX models

def _droplet(s: dict) -> dict:
    from analysis.flex_classifier import predict_outcome

    r = predict_outcome(s.get("fuel", ""), s["o2_frac"], s["pressure_kPa"], s.get("x_co2", 0.0), s.get("x_he", 0.0),
                        s.get("d0_mm", 2.5))
    out = {"prediction": None, "why_no_prediction": r["reason"], "valid_range": r.get("valid_range"),
           "model": {"n_train": r["n_train"], "cv_accuracy": r["cv_accuracy"], "baseline_accuracy": r["baseline_accuracy"]}}
    if r["probs"]:
        pe, pc = r["probs"]["extinction"], r["probs"]["completion"]
        out["prediction"] = {"probs": r["probs"], "predicted": r["predicted"],
                             # among burns that were not disrupted: chance the flame goes out with fuel left
                             "p_goes_out": round(pe / (pe + pc), 3) if pe + pc > 0 else None}
        try:  # burning rate from pc2's regression (same honesty rules); optional
            from ai.model import predict_flex
            dil = "CO2" if s.get("x_co2", 0) > 0.005 else "He" if s.get("x_he", 0) > 0.005 else "N2"
            k = predict_flex(s["o2_frac"], s["pressure_kPa"], dil, s.get("d0_mm", 2.5))
            if k and k.get("k") is not None:
                out["prediction"]["burning_rate_k_mm2_s"] = {"value": k["k"], "low": k["k_low"], "high": k["k_high"]}
        except Exception:  # noqa: BLE001 - the classifier result stands on its own
            pass
    return out


# ------------------------------------------------------------------ solids: evidence, never a prediction

@lru_cache(maxsize=1)
def gravity_evidence() -> list[dict]:
    """Saffire-II (PSI-99): the same 9 samples burned in microgravity and in the 1 g ground test."""
    out = []
    for t in _tests():
        if t["investigation_id"] != "PSI-99":
            continue
        raw = json.loads(t["raw_row"])
        keys = list(raw)
        out.append({"sample": t["test_id"], "material": t["fuel"], "thickness_mm": t["thickness_mm"] or None,
                    "flow_direction": t["flow_direction"], "o2_frac": float(t["o2_frac"]) if t["o2_frac"] else None,
                    "flow_cm_s": float(t["flow_cm_s"]) if t["flow_cm_s"] else None,
                    "microgravity": {"burn_length": raw[keys[11]].strip(), "spread_rate": raw[keys[12]].strip()},
                    "earth_1g": {"burn_length": raw[keys[13]].strip(), "spread_rate": raw[keys[14]].strip()},
                    "source": SRC["saffire2"]})
    return out


def _solid(s: dict) -> dict:
    mat = (s.get("fuel") or "").lower()
    tests = [t for t in _tests() if t["row_kind"] == "test" and t["fuel"] and mat and mat in t["fuel"].lower()
             and t["investigation_id"] in ("PSI-25", "PSI-26", "PSI-98", "PSI-99", "PSI-100", "PSI-101", "PSI-102")]
    o2 = [float(t["o2_frac"]) for t in tests if t["o2_frac"]]
    by_inv = {}
    for t in tests:
        by_inv[t["investigation_id"]] = by_inv.get(t["investigation_id"], 0) + 1
    grav = [g for g in gravity_evidence() if mat and mat in (g["material"] or "").lower()]
    gap = None
    if o2 and s["o2_frac"] > max(o2) + 0.02:
        gap = (f"{s.get('fuel')} was tested only up to {max(o2) * 100:.1f} % O2; your scenario is {s['o2_frac'] * 100:.0f} % — "
               "there is no test of this material in such an oxygen-rich atmosphere.")
    return {"prediction": None,
            "why_no_prediction": "PSI experiment tables record no ignition / spread / extinction outcomes for solid materials "
                                 "(except Saffire-II below), so a model cannot be trained. Shown instead: the evidence that exists.",
            "tests_of_this_material": {"n": len(tests), "by_investigation": by_inv,
                                       "o2_range": [round(min(o2), 3), round(max(o2), 3)] if o2 else None},
            "gravity_comparison": grav, "gap": gap}


# ------------------------------------------------------------------ public API

def assess(s: dict) -> dict:
    """One fire scenario -> prediction (where the data allows), evidence, gravity statement, limits, sources."""
    s = dict(s)
    fuel_class, gravity = s.get("fuel_class"), s.get("gravity", "micro")
    if fuel_class not in ("droplet", "gas", "solid"):
        raise ValueError("fuel_class must be droplet | gas | solid")
    block = {"droplet": _droplet, "gas": _gas, "solid": _solid}[fuel_class](s)
    g = dict(GRAVITY[gravity], level=gravity)
    limits = []
    if gravity != "micro" and block.get("prediction"):
        block["prediction_valid_for"] = "microgravity only"
        limits.append(f"The prediction is for microgravity. {g['statement']}")
        if g["data"] is False:
            block["prediction"], block["why_no_prediction"] = None, g["statement"]
    if fuel_class != "solid":
        limits.append("Liquid-droplet and gas-flame results do not transfer to fabrics and plastics.")
    return {"inputs": s, "fuel_class": fuel_class, **block, "gravity": g, "limits": limits,
            "sources": [SRC[{"droplet": "flex", "gas": "cfig", "solid": "saffire2"}[fuel_class]]] +
                       ([SRC["goldilocks"]] if gravity in ("lunar", "mars") else []),
            "disclaimer": DISCLAIMER}


@lru_cache(maxsize=1)
def suppression_evidence() -> dict:
    """Computed from FLEX: how often a burning droplet went out, by oxygen level and by the suppressant gas
    in the atmosphere. 'went out' = extinction / (extinction + completion); disrupted tests excluded."""
    from analysis.datasets import load

    f = load("flex").dropna(subset=["went_out", "o2_frac"])
    f["agent"] = np.where(f["x_co2"] > 0.005, "CO2", np.where(f["x_he"] > 0.005, "He", "N2 only"))
    bands = [(0.0, 0.16, "below 16 %"), (0.16, 0.19, "16–19 %"), (0.19, 0.22, "19–22 % (ISS air)"), (0.22, 0.36, "above 22 % (oxygen-enriched)")]
    table = []
    for lo, hi, label in bands:
        row = {"o2_band": label}
        for agent in ("N2 only", "CO2", "He"):
            d = f[(f["o2_frac"] > lo) & (f["o2_frac"] <= hi) & (f["agent"] == agent)]
            row[agent] = {"n": int(len(d)), "share_went_out": round(float(d["went_out"].mean()), 2) if len(d) else None}
        table.append(row)
    rich = table[-1]
    return {"table": table, "source": SRC["flex"],
            "headline": (f"In oxygen-enriched atmospheres (>22 % O2) a burning droplet went out on its own in "
                         f"{rich['N2 only']['share_went_out'] * 100:.0f} % of tests with nitrogen only (n={rich['N2 only']['n']}), "
                         f"but in {rich['CO2']['share_went_out'] * 100:.0f} % with CO2 in the atmosphere (n={rich['CO2']['n']}) and "
                         f"{rich['He']['share_went_out'] * 100:.0f} % with helium (n={rich['He']['n']})."),
            "caveats": ["Liquid fuel droplets in microgravity, not fabrics or plastics.",
                        "Small groups (see n) — read as evidence, not as a guarantee.",
                        "FLEX set the suppressant fraction before ignition; it did not test discharging an agent onto a fire."]}


def suppressant_scan(fuel: str, o2_frac: float, pressure_kPa: float, agent: str = "CO2", d0_mm: float = 2.5,
                     mode: str = "discharge", steps: int = 15) -> dict:
    """Model-based 'what if we add suppressant gas': P(flame goes out) vs agent fraction for a FLEX droplet.
    mode='discharge': agent is released into the cabin, so every other gas is diluted (O2 falls to o2*(1-x)).
    mode='replace_n2': agent replaces nitrogen at constant O2 fraction (how FLEX ran its tests)."""
    key = {"CO2": "x_co2", "He": "x_he"}[agent]
    points = []
    for x in np.linspace(0, 0.6, steps):
        o2 = o2_frac * (1 - x) if mode == "discharge" else o2_frac
        r = _droplet({"fuel": fuel, "o2_frac": round(float(o2), 4), "pressure_kPa": pressure_kPa, key: round(float(x), 4), "d0_mm": d0_mm})
        points.append({"agent_frac": round(float(x), 3), "o2_frac": round(float(o2), 4),
                       "p_goes_out": r["prediction"]["p_goes_out"] if r["prediction"] else None,
                       "why_no_prediction": r["why_no_prediction"]})
    ok = [p for p in points if p["p_goes_out"] is not None]
    reach = next((p for p in ok if p["p_goes_out"] >= 0.9), None)
    return {"fuel": fuel, "agent": agent, "mode": mode, "start": {"o2_frac": o2_frac, "pressure_kPa": pressure_kPa},
            "points": points, "supported_points": len(ok),
            "first_fraction_reaching_90pct": reach["agent_frac"] if reach else None,
            "note": "Model estimate inside FLEX's tested ranges only (null = outside). Pressure is held constant.",
            "source": SRC["flex"], "disclaimer": DISCLAIMER}


GUIDE_VERDICT = ("NASA: research in ground-based laboratories and microgravity experiments 'demonstrates the fallacy of "
                 "these assumptions' (Researcher's Guide, p. 35).")
# Verbatim from the NASA Researcher's Guide to ISS Combustion Science (2015), p. 35 - nothing paraphrased.
RESPONSE_FACTS = [
    {"topic": "ventilation", "status": "misconception",
     "claim": "A fire will self-extinguish if ventilation is shut off.", "verdict": GUIDE_VERDICT, "source": SRC["guide"]},
    {"topic": "still air", "status": "misconception",
     "claim": "A steady fire cannot exist in a quiescent microgravity environment.", "verdict": GUIDE_VERDICT, "source": SRC["guide"]},
    {"topic": "ground standards", "status": "misconception",
     "claim": "Because the reduced-gravity environment is inherently safer, terrestrial fire standards and tests "
              "represent a very conservative approach to fire safety.", "verdict": GUIDE_VERDICT, "source": SRC["guide"]},
    {"topic": "detection", "status": "finding",
     "claim": "Since many fire detectors operate by sensing fire-generated particulates, they would need to use different "
              "signatures to detect fires in microgravity.",
     "verdict": "NASA: microgravity research 'highlighted the vastly different particulate size associated with flames in "
                "microgravity' (Researcher's Guide, p. 35); see SAME / SAME-R (PSI-102, PSI-101).", "source": SRC["guide"]},
]


def response_brief(s: dict) -> dict:
    """Evidence pack for 'there is a fire in this atmosphere - what does the research say about fighting it?'."""
    a = assess(s)
    out = {"scenario": a, "suppression": suppression_evidence(), "misconceptions": RESPONSE_FACTS, "disclaimer": DISCLAIMER}
    if s.get("fuel_class") == "droplet" and a.get("prediction"):
        out["suppressant_scans"] = [suppressant_scan(s["fuel"], s["o2_frac"], s["pressure_kPa"], agent, s.get("d0_mm", 2.5))
                                    for agent in ("CO2", "He")]
    return out


def main() -> None:
    (P / "analysis").mkdir(parents=True, exist_ok=True)
    data = {"suppression": suppression_evidence(), "gravity_evidence": gravity_evidence(), "gas_model_card": gas_model_card(),
            "misconceptions": RESPONSE_FACTS, "gravity_levels": GRAVITY, "disclaimer": DISCLAIMER}
    (P / "analysis" / "fire_response.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote processed/analysis/fire_response.json")
    print(" ", data["suppression"]["headline"])
    for t, m in data["gas_model_card"]["models"].items():
        print(f"  gas {t}: n={m['n_train']} acc {m['cv_accuracy']} vs baseline {m['baseline_accuracy']} (balanced {m['cv_balanced_accuracy']})")


if __name__ == "__main__":
    main()
