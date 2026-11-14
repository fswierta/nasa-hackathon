"""Model registry (#57): one interface to every model the app ships, for the ML-oriented UI.

    from analysis.registry import describe, predict, explain, habitat_predictions
    describe()                      # metadata + metrics + importance + reliability for every model
    predict("flex_outcome", fuel="Heptane", o2_frac=0.21, pressure_kPa=101.3)
    explain("flex_outcome", fuel="Heptane", o2_frac=0.30, pressure_kPa=101.3, x_co2=0.2)
    habitat_predictions()           # what the models say for each habitat preset

kind:  "ml" (trained, cross-validated model) | "unsupervised" (similarity map) | "formula" (the scenario
ranking - a transparent weighted score, NOT machine learning; the UI must label it that way).

`py -m analysis.registry` writes processed/analysis/models.json and habitat_predictions.json.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "processed" / "analysis"
SEED = 0
FLEX = {"psi_id": "PSI-69", "name": "FLEX", "doi": "10.60555/mbq8-0451"}
CFIG = {"psi_id": "PSI-159", "name": "ACME CFI-G", "doi": "10.60555/cyt1-kh64"}
LABELS = {"fuel": "Fuel", "o2_frac": "O₂ (mole fraction)", "pressure_kPa": "Pressure (kPa)", "x_co2": "CO₂ (mole fraction)",
          "x_he": "Helium (mole fraction)", "d0_mm": "Droplet diameter (mm)", "fuel_frac": "Fuel fraction in the burner gas",
          "hc_rate_mg_s": "Fuel flow (mg/s)", "heptane": "Fuel = heptane", "ln_p": "Pressure (log)",
          "ln_pressure": "Pressure (log)", "carbon_number": "Fuel chain length (C atoms)"}


# ------------------------------------------------------------------ per-model adapters

def _flex_parts():
    from analysis import flex_classifier as fc
    model, rows, card = fc._fitted()
    X = np.array([fc.features(r) for r in rows])
    y = np.array([r["outcome"] for r in rows])
    return fc, model, rows, card, X, y


def _gas_parts(target: str):
    from analysis import scenario as sc
    models, ranges = sc._gas_models()
    m = models[target]
    d = m["rows"]
    return sc, m["model"], d, m, ranges, d[sc.GAS_FEATURES].to_numpy(), d[target].astype(int).to_numpy()


def _range(vr: dict, key: str):
    vals = [v[key] for v in vr.values()] if all(isinstance(v, dict) for v in vr.values()) else [vr[key]]
    return round(min(v[0] for v in vals), 4), round(max(v[1] for v in vals), 4)


def _num_input(name, lo, hi, default, unit):
    return {"name": name, "label": LABELS.get(name, name), "unit": unit, "type": "number", "min": lo, "max": hi, "default": default}


def _importance(model, X, y, names) -> list[dict]:
    """Permutation importance (drop in balanced accuracy when a feature is shuffled), on the training data."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring="balanced_accuracy", n_repeats=8, random_state=SEED)
    out = [{"feature": n, "label": LABELS.get(n, n), "value": round(float(v), 4)} for n, v in zip(names, r.importances_mean)]
    return sorted(out, key=lambda d: -d["value"])


def _reliability(make_model, X, y, positive, bins=5) -> list[dict]:
    """Out-of-fold predicted probability of `positive` vs how often it really happened."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_model()
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = sorted(set(y.tolist()))
    p = proba[:, classes.index(positive)]
    obs = (y == positive).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum():
            out.append({"p_predicted": round(float(p[sel].mean()), 3), "p_observed": round(float(obs[sel].mean()), 3), "n": int(sel.sum())})
    return out


@lru_cache(maxsize=1)
def describe() -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier

    fc, fmodel, frows, fcard, fX, fy = _flex_parts()
    chosen = fcard["cv"][fcard["model"]]
    vr = fcard["valid_range"]
    flex_inputs = [{"name": "fuel", "label": "Fuel", "unit": "", "type": "choice", "choices": fc.FUELS, "default": "Heptane"},
                   _num_input("o2_frac", *_range(vr, "o2_frac"), 0.21, "mole fraction"),
                   _num_input("pressure_kPa", *_range(vr, "pressure_kPa"), 101.3, "kPa"),
                   _num_input("x_co2", *_range(vr, "x_co2"), 0.0, "mole fraction"),
                   _num_input("x_he", *_range(vr, "x_he"), 0.0, "mole fraction"),
                   _num_input("d0_mm", *_range(vr, "d0_mm"), 2.5, "mm")]
    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED)  # noqa: E731
    models = [{
        "id": "flex_outcome", "title": "Will a burning droplet go out?", "kind": "ml", "task": "classification",
        "algorithm": "Random forest (300 trees, class-balanced)", "dataset": "flex", "target": "outcome", "classes": fc.CLASSES,
        "headline": "p_goes_out", "inputs": flex_inputs,
        "metrics": {"n_train": fcard["n_train"], "cv_scheme": fcard["cv_scheme"], "cv_accuracy": chosen["cv_accuracy"],
                    "baseline_accuracy": fcard["baseline_accuracy"], "balanced_accuracy": chosen["cv_balanced_accuracy"],
                    "recall": chosen["recall"]},
        "importance": _importance(fmodel, fX, fy, fc.FEATURES),
        "reliability": _reliability(rf, fX, fy, "extinction"), "reliability_of": "P(extinction)",
        "scope": fcard["scope"], "valid_range": vr,
        "limits": ["Liquid fuel droplets only - not fabrics or plastics.", "Microgravity only (ISS).",
                   "Refuses inputs outside the tested range of each fuel.",
                   "'disruption' is a test ending, not a fire-safety outcome; only 27 'completion' examples."],
        "source": FLEX, "page": "predict",
    }]

    from ai.model import model_card as rate_card
    rc = rate_card()["k"]
    models.append({
        "id": "flex_burning_rate", "title": "How fast does a droplet burn?", "kind": "ml", "task": "regression",
        "algorithm": "Linear regression (ridge-stabilised)", "dataset": "flex", "target": "burning_rate_k_mm2_s", "unit": "mm²/s",
        "headline": "k", "inputs": flex_inputs,
        "metrics": {"n_train": rc["n_train"], "cv_scheme": "5-fold cross-validation", "r2": rc["cv_r2"], "rmse": rc["cv_rmse"],
                    "baseline_r2": 0.0, "baseline": "predicting the mean burning rate"},
        "importance": [{"feature": k, "label": LABELS.get(k, k), "value": round(abs(float(v)), 4)} for k, v in
                       sorted(rc.get("coefficients", {}).items(), key=lambda kv: -abs(kv[1]))],
        "importance_note": "absolute model coefficients (inputs are not standardised - compare with care)",
        "reliability": [], "scope": fcard["scope"], "valid_range": rc.get("valid_range"),
        "limits": ["Liquid fuel droplets only.", "Refuses inputs outside the tested range.", "±1.96 RMSE interval, not a guarantee."],
        "source": FLEX, "page": "predict",
    })

    for target, title, positive_label in (("hot_ignition", "Will a gas flame ignite?", "P(hot ignition)"),
                                          ("self_extinguished", "Will the gas flame go out by itself?", "P(self-extinguish)")):
        sc, gmodel, gd, gm, ranges, gX, gy = _gas_parts(target)
        models.append({
            "id": "cfig_ignition" if target == "hot_ignition" else "cfig_self_extinguish", "title": title, "kind": "ml",
            "task": "classification", "algorithm": "Random forest (300 trees, class-balanced)", "dataset": "cfig", "target": target,
            "classes": [0, 1], "headline": "p_hot_ignition" if target == "hot_ignition" else "p_self_extinguish_if_ignited",
            "inputs": [{"name": "fuel", "label": "Fuel", "unit": "", "type": "choice", "choices": list(sc.GAS_FUELS), "default": "propane"},
                       _num_input("o2_frac", *ranges["o2_frac"], 0.21, "mole fraction"),
                       _num_input("pressure_kPa", *ranges["pressure_kPa"], 101.3, "kPa"),
                       _num_input("fuel_frac", *ranges["fuel_frac"], 1.0, "mole fraction"),
                       _num_input("hc_rate_mg_s", *ranges["hc_rate_mg_s"], 1.0, "mg/s")],
            "metrics": {"n_train": gm["n_train"], "cv_scheme": "stratified 5-fold, repeated 3x", "cv_accuracy": gm["cv_accuracy"],
                        "baseline_accuracy": gm["baseline_accuracy"], "balanced_accuracy": gm["cv_balanced_accuracy"]},
            "importance": _importance(gmodel, gX, gy, sc.GAS_FEATURES),
            "reliability": _reliability(rf, gX, gy, 1), "reliability_of": positive_label,
            "scope": "Ethane / propane / butane flames on a spherical burner in microgravity (normal flames only).",
            "valid_range": ranges,
            "limits": ["Gaseous fuel flames only - not fabrics or plastics.", "Microgravity only (ISS).",
                       "Inverse flames excluded: their O₂ is the burner stream, not an atmosphere.",
                       "Refuses inputs outside the tested ranges."],
            "source": CFIG, "page": "scenario",
        })

    sim_path = OUT / "similarity.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text(encoding="utf-8"))
        models.append({
            "id": "similarity", "title": "Which investigations are alike?", "kind": "unsupervised", "task": "clustering",
            "algorithm": "MiniLM text embeddings + condition features → MDS map, hierarchical clustering",
            "dataset": "investigations", "inputs": [],
            "metrics": {"n_train": len(sim["points"]), "clusters": sim["cluster_quality"]["k"], "silhouette": sim["cluster_quality"]["silhouette"],
                        "related_links_found": sim["validation"]["found_in_top5_neighbours"],
                        "related_links_total": sim["validation"]["psi_related_pairs"]},
            "importance": [], "reliability": [], "scope": "All 24 PSI combustion investigations.",
            "limits": ["No labels were used; clusters are a description, not a prediction.",
                       "Validated only against PSI's own 'related investigations' links."],
            "source": {"psi_id": None, "name": "processed/investigations.json", "doi": None}, "page": "compare",
        })

    from ai.ranking import DEFAULT_WEIGHTS
    models.append({
        "id": "ranking", "title": "Which experiments matter for a habitat?", "kind": "formula", "task": "ranking",
        "algorithm": "Transparent weighted score - not machine learning", "dataset": "investigations", "inputs": [],
        "metrics": {"weights": DEFAULT_WEIGHTS}, "importance": [], "reliability": [],
        "scope": "Ranks the 24 investigations for a habitat scenario; every component is shown and adjustable.",
        "limits": ["A formula chosen by the team, not learned from data.", "No partial-gravity data: Moon / Mars gravity scores 'no data'."],
        "source": {"psi_id": None, "name": "ai/ranking.py", "doi": None}, "page": "ranker",
    })
    return models


def _spec(model_id: str) -> dict:
    return next(m for m in describe() if m["id"] == model_id)


def _defaults(spec: dict, inputs: dict) -> dict:
    return {**{i["name"]: i["default"] for i in spec["inputs"]}, **{k: v for k, v in inputs.items() if v is not None}}


# ------------------------------------------------------------------ predict / explain

def _raw_predict(model_id: str, x: dict) -> dict:
    """-> {"value": headline number | None, "prediction": {...} | None, "reason": str | None, "valid_range": ...}"""
    from analysis import scenario as sc
    if model_id == "flex_outcome":
        r = sc._droplet({**x, "fuel_class": "droplet"})
        p = r["prediction"]
        return {"value": p["p_goes_out"] if p else None, "prediction": p, "reason": r["why_no_prediction"], "valid_range": r.get("valid_range")}
    if model_id == "flex_burning_rate":
        from ai.model import predict_flex
        dil = "N2+CO2" if x.get("x_co2", 0) > 0.005 else "N2+He" if x.get("x_he", 0) > 0.005 else "N2"
        r = predict_flex(x["o2_frac"], x["pressure_kPa"], dil, x.get("d0_mm", 2.5), fuel=str(x.get("fuel", "heptane")),
                         co2_frac=x.get("x_co2", 0.0), he_frac=x.get("x_he", 0.0))
        ok = r.get("k") is not None
        return {"value": r.get("k"), "prediction": {"k": r["k"], "k_low": r["k_low"], "k_high": r["k_high"], "unit": "mm²/s"} if ok else None,
                "reason": r.get("reason"), "valid_range": r.get("valid_range")}
    if model_id in ("cfig_ignition", "cfig_self_extinguish"):
        r = sc._gas({**x, "fuel_class": "gas"})
        key = "p_hot_ignition" if model_id == "cfig_ignition" else "p_self_extinguish_if_ignited"
        p = r["prediction"]
        return {"value": p[key] if p else None, "prediction": p, "reason": r["why_no_prediction"], "valid_range": r.get("valid_range")}
    raise KeyError(f"{model_id} has no predict()")


def _nearest(model_id: str, x: dict, k: int = 5) -> list[dict]:
    """The k real tests closest to the inputs (same fuel, range-scaled distance) with what actually happened."""
    spec = _spec(model_id)
    nums = [i for i in spec["inputs"] if i["type"] == "number"]
    if spec["dataset"] == "flex":
        _, _, rows, *_ = _flex_parts()
        pool = [dict(r, actual=r["outcome"]) for r in rows if r["fuel"].lower() == str(x.get("fuel", "")).lower()]
    else:
        sc, _, d, *_ = _gas_parts("hot_ignition")
        cn = sc.GAS_FUELS.get(str(x.get("fuel", "")).lower())
        pool = [dict(r, actual="hot ignition" if r["hot_ignition"] == 1 else "no hot ignition")
                for r in d.to_dict("records") if r["carbon_number"] == cn]
    def dist(r):
        s = 0.0
        for i in nums:
            v = r.get(i["name"])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            span = (i["max"] - i["min"]) or 1.0
            s += ((float(v) - float(x[i["name"]])) / span) ** 2
        return s
    out = []
    for r in sorted(pool, key=dist)[:k]:
        out.append({"test_id": r["test_id"], "actual": r["actual"],
                    "conditions": {i["name"]: (round(float(r[i["name"]]), 4) if r.get(i["name"]) is not None else None) for i in nums}})
    return out


def predict(model_id: str, **inputs) -> dict:
    spec = _spec(model_id)
    x = _defaults(spec, inputs)
    r = _raw_predict(model_id, x)
    return {"ok": r["prediction"] is not None, "model_id": model_id, "inputs": x, "headline": spec["headline"],
            "value": r["value"], "prediction": r["prediction"], "reason": r["reason"], "valid_range": r["valid_range"],
            "nearest_tests": _nearest(model_id, x), "source": spec["source"], "kind": spec["kind"]}


def _medians(model_id: str, fuel) -> dict:
    spec = _spec(model_id)
    nums = [i["name"] for i in spec["inputs"] if i["type"] == "number"]
    if spec["dataset"] == "flex":
        _, _, rows, *_ = _flex_parts()
        pool = [r for r in rows if r["fuel"].lower() == str(fuel).lower()] or rows
        return {n: float(np.median([r[n] for r in pool])) for n in nums}
    sc, _, d, *_ = _gas_parts("hot_ignition")
    sub = d[d["carbon_number"] == sc.GAS_FUELS.get(str(fuel).lower(), -1)]
    sub = sub if len(sub) else d
    return {n: float(sub[n].median()) for n in nums}


def explain(model_id: str, **inputs) -> list[dict]:
    """What-if explanation: for each input, how much the headline prediction changes compared with setting that
    input to the training median (for this fuel). effect > 0 = this input pushes the prediction up."""
    spec = _spec(model_id)
    x = _defaults(spec, inputs)
    base = _raw_predict(model_id, x)["value"]
    if base is None:
        return []
    med = _medians(model_id, x.get("fuel"))
    out = []
    for i in spec["inputs"]:
        n = i["name"]
        if i["type"] == "number":
            alt = _raw_predict(model_id, {**x, n: med[n]})["value"]
            ref = f"median {med[n]:.3g}"
        else:
            others = [_raw_predict(model_id, {**x, n: c})["value"] for c in i["choices"] if str(c).lower() != str(x[n]).lower()]
            others = [o for o in others if o is not None]
            alt = float(np.mean(others)) if others else None
            ref = "other fuels"
        if alt is None:
            continue
        out.append({"feature": n, "label": i["label"], "value": x[n], "reference": ref, "effect": round(float(base - alt), 4)})
    return sorted(out, key=lambda d: -abs(d["effect"]))


# ------------------------------------------------------------------ habitat predictions

def habitat_predictions() -> list[dict]:
    """For every habitat preset: what the models say for that atmosphere in microgravity, the statement for the
    habitat's own gravity, and the evidence state for solids. Never a number where PSI has no data."""
    from ai.habitats import HABITATS
    from analysis.scenario import GRAVITY, assess

    out = []
    for name, h in HABITATS.items():
        atm = {"o2_frac": h["o2_frac"], "pressure_kPa": h["pressure_kPa"]}
        def one(model_id, **kw):
            r = predict(model_id, **atm, **kw)
            return {"model_id": model_id, "inputs": r["inputs"], "value": r["value"], "reason": r["reason"]}
        solid = assess({"fuel_class": "solid", "fuel": "SIBAL", **atm})
        g = h.get("gravity", "micro")
        out.append({
            "habitat": name, **atm, "gravity": g,
            "in_microgravity": {
                "droplet_goes_out": [dict(one("flex_outcome", fuel=f), fuel=f) for f in ("Heptane", "Methanol")],
                "gas_ignites": [dict(one("cfig_ignition", fuel="propane", fuel_frac=0.3), fuel="propane")],
                "gas_goes_out": [dict(one("cfig_self_extinguish", fuel="propane", fuel_frac=0.3), fuel="propane")],
            },
            "at_habitat_gravity": None if g == "micro" else {"prediction": None, "statement": GRAVITY[g]["statement"]},
            "solids": {"prediction": None, "reason": solid["why_no_prediction"], "gap": solid["gap"],
                       "tests_of_sibal": solid["tests_of_this_material"]},
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = describe()
    (OUT / "models.json").write_text(json.dumps(models, indent=1, ensure_ascii=False), encoding="utf-8")
    hp = habitat_predictions()
    (OUT / "habitat_predictions.json").write_text(json.dumps(hp, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote models.json ({len(models)} entries) and habitat_predictions.json ({len(hp)} habitats)")
    for m in models:
        met = m["metrics"]
        score = (f"acc {met['cv_accuracy']} vs baseline {met['baseline_accuracy']}" if "cv_accuracy" in met else
                 f"R² {met['r2']}" if "r2" in met else str({k: v for k, v in met.items() if k != 'weights'}))
        print(f"  {m['id']:22s} {m['kind']:12s} {score}  top: {[i['feature'] for i in m['importance'][:3]]}")
    for h in hp:
        d = h["in_microgravity"]["droplet_goes_out"][0]; g = h["in_microgravity"]["gas_ignites"][0]
        print(f"  {h['habitat']:24s} heptane goes out: {d['value']}  propane ignites: {g['value']}  gravity note: {bool(h['at_habitat_gravity'])}")


if __name__ == "__main__":
    main()
