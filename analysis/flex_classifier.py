"""FLEX droplet-flame outcome classifier (issue #23).

Predicts how a FLEX test ended - extinction / disruption / completion - from the ambient
conditions, and writes processed/analysis/flex_boundary.json (P(extinction) maps for the app).

What the outcomes mean (FLEX "Test end" column, PSI-69):
  extinction  - the flame went out while fuel was left (radiative or diluent-driven extinction)
  completion  - the droplet burned completely
  disruption  - the test ended early (droplet broke up / moved); not a combustion outcome as such
So P(extinction) answers "does this atmosphere let a burning droplet go out on its own?" -
relevant to suppression (CO2, He dilution) and to oxygen-enriched cabins. It is about liquid
droplets in microgravity, NOT about solid materials.

Honesty rules: trained only on PSI-69 rows of processed/tests.csv; predict_outcome() refuses
inputs outside the tested ranges per fuel; grid cells far from any real test are marked
unsupported (probability null).

Run from the repo root:  py -m analysis.flex_classifier      (prints the model card, writes JSON)
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "processed" / "tests.csv"
OUT_DIR = ROOT / "processed" / "analysis"
SOURCE = {"psi_id": "PSI-69", "doi": "10.60555/mbq8-0451"}
CLASSES = ["extinction", "disruption", "completion"]
FUELS = ["Heptane", "Methanol"]
FEATURES = ["heptane", "o2_frac", "ln_p", "x_co2", "x_he", "d0_mm"]
SEED = 0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows() -> list[dict]:
    rows = []
    with TESTS.open(encoding="utf-8", newline="") as f:
        for t in csv.DictReader(f):
            if t["investigation_id"] != "PSI-69" or t["outcome"] not in CLASSES:
                continue
            raw = json.loads(t["raw_row"])
            r = {"fuel": t["fuel"], "outcome": t["outcome"], "test_id": t["test_id"],
                 "o2_frac": _num(t["o2_frac"]), "pressure_kPa": _num(t["pressure_kPa"]),
                 "x_co2": _num(raw.get("CO initial ambient composition; mole fraction")) or 0.0,
                 "x_he": _num(raw.get("He initial ambient composition; mole fraction")) or 0.0,
                 "d0_mm": _num(t["d0_mm"])}
            if None in (r["o2_frac"], r["pressure_kPa"], r["d0_mm"]) or r["fuel"] not in FUELS:
                continue
            rows.append(r)
    return rows


def features(r: dict) -> list[float]:
    return [1.0 if r["fuel"] == "Heptane" else 0.0, r["o2_frac"], math.log(r["pressure_kPa"]),
            r["x_co2"], r["x_he"], r["d0_mm"]]


def _models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        # class_weight="balanced": completion has only 27 examples; without it the models almost
        # never predict it (recall 0.02-0.16). Compared on balanced accuracy, see model card.
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=SEED),
    }


def evaluate(rows):
    """Stratified 5-fold CV, repeated 5x. Returns per-model metrics."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = np.array([features(r) for r in rows])
    y = np.array([r["outcome"] for r in rows])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    out = {}
    for name, model in _models().items():
        acc, bal, rec = [], [], []
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            acc.append(accuracy_score(y[te], pred))
            bal.append(balanced_accuracy_score(y[te], pred))
            rec.append(recall_score(y[te], pred, labels=CLASSES, average=None, zero_division=0))
        out[name] = {"cv_accuracy": round(float(np.mean(acc)), 3), "cv_accuracy_sd": round(float(np.std(acc)), 3),
                     "cv_balanced_accuracy": round(float(np.mean(bal)), 3),
                     "recall": dict(zip(CLASSES, np.round(np.mean(rec, axis=0), 3).tolist()))}
    return out


# ------------------------------------------------------------------ class imbalance (#67)
# 179 extinction / 55 disruption / 27 completion. Class weights alone leave completion recall at 0.30:
# the 3-class model spends capacity on 'disruption' (droplet broke up / moved - a test ending, not a
# combustion outcome). The fire-safety question is two-class: among burns that ran to their end, did
# the flame go out with fuel left (extinction) or burn the droplet completely (completion)? That model
# drives P(goes out). Its decision threshold is chosen on out-of-fold probabilities of the training
# data only (maximise balanced accuracy), and evaluated with nested CV - the test fold never sees it.

BURNOUT_CLASSES = ["extinction", "completion"]
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.025), 3)


def _binary_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED,
                                  n_jobs=-1)


def _burnout_xy(rows):
    sel = [r for r in rows if r["outcome"] in BURNOUT_CLASSES]
    return sel, np.array([features(r) for r in sel]), np.array([r["outcome"] == "completion" for r in sel]).astype(int)


def _tune_threshold(X, y) -> float:
    """Threshold on P(completion) that maximises balanced accuracy on out-of-fold training predictions."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    p = cross_val_predict(_binary_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
                          method="predict_proba")[:, 1]
    scores = [balanced_accuracy_score(y, p >= t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


@lru_cache(maxsize=1)
def burnout_card() -> dict:
    return evaluate_burnout(load_rows())


@lru_cache(maxsize=1)
def imbalance_card() -> tuple:
    return tuple(imbalance_study(load_rows()))


def evaluate_burnout(rows, repeats: int = 3) -> dict:
    """Nested CV of the two-class model at the default 0.5 threshold and at the tuned threshold."""
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import RepeatedStratifiedKFold
    _, X, y = _burnout_xy(rows)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    res = {"default 0.5": [], "tuned": []}
    auc, ap, ts = [], [], []
    for tr, te in cv.split(X, y):
        t = _tune_threshold(X[tr], y[tr])
        p = _binary_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p)); ts.append(t)
        for name, thr in (("default 0.5", 0.5), ("tuned", t)):
            pr = (p >= thr).astype(int)
            res[name].append([accuracy_score(y[te], pr), balanced_accuracy_score(y[te], pr),
                              recall_score(y[te], pr, zero_division=0), precision_score(y[te], pr, zero_division=0),
                              recall_score(1 - y[te], 1 - pr, zero_division=0)])
    keys = ["cv_accuracy", "cv_balanced_accuracy", "recall_completion", "precision_completion", "recall_extinction"]
    out = {name: dict(zip(keys, np.round(np.mean(v, axis=0), 3).tolist())) for name, v in res.items()}
    return {"n_train": int(len(y)), "class_counts": {"extinction": int((y == 0).sum()), "completion": int(y.sum())},
            "baseline_accuracy": round(float(1 - y.mean()), 3),
            "baseline": "always predict extinction (the flame goes out)",
            "roc_auc_completion": round(float(np.mean(auc)), 3), "pr_auc_completion": round(float(np.mean(ap)), 3),
            "pr_auc_baseline": round(float(y.mean()), 3), "thresholds_chosen": [round(min(ts), 3), round(max(ts), 3)],
            "operating_points": out, "cv_scheme": f"nested: stratified 5-fold x{repeats} outside, 5-fold inside for the threshold"}


def imbalance_study(rows) -> list[dict]:
    """How each imbalance strategy changes completion recall (same CV for all). For the model card."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    X = np.array([features(r) for r in rows])
    y = np.array([r["outcome"] for r in rows])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    strategies = {
        "3 classes, no weighting": dict(class_weight=None),
        "3 classes, class_weight=balanced (previous model)": dict(class_weight="balanced"),
        "3 classes, balanced + bigger leaves (8)": dict(class_weight="balanced", min_samples_leaf=8),
    }
    out = []
    for name, kw in strategies.items():
        kw = {"min_samples_leaf": 3, **kw}
        m = []
        for tr, te in cv.split(X, y):
            pr = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **kw).fit(X[tr], y[tr]).predict(X[te])
            m.append([accuracy_score(y[te], pr), balanced_accuracy_score(y[te], pr),
                      recall_score(y[te], pr, labels=["completion"], average=None, zero_division=0)[0],
                      precision_score(y[te], pr, labels=["completion"], average=None, zero_division=0)[0]])
        a = np.round(np.mean(m, axis=0), 3).tolist()
        out.append({"strategy": name, "cv_accuracy": a[0], "cv_balanced_accuracy": a[1],
                    "recall_completion": a[2], "precision_completion": a[3]})
    b = burnout_card()   # the same full FLEX rows
    for name, op in b["operating_points"].items():
        out.append({"strategy": f"2 classes (disruption left out), balanced, threshold {name}",
                    "cv_accuracy": op["cv_accuracy"], "cv_balanced_accuracy": op["cv_balanced_accuracy"],
                    "recall_completion": op["recall_completion"], "precision_completion": op["precision_completion"],
                    "note": f"vs baseline {b['baseline_accuracy']} (2-class)"})
    return out


@lru_cache(maxsize=1)
def _fitted_burnout():
    rows = load_rows()
    _, X, y = _burnout_xy(rows)
    threshold = _tune_threshold(X, y)
    return _binary_model().fit(X, y), threshold


@lru_cache(maxsize=1)
def _fitted():
    rows = load_rows()
    metrics = evaluate(rows)
    X = np.array([features(r) for r in rows])
    y = np.array([r["outcome"] for r in rows])
    counts = {c: int((y == c).sum()) for c in CLASSES}
    baseline = max(counts.values()) / len(rows)
    # a model must beat "always extinction" on plain accuracy; among those, best balanced accuracy
    ok = {n: m for n, m in metrics.items() if m["cv_accuracy"] > baseline} or metrics
    chosen = max(ok, key=lambda n: ok[n]["cv_balanced_accuracy"])
    model = _models()[chosen]
    model.fit(X, y)
    valid = {}
    for fuel in FUELS:
        fr = [r for r in rows if r["fuel"] == fuel]
        valid[fuel] = {k: [round(min(r[k] for r in fr), 4), round(max(r[k] for r in fr), 4)]
                       for k in ("o2_frac", "pressure_kPa", "x_co2", "x_he", "d0_mm")}
    card = {
        "model": chosen, "features": FEATURES, "n_train": len(rows), "class_counts": counts,
        "baseline_accuracy": round(max(counts.values()) / len(rows), 3),
        "baseline": "always predict the most common outcome (extinction)",
        "cv": metrics, "cv_scheme": "stratified 5-fold, repeated 5x",
        "valid_range": valid, "source": SOURCE,
        "scope": "Liquid fuel droplets (methanol, heptane) in microgravity, FLEX on ISS. Not solid materials.",
    }
    return model, rows, card


def model_card() -> dict:
    return _fitted()[2]


def predict_outcome(fuel: str, o2_frac: float, pressure_kPa: float, x_co2: float = 0.0,
                    x_he: float = 0.0, d0_mm: float = 2.5) -> dict:
    model, _, card = _fitted()
    base = {"probs": None, "predicted": None, "reason": None, "n_train": card["n_train"],
            "cv_accuracy": card["cv"][card["model"]]["cv_accuracy"],
            "baseline_accuracy": card["baseline_accuracy"], "source": SOURCE}
    fuel = (fuel or "").strip().capitalize()
    if fuel not in FUELS:
        return {**base, "reason": f"fuel '{fuel}' not tested in FLEX (only {', '.join(FUELS)})"}
    vr = card["valid_range"][fuel]
    base["valid_range"] = vr
    for k, v in {"o2_frac": o2_frac, "pressure_kPa": pressure_kPa, "x_co2": x_co2, "x_he": x_he, "d0_mm": d0_mm}.items():
        lo, hi = vr[k]
        if v is None or not lo - 1e-9 <= v <= hi + 1e-9:
            return {**base, "reason": f"{k}={v} is outside the FLEX range for {fuel.lower()} ({lo}–{hi}) — not extrapolating."}
    r = {"fuel": fuel, "o2_frac": o2_frac, "pressure_kPa": pressure_kPa, "x_co2": x_co2, "x_he": x_he, "d0_mm": d0_mm}
    p = model.predict_proba(np.array([features(r)]))[0]
    probs = {c: round(float(p[list(model.classes_).index(c)]), 3) for c in CLASSES}
    bmodel, thr = _fitted_burnout()
    pc = float(bmodel.predict_proba(np.array([features(r)]))[0][1])
    burnout = {"p_completion": round(pc, 3), "p_goes_out": round(1 - pc, 3),
               "predicted": "completion" if pc >= 0.5 else "extinction",
               # safety view: saying "it goes out" when it burns completely is the dangerous error. The
               # conservative threshold (tuned on training folds) finds more completions, at lower precision.
               "conservative_threshold": thr, "may_burn_completely": bool(pc >= thr),
               "note": "two-class model trained on tests that were not disrupted; class-balanced, so a score, "
                       "not a calibrated frequency"}
    return {**base, "probs": probs, "predicted": max(probs, key=probs.get), "burnout": burnout}


def _near_real_test(rows, fuel, pressure_kPa, o2, xd, diluent):
    """A grid cell is supported if a real test of this fuel at similar pressure is close by."""
    for r in rows:
        if r["fuel"] != fuel or abs(math.log(r["pressure_kPa"] / pressure_kPa)) > 0.15:
            continue
        other = r["x_he"] if diluent == "x_co2" else r["x_co2"]
        if other > 0.01:
            continue
        if abs(r["o2_frac"] - o2) <= 0.03 and abs(r[diluent] - xd) <= 0.06:
            return True
    return False


def boundary_grids() -> dict:
    model, rows, card = _fitted()
    grids = []
    for fuel in FUELS:
        d0 = float(np.median([r["d0_mm"] for r in rows if r["fuel"] == fuel]))
        for p_kpa in (70.9, 101.3):
            for diluent in ("x_co2", "x_he"):
                vr = card["valid_range"][fuel]
                o2s = np.round(np.linspace(vr["o2_frac"][0], vr["o2_frac"][1], 23), 3)
                xs = np.round(np.linspace(0, vr[diluent][1], 21), 3)
                z = []
                for xd in xs:
                    row = []
                    for o2 in o2s:
                        if not _near_real_test(rows, fuel, p_kpa, o2, xd, diluent):
                            row.append(None)
                            continue
                        r = {"fuel": fuel, "o2_frac": o2, "pressure_kPa": p_kpa, "d0_mm": d0,
                             "x_co2": xd if diluent == "x_co2" else 0.0, "x_he": xd if diluent == "x_he" else 0.0}
                        pe = model.predict_proba(np.array([features(r)]))[0][list(model.classes_).index("extinction")]
                        row.append(round(float(pe), 3))
                    z.append(row)
                pts = [{"test_id": r["test_id"], "o2_frac": r["o2_frac"], diluent: r[diluent], "outcome": r["outcome"]}
                       for r in rows if r["fuel"] == fuel and abs(math.log(r["pressure_kPa"] / p_kpa)) <= 0.15
                       and (r["x_he"] if diluent == "x_co2" else r["x_co2"]) <= 0.01]
                grids.append({"fuel": fuel, "pressure_kPa": p_kpa, "diluent": diluent.replace("x_", "").upper(),
                              "d0_mm": round(d0, 2), "o2_frac": o2s.tolist(), "x_diluent": xs.tolist(),
                              "p_extinction": z, "tests": pts})
    return {"model_card": card, "grids": grids,
            "note": "p_extinction is null where no real FLEX test of that fuel lies within ±0.03 O2 and "
                    "±0.06 diluent mole fraction at similar pressure (±15 %): the model is not shown there."}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = boundary_grids()
    data["model_card"] = {**data["model_card"], "burnout": burnout_card(), "imbalance_study": list(imbalance_card())}
    (OUT_DIR / "flex_boundary.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    card = data["model_card"]
    print(f"model: {card['model']}  n={card['n_train']}  baseline acc {card['baseline_accuracy']}")
    for name, m in card["cv"].items():
        print(f"  {name:20s} acc {m['cv_accuracy']} ±{m['cv_accuracy_sd']}  balanced {m['cv_balanced_accuracy']}  recall {m['recall']}")
    b = card["burnout"]
    print(f"goes out vs burns out (n={b['n_train']}, baseline {b['baseline_accuracy']}, ROC-AUC {b['roc_auc_completion']}, "
          f"PR-AUC {b['pr_auc_completion']} vs {b['pr_auc_baseline']}):")
    for name, op in b["operating_points"].items():
        print(f"  threshold {name:12s} {op}")
    print("imbalance study:")
    for s in card["imbalance_study"]:
        print(f"  {s['strategy']:62s} acc {s['cv_accuracy']}  bal {s['cv_balanced_accuracy']}  "
              f"recall(completion) {s['recall_completion']}  precision {s['precision_completion']}")
    cells = [c for g in data["grids"] for row in g["p_extinction"] for c in row]
    print(f"wrote processed/analysis/flex_boundary.json: {len(data['grids'])} grids, "
          f"{sum(c is not None for c in cells)}/{len(cells)} cells supported by nearby tests")


if __name__ == "__main__":
    main()
