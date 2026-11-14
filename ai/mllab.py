"""ML Lab backend (issue #47): classical, cross-validated ML on the PSI test tables — no LLM.

    from ai.mllab import run, predict, default_config, shipped_config, mutual_information, correlations
    res = run(default_config("cfig"))          # dict, see run()
    predict(cfg, {"o2_frac": 0.18, ...})       # probabilities / value + CV error + 5 nearest real tests

Data comes only from `analysis.datasets` (main's data layer, one row per test, built from
processed/tests.csv with every ETL fix and flag). Nothing here re-parses tests.csv.

Honesty rules enforced here (the page only displays them):
  * every score is cross-validated (repeated, stratified for classification) and returned next to a
    baseline evaluated in the same folds (majority class / mean predictor);
  * a feature whose catalog kind is `post_test`, or whose `leak` list contains the target, is refused —
    unless `config["leakage_demo"]` is True, then the result is flagged `leakage_demo` for a red label;
  * warnings: < 10 samples per feature, a class with < 20 examples, collinear pairs |r| > 0.95,
    censored cfig burn time, cfig O2 of inverse flames is the burner stream;
  * predict() refuses inputs outside the min–max of the training data.
"""
from __future__ import annotations

import json
import os
import math
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from analysis import datasets as D

CLASSIFIERS = ["logreg", "random_forest", "gradient_boosting", "knn"]
REGRESSORS = ["ridge", "random_forest", "gradient_boosting"]
MODEL_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest",
                "gradient_boosting": "Gradient boosting", "knn": "k-nearest neighbours", "ridge": "Ridge regression"}
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "random_forest": {"n_estimators": 50, "max_depth": None, "min_samples_leaf": 2},  # 50 vs 100: same CV score, half the time
    "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
    "knn": {"n_neighbors": 7},
    "ridge": {"alpha": 1.0},
}
DEFAULT_TARGET = {"flex": "outcome", "cfig": "self_extinguished"}
COLLINEAR_R = 0.95
MIN_PER_FEATURE = 10
MIN_PER_CLASS = 20
N_JOBS = 1         # measured: threads cost more than they save on forests this small (35 s vs 17 s)
PERM_REPEATS = 5   # permutation-importance shuffles per feature and fold (batched: cheap)
FOLD_THREADS = min(8, os.cpu_count() or 1)  # CV folds run in parallel threads


class FeatureRefused(ValueError):
    """Raised when the selected features leak the target and the leakage demo is off."""

    def __init__(self, refused: list[dict]):
        self.refused = refused
        super().__init__("; ".join(f"{r['feature']}: {r['reason']}" for r in refused))


# ------------------------------------------------------------------ configs
def target_type(dataset: str, target: str) -> str:
    return D.DATASETS[dataset]["targets"][target]["type"]


def default_config(dataset: str, target: str | None = None) -> dict:
    target = target or DEFAULT_TARGET[dataset]
    model = "random_forest"
    return {"dataset": dataset, "target": target, "features": D.default_features(dataset, target),
            "scale": True, "impute": "drop", "exclude_flagged": False, "model": model,
            "params": dict(DEFAULT_PARAMS[model]),
            "class_weight": "balanced" if target_type(dataset, target) == "classification" else None,
            "cv": {"folds": 5, "repeats": 3, "seed": 0}, "leakage_demo": False}


def shipped_config() -> dict:
    """The configuration behind analysis.flex_classifier (the model on the Predict-extinction page)."""
    return {"dataset": "flex", "target": "outcome",
            "features": ["is_heptane", "o2_frac", "ln_pressure", "x_co2", "x_he", "d0_mm"],
            "scale": False, "impute": "drop", "exclude_flagged": False, "model": "random_forest",
            "params": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
            "class_weight": "balanced", "cv": {"folds": 5, "repeats": 5, "seed": 0}, "leakage_demo": False}


def _key(config: dict) -> str:
    """Canonical cache key: feature order does not change the model, so features are put in catalog order."""
    order = {f["name"]: i for i, f in enumerate(D.feature_catalog(config["dataset"]))}
    canon = dict(config, features=sorted(config.get("features") or [], key=lambda f: order.get(f, len(order))))
    return json.dumps(canon, sort_keys=True, default=str)


# ------------------------------------------------------------------ catalog checks
def catalog(dataset: str) -> dict[str, dict]:
    return {f["name"]: f for f in D.feature_catalog(dataset)}


def check_features(dataset: str, target: str, features: list[str]) -> list[dict]:
    """Features that would leak the answer for `target` -> [{feature, reason}]."""
    cat, out = catalog(dataset), []
    for f in features:
        spec = cat.get(f)
        if f == target:
            out.append({"feature": f, "reason": "is the target itself"})
        elif spec is None:
            out.append({"feature": f, "reason": "not in the feature catalog"})
        elif spec["kind"] == "post_test":
            out.append({"feature": f, "reason": f"measured during/after the test ({spec['why']}) — leaks the answer"})
        elif target in spec.get("leak", []):
            out.append({"feature": f, "reason": f"its catalog entry says it leaks '{target}'"})
    return out


# ------------------------------------------------------------------ data preparation
def _frame(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int, list[str]]:
    """X (selected features), y, meta rows, n_dropped, notes."""
    ds, target, feats = config["dataset"], config["target"], list(config["features"])
    df = D.load(ds)
    notes = []
    n0 = len(df)
    df = df[df[target].notna()]
    if config.get("exclude_flagged"):
        corrected = df["flags"].fillna("").map(lambda f: any(p.startswith("F") for p in f.split(" | ") if p))
        notes.append(f"{int(corrected.sum())} rows with ETL corrections excluded")
        df = df[~corrected]
    if config.get("impute", "drop") == "drop":
        df = df.dropna(subset=[f for f in feats if f in df.columns])
    n_dropped = n0 - len(df)
    meta_cols = [c for c in ("test_id", "date_iso", "flags", "fuel", "flame_type", "fuel_species", "o2_role",
                             "outcome", "outcome_text") if c in df.columns]
    return df[feats].reset_index(drop=True), df[target].reset_index(drop=True), \
        df[meta_cols + [c for c in [target] if c not in meta_cols]].reset_index(drop=True), n_dropped, notes


def _categorical(dataset: str, features: list[str]) -> list[str]:
    cat = catalog(dataset)
    return [f for f in features if cat.get(f, {}).get("unit") == "category"]


def _pipeline(config: dict, task: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    feats = list(config["features"])
    cats = _categorical(config["dataset"], feats)
    nums = [f for f in feats if f not in cats]
    num_steps = [("impute", SimpleImputer(strategy="median"))]  # no-op when rows with NaN were dropped
    if config.get("scale"):
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([("num", Pipeline(num_steps), nums)]
                            + ([("cat", OneHotEncoder(handle_unknown="ignore"), cats)] if cats else []))
    p, cw, seed = dict(config.get("params") or {}), config.get("class_weight"), config["cv"].get("seed", 0)
    m = config["model"]
    if task == "classification":
        est = {"logreg": lambda: LogisticRegression(C=float(p.get("C", 1.0)), max_iter=3000, class_weight=cw),
               "random_forest": lambda: RandomForestClassifier(
                   n_estimators=int(p.get("n_estimators", 150)), max_depth=p.get("max_depth"),
                   min_samples_leaf=int(p.get("min_samples_leaf", 2)), class_weight=cw, random_state=seed,
                   n_jobs=N_JOBS),
               "gradient_boosting": lambda: GradientBoostingClassifier(
                   n_estimators=int(p.get("n_estimators", 150)), learning_rate=float(p.get("learning_rate", 0.1)),
                   max_depth=int(p.get("max_depth", 3)), random_state=seed),
               "knn": lambda: KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 7)))}
    else:
        est = {"ridge": lambda: Ridge(alpha=float(p.get("alpha", 1.0))),
               "random_forest": lambda: RandomForestRegressor(
                   n_estimators=int(p.get("n_estimators", 150)), max_depth=p.get("max_depth"),
                   min_samples_leaf=int(p.get("min_samples_leaf", 2)), random_state=seed, n_jobs=N_JOBS),
               "gradient_boosting": lambda: GradientBoostingRegressor(
                   n_estimators=int(p.get("n_estimators", 150)), learning_rate=float(p.get("learning_rate", 0.1)),
                   max_depth=int(p.get("max_depth", 3)), random_state=seed)}
    if m not in est:
        raise ValueError(f"model '{m}' is not available for {task} (choose from {', '.join(est)})")
    return Pipeline([("prep", pre), ("model", est[m]())])


# ------------------------------------------------------------------ warnings
def warnings_for(config: dict, X: pd.DataFrame, y: pd.Series) -> list[str]:
    ds, target = config["dataset"], config["target"]
    task = target_type(ds, target)
    out = []
    k = max(1, len(config["features"]))
    if len(X) / k < MIN_PER_FEATURE:
        out.append(f"Only {len(X) / k:.1f} samples per feature (< {MIN_PER_FEATURE}): high risk of overfitting — "
                   "use fewer features.")
    if task == "classification":
        for cls, n in y.value_counts().items():
            if n < MIN_PER_CLASS:
                out.append(f"Class '{_cls(cls)}' has only {n} examples (< {MIN_PER_CLASS}): its recall is unreliable.")
    nums = [f for f in config["features"] if f not in _categorical(ds, config["features"])]
    if len(nums) >= 2:
        c = X[nums].corr().abs()
        pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(nums) for b in nums[i + 1:]
                 if pd.notna(c.loc[a, b]) and c.loc[a, b] > COLLINEAR_R]
        if pairs:
            listed = ", ".join(f"{a} ~ {b} (|r| = {r:.2f})" for a, b, r in pairs)
            out.append(f"Almost collinear pairs (|r| > {COLLINEAR_R}): {listed}. Each pair carries the same information, "
                       "which splits importance between them and can destabilise linear models — consider keeping one.")
    if ds == "cfig" and target == "burn_time_s":
        out.append("cfig burn time is censored: in flow-terminated tests the operator stopped the flame, so the "
                   "true burn time is longer than recorded. Treat predictions as lower bounds.")
    if ds == "cfig" and "o2_frac" in config["features"]:
        out.append("cfig o2_frac: for inverse flames this is the O₂ in the burner stream, not an ambient "
                   "atmosphere (o2_role = burner) — do not read it as a cabin O₂ level.")
    return out


# Readable names for the 0/1 targets (also avoids a scikit-learn quirk: string labels '0'/'1' break
# class_weight='balanced').
CLASS_LABELS = {
    ("flex", "went_out"): {1: "went out", 0: "burned completely"},
    ("cfig", "hot_ignition"): {1: "ignited", 0: "no hot ignition"},
    ("cfig", "self_extinguished"): {1: "self-extinguished", 0: "flow-terminated"},
}


def class_label(dataset: str, target: str, v) -> str:
    names = CLASS_LABELS.get((dataset, target))
    try:
        f = float(v)
        if names and f.is_integer() and int(f) in names:
            return names[int(f)]
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


def _labels(dataset: str, target: str, y: pd.Series) -> pd.Series:
    return y.map(lambda v: class_label(dataset, target, v)).astype(object)


def _cls(v):
    return class_label("", "", v)


# ------------------------------------------------------------------ disk cache (committed)
# Precomputed results for the default config of every dataset × target and for the shipped model
# (`python -m ai.mllab --precompute`), so the page opens instantly and the demo never waits; any
# changed setting trains live. Keys include a fingerprint of the dataset rows, so an ETL change makes
# the cached entries unreachable (stale results are never served) — re-run --precompute.
RUNS_FILE = None  # set below (needs ai.config)


@lru_cache(maxsize=None)
def data_fingerprint(dataset: str) -> str:
    import hashlib

    return hashlib.sha256(pd.util.hash_pandas_object(D.load(dataset), index=False).values.tobytes()).hexdigest()[:16]


def _disk_key(kind: str, config: dict) -> str:
    """Config + data fingerprint + scikit-learn version: forest results differ between sklearn releases,
    so a result computed under another version is never served (versions are pinned, #56)."""
    import sklearn

    return f"{kind}|{data_fingerprint(config['dataset'])}|sklearn {sklearn.__version__}|{_key(config)}"


@lru_cache(maxsize=1)
def _disk() -> dict:
    try:
        return json.loads(RUNS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


# ------------------------------------------------------------------ run
_LIVE: dict[str, dict] = {}  # results trained in this process (for peek)


def run(config: dict) -> dict:
    """Cross-validate `config`. Raises FeatureRefused for leaking features unless config['leakage_demo']."""
    hit = _disk().get(_disk_key("run", config))
    if hit is not None:
        return _fresh_warnings(dict(json.loads(hit), from_cache=True), config)
    res = json.loads(_run_cached(_key(config)))
    _LIVE[_key(config)] = res
    return res


def peek(config: dict) -> dict | None:
    """The result for `config` if it is precomputed or was trained earlier in this process — never trains."""
    hit = _disk().get(_disk_key("run", config))
    if hit is not None:
        return _fresh_warnings(dict(json.loads(hit), from_cache=True), config)
    return _LIVE.get(_key(config))


def _fresh_warnings(res: dict, config: dict) -> dict:
    """Scores may come from the committed cache; warnings are cheap, so always recompute them with today's code."""
    X, y, _, _, notes = _frame(json.loads(_key(config)))
    res["warnings"] = warnings_for(config, X, y) + notes
    return res


@lru_cache(maxsize=64)
def _run_cached(key: str) -> str:
    return json.dumps(_run(json.loads(key)), default=float)


def _run(config: dict) -> dict:
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error,
                                 r2_score, recall_score)
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

    t0 = time.perf_counter()
    ds, target = config["dataset"], config["target"]
    if not config.get("features"):
        raise ValueError("select at least one feature")
    refused = check_features(ds, target, config["features"])
    if refused and not config.get("leakage_demo"):
        raise FeatureRefused(refused)
    task = target_type(ds, target)
    X, y, meta, n_dropped, notes = _frame(config)
    if task == "classification":
        y = _labels(ds, target, y)
    folds, repeats, seed = int(config["cv"]["folds"]), int(config["cv"]["repeats"]), int(config["cv"]["seed"])
    if len(X) < folds * 2:
        raise ValueError(f"only {len(X)} usable rows — not enough for {folds}-fold cross-validation")
    warns = warnings_for(config, X, y) + notes
    pipe = _pipeline(config, task)

    if task == "classification":
        classes = sorted(y.unique(), key=str)
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        base = DummyClassifier(strategy="most_frequent")
    else:
        classes = None
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        base = DummyRegressor(strategy="mean")

    def one_fold(i, tr, te):
        """Everything computed on one CV fold (independent of the other folds -> run in parallel)."""
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        b = clone(base).fit(X.iloc[tr], y.iloc[tr])
        yt, pred, bpred = y.iloc[te], m.predict(X.iloc[te]), b.predict(X.iloc[te])
        out = {"i": i}
        if task == "classification":
            out["s"] = {"accuracy": accuracy_score(yt, pred), "balanced_accuracy": balanced_accuracy_score(yt, pred)}
            out["b"] = {"accuracy": accuracy_score(yt, bpred), "balanced_accuracy": balanced_accuracy_score(yt, bpred)}
            out["recall"] = recall_score(yt, pred, labels=classes, average=None, zero_division=0)
            if i < folds:  # first repeat: every test predicted exactly once, out of fold
                out["cm"] = confusion_matrix(yt, pred, labels=classes)
        else:
            out["s"] = {"r2": r2_score(yt, pred), "mae": mean_absolute_error(yt, pred)}
            out["b"] = {"r2": r2_score(yt, bpred), "mae": mean_absolute_error(yt, bpred)}
        if i < folds:  # permutation importance on the held-out fold of the first repeat
            out["imp"] = _permutation_importance(m, X.iloc[te], yt, task, seed + i)
        return out

    from joblib import Parallel, delayed

    # threads, one CV fold each: tree building releases the GIL, and threads avoid process start-up on Windows
    parts = Parallel(n_jobs=FOLD_THREADS, backend="threading")(
        delayed(one_fold)(i, tr, te) for i, (tr, te) in enumerate(cv.split(X, y)))
    parts.sort(key=lambda p: p["i"])
    scores = {k: [p["s"][k] for p in parts] for k in parts[0]["s"]}
    base_scores = {k: [p["b"][k] for p in parts] for k in parts[0]["b"]}
    recalls = [p["recall"] for p in parts if "recall" in p]
    cm = sum(p["cm"] for p in parts if "cm" in p) if task == "classification" else None
    imps = [p["imp"] for p in parts if "imp" in p]

    def ms(v):
        return {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}

    imp = np.array(imps)
    importance = sorted(({"feature": f, "mean": round(float(imp[:, j].mean()), 4), "sd": round(float(imp[:, j].std()), 4)}
                         for j, f in enumerate(config["features"])), key=lambda d: -d["mean"])
    res = {
        "config": config, "task": task, "n_rows": int(len(X)), "n_dropped": int(n_dropped),
        "n_features": len(config["features"]), "metrics": {k: ms(v) for k, v in scores.items()},
        "baseline": {"name": "always the most common class" if task == "classification" else "always the mean",
                     **{k: ms(v) for k, v in base_scores.items()}},
        "permutation_importance": importance,
        "permutation_scoring": "balanced accuracy" if task == "classification" else "R²",
        "warnings": warns, "refused": refused, "leakage_demo": bool(refused and config.get("leakage_demo")),
        "cv": {"scheme": f"{'stratified ' if task == 'classification' else ''}{folds}-fold, repeated {repeats}×",
               "n_fits": folds * repeats},
        "source": D.DATASETS[ds]["source"], "scope": D.DATASETS[ds]["scope"],
    }
    if task == "classification":
        res["classes"] = [str(c) for c in classes]
        res["class_counts"] = {str(k): int(v) for k, v in y.value_counts().items()}
        res["recall"] = {str(c): round(float(r), 4) for c, r in zip(classes, np.mean(recalls, axis=0))}
        res["confusion_matrix"] = {"labels": [str(c) for c in classes], "counts": cm.tolist(),
                                   "note": "out-of-fold predictions of the first CV repeat (each test once)"}
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, task: str, seed: int) -> np.ndarray:
    """Drop in score when one feature's values are shuffled (same definition as
    sklearn.inspection.permutation_importance), but all shuffled copies of the fold are stacked and
    predicted in ONE call: a forest's predict cost is per call, not per row, so this is ~25x faster."""
    from sklearn.metrics import balanced_accuracy_score, r2_score

    score = balanced_accuracy_score if task == "classification" else r2_score
    rng = np.random.default_rng(seed)
    n, feats = len(X), list(X.columns)
    blocks = []
    for f in feats:
        for _ in range(PERM_REPEATS):
            Xp = X.copy()
            Xp[f] = X[f].to_numpy()[rng.permutation(n)]
            blocks.append(Xp)
    preds = model.predict(pd.concat([X] + blocks, ignore_index=True))
    base = score(y, preds[:n])
    drops = np.array([base - score(y, preds[n * (b + 1): n * (b + 2)]) for b in range(len(blocks))])
    return drops.reshape(len(feats), PERM_REPEATS).mean(axis=1)


def learning_curve(config: dict) -> dict:
    """Train / validation score vs training-set size (3-fold, 3 sizes): does more data still help?
    Separate from run() (and cached) so the page can show the CV metrics first."""
    hit = _disk().get(_disk_key("learning_curve", config))
    if hit is not None:
        return json.loads(hit)
    return json.loads(_learning_curve_cached(_key(config)))


@lru_cache(maxsize=32)
def _learning_curve_cached(key: str) -> str:
    from sklearn.model_selection import KFold, StratifiedKFold, learning_curve as lc

    config = json.loads(key)
    refused = check_features(config["dataset"], config["target"], config["features"])
    if refused and not config.get("leakage_demo"):
        raise FeatureRefused(refused)
    task = target_type(config["dataset"], config["target"])
    X, y, *_ = _frame(config)
    if task == "classification":
        y = _labels(config["dataset"], config["target"], y)
    pipe = _pipeline(config, task)
    seed = int(config["cv"]["seed"])
    cv = (StratifiedKFold(3, shuffle=True, random_state=seed) if task == "classification"
          else KFold(3, shuffle=True, random_state=seed))
    scoring = "balanced_accuracy" if task == "classification" else "r2"
    sizes, tr, va = lc(pipe, X, y, cv=cv, train_sizes=np.linspace(0.3, 1.0, 3), scoring=scoring, shuffle=True,
                       random_state=seed)
    return json.dumps({"scoring": scoring.replace("_", " "), "train_sizes": sizes.tolist(),
                       "train_mean": np.round(tr.mean(axis=1), 4).tolist(),
                       "valid_mean": np.round(va.mean(axis=1), 4).tolist(),
                       "valid_sd": np.round(va.std(axis=1), 4).tolist()})


# ------------------------------------------------------------------ exploration helpers
def mutual_information(dataset: str, target: str, features: list[str]) -> list[dict]:
    """Univariate signal: mutual information of each feature with the target (rows with both present)."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    df = D.load(dataset)
    task = target_type(dataset, target)
    cats = set(_categorical(dataset, features))
    out = []
    for f in features:
        sub = df[[f, target]].dropna()
        if len(sub) < 10:
            continue
        x = sub[[f]].astype("category").apply(lambda s: s.cat.codes) if f in cats else sub[[f]].astype(float)
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        y = _labels(dataset, target, sub[target]) if task == "classification" else sub[target]
        mi = fn(x.values, y.values, discrete_features=[f in cats], random_state=0)[0]
        out.append({"feature": f, "mi": round(float(mi), 4), "n": int(len(sub))})
    return sorted(out, key=lambda d: -d["mi"])


def correlations(dataset: str, features: list[str]) -> pd.DataFrame:
    df = D.load(dataset)
    nums = [f for f in features if f not in _categorical(dataset, features) and f in df.columns]
    return df[nums].corr()


# ------------------------------------------------------------------ prediction
@lru_cache(maxsize=16)
def _fitted(key: str):
    config = json.loads(key)
    task = target_type(config["dataset"], config["target"])
    X, y, meta, _, _ = _frame(config)
    if task == "classification":
        y = _labels(config["dataset"], config["target"], y)
    model = _pipeline(config, task).fit(X, y)
    return model, X, y, meta, task


def input_ranges(config: dict) -> dict:
    """{feature: [min, max]} of the training rows (numeric) or the seen categories."""
    _, X, *_ = _fitted(_key(config))
    cats = set(_categorical(config["dataset"], config["features"]))
    return {f: (sorted(X[f].dropna().astype(str).unique().tolist()) if f in cats
                else [float(X[f].min()), float(X[f].max())]) for f in config["features"]}


def predict(config: dict, inputs: dict) -> dict:
    """Prediction for one input row, or a refusal. Includes the CV error of this config and the
    5 nearest real tests (standardised distance over the numeric features)."""
    if check_features(config["dataset"], config["target"], config["features"]) and not config.get("leakage_demo"):
        raise FeatureRefused(check_features(config["dataset"], config["target"], config["features"]))
    model, X, y, meta, task = _fitted(_key(config))
    res = run(config)
    out = {"task": task, "refused": None, "n_train": int(len(X)), "leakage_demo": res["leakage_demo"]}
    if task == "classification":
        out["cv"] = {"accuracy": res["metrics"]["accuracy"], "balanced_accuracy": res["metrics"]["balanced_accuracy"],
                     "baseline_accuracy": res["baseline"]["accuracy"]["mean"]}
    else:
        out["cv"] = {"mae": res["metrics"]["mae"], "r2": res["metrics"]["r2"]}
    ranges = input_ranges(config)
    for f, rng in ranges.items():
        v = inputs.get(f)
        if v is None or (isinstance(rng[0], str) and str(v) not in rng):
            out["refused"] = f"{f} = {v!r} was never tested (tested: {', '.join(map(str, rng))})"
            break
        if not isinstance(rng[0], str) and not (rng[0] - 1e-9 <= float(v) <= rng[1] + 1e-9):
            out["refused"] = (f"{f} = {float(v):g} is outside the tested range {rng[0]:g}–{rng[1]:g} — "
                              "the model would be extrapolating, so no prediction is made.")
            break
    if out["refused"] is None:
        row = pd.DataFrame([{f: inputs[f] for f in config["features"]}])
        if task == "classification":
            p = model.predict_proba(row)[0]
            out["probs"] = {str(c): round(float(v), 4) for c, v in zip(model.classes_, p)}
            out["predicted"] = str(model.classes_[int(np.argmax(p))])
        else:
            out["value"] = round(float(model.predict(row)[0]), 4)
            out["unit"] = D.DATASETS[config["dataset"]]["targets"][config["target"]].get("unit", "")
    out["nearest"] = nearest_tests(config, inputs, X, meta)
    return out


def nearest_tests(config: dict, inputs: dict, X: pd.DataFrame, meta: pd.DataFrame, k: int = 5) -> list[dict]:
    nums = [f for f in config["features"] if f not in _categorical(config["dataset"], config["features"])
            and inputs.get(f) is not None]
    if not nums:
        return []
    A = X[nums].astype(float)
    mu, sd = A.mean(), A.std().replace(0, 1.0)
    q = (pd.Series({f: float(inputs[f]) for f in nums}) - mu) / sd
    d = (((A - mu) / sd - q) ** 2).sum(axis=1, min_count=1).pow(0.5)
    out = []
    for i in d.nsmallest(k).index:
        row = {"test_id": meta.loc[i, "test_id"], "distance": round(float(d[i]), 3)}
        row.update({f: (None if pd.isna(X.loc[i, f]) else (round(float(X.loc[i, f]), 4)
                                                          if not isinstance(X.loc[i, f], str) else X.loc[i, f]))
                    for f in config["features"]})
        row["actual"] = class_label(config["dataset"], config["target"], meta.loc[i, config["target"]])
        out.append(row)
    return out


def precomputed_configs() -> list[dict]:
    return [default_config(ds, t) for ds in D.DATASETS for t in D.DATASETS[ds]["targets"]] + [shipped_config()]


def precompute() -> None:
    """Compute and store the default + shipped runs (and learning curves) in RUNS_FILE. Incremental:
    entries whose key (config + data fingerprint) is unchanged are kept, only missing ones are trained."""
    old, store = _disk(), {}
    for cfg in precomputed_configs():
        t = time.perf_counter()
        for kind, fn in (("run", _run_cached), ("learning_curve", _learning_curve_cached)):
            k = _disk_key(kind, cfg)
            store[k] = old.get(k) or fn(_key(cfg))
        r = json.loads(store[_disk_key("run", cfg)])
        k = "accuracy" if r["task"] == "classification" else "r2"
        print(f"{cfg['dataset']:4s} {cfg['target']:24s} {cfg['model']:14s} n={r['n_rows']:3d}  "
              f"{k} {r['metrics'][k]['mean']:.3f} ± {r['metrics'][k]['sd']:.3f}  baseline {r['baseline'][k]['mean']:.3f}"
              f"  ({time.perf_counter() - t:.1f} s)")
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(store, indent=0, sort_keys=True), encoding="utf-8")
    _disk.cache_clear()
    print(f"wrote {len(store)} entries -> {RUNS_FILE}")


def _init_paths():
    global RUNS_FILE
    from .config import AI_OUT
    RUNS_FILE = AI_OUT / "mllab_runs.json"


_init_paths()

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true", help="write processed/ai/mllab_runs.json")
    if ap.parse_args().precompute:
        precompute()
    else:
        for cfg in (default_config("flex"), default_config("cfig"), shipped_config()):
            r = run(cfg)
            print(cfg["dataset"], cfg["target"], {k: v["mean"] for k, v in r["metrics"].items()},
                  "from cache" if r.get("from_cache") else f"{r['seconds']} s")
