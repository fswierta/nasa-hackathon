"""ML Lab backend (issues #47, #59): classical, cross-validated ML on the PSI test tables — no LLM.

    from ai.mllab import run, predict, default_config, shipped_config, mutual_information, correlations
    res = run(default_config("cfig"))          # dict, see run()
    predict(cfg, {"o2_frac": 0.18, ...})       # probabilities / value + CV error + 5 nearest real tests
    start_config("cfig")                       # the shipped model's config if one exists (#59), else the default
    leaderboard(cfg); ablation(cfg)            # every algorithm / raw vs engineered features, same CV (#59)
    partial_dependence(cfg); diagnostics(cfg)  # what it learned; calibration + worst-predicted tests (#59)
    encode_config(cfg) / decode_config(token)  # ?config= deep links (decode validates untrusted input)

The shipped models are the ones in analysis.registry (Models hub): SHIPPED maps each registry id to the Lab
configuration that reproduces its cross-validated score, and ai/tests/test_mllab.py checks they agree.

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


# ------------------------------------------------------------------ shipped models (analysis.registry, #59)
# Row subsets a config may select ("rows": {column: value}); CFI-G's shipped models use normal flames only,
# because the O2 of an inverse flame is the burner stream, not an atmosphere.
ROW_FILTERS = {"cfig": {"flame_type": ["normal", "inverse"]}}
# the class a probability / partial-dependence curve is about ("P(...)")
POSITIVE = {("flex", "outcome"): "extinction", ("flex", "went_out"): "went out",
            ("cfig", "hot_ignition"): "ignited", ("cfig", "self_extinguished"): "self-extinguished"}
# registry feature names that differ from the analysis.datasets column names
_REGISTRY_NAMES = {"heptane": "is_heptane", "ln_p": "ln_pressure"}
SHIP_RULE = ("must beat always guessing the most common class on accuracy; among those, the best balanced "
             "accuracy wins (analysis/flex_classifier.py) — rare outcomes count as much as common ones")


def _shipped_table() -> dict[str, dict]:
    """registry model id -> the Lab config that reproduces it. Features are read from the model modules, so a
    change there shows up here; the hyper-parameters are those modules' (checked by the reproduction test)."""
    from analysis import flex_classifier as F
    from analysis import scenario as S

    rf = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3}
    base = {"scale": False, "impute": "drop", "exclude_flagged": False, "model": "random_forest",
            "class_weight": "balanced", "leakage_demo": False}
    gas = dict(base, dataset="cfig", features=list(S.GAS_FEATURES), rows={"flame_type": "normal"},
               cv={"folds": 5, "repeats": 3, "seed": S.SEED})
    return {
        "flex_outcome": dict(base, dataset="flex", target="outcome",
                             features=[_REGISTRY_NAMES.get(f, f) for f in F.FEATURES], params=dict(rf),
                             cv={"folds": 5, "repeats": 5, "seed": F.SEED}),
        "cfig_ignition": dict(gas, target="hot_ignition", params=dict(rf)),
        "cfig_self_extinguish": dict(gas, target="self_extinguished", params=dict(rf)),
    }


def registry_spec(model_id: str) -> dict | None:
    """The registry entry (metrics, title, scope) as the Models hub shows it: analysis.registry.specs() is the
    committed output of describe(), so the Lab and the hub always quote the same numbers."""
    from analysis import registry
    return next((m for m in registry.specs() if m["id"] == model_id), None)


def shipped_models() -> list[dict]:
    """[{model_id, config, spec}] for every registry model the Lab can reproduce."""
    return [{"model_id": mid, "config": cfg, "spec": registry_spec(mid)} for mid, cfg in _shipped_table().items()]


def shipped_for(dataset: str, target: str) -> dict | None:
    return next((s for s in shipped_models() if s["config"]["dataset"] == dataset
                 and s["config"]["target"] == target), None)


def shipped_config(model_id: str = "flex_outcome") -> dict:
    """The configuration behind a shipped model (default: analysis.flex_classifier, the Predict-extinction page)."""
    return json.loads(json.dumps(_shipped_table()[model_id]))


def start_config(dataset: str, target: str | None = None) -> dict:
    """What the Lab opens on (#59): the shipped model for this dataset × target, else the default config."""
    target = target or DEFAULT_TARGET[dataset]
    s = shipped_for(dataset, target)
    return json.loads(json.dumps(s["config"])) if s else default_config(dataset, target)


def is_shipped(config: dict) -> str | None:
    """The registry id when `config` is exactly a shipped model's config, else None."""
    k = _key(config)
    return next((mid for mid, c in _shipped_table().items() if _key(c) == k), None)


@lru_cache(maxsize=1)
def _shipped_orders() -> tuple:
    return tuple((c["dataset"], tuple(c["features"])) for c in _shipped_table().values())


def canonical_features(dataset: str, features: list[str]) -> list[str]:
    """The column order a feature set is trained in. A forest's random feature draws depend on column order, so
    the order is fixed: a shipped model's feature set keeps that model's order (so the Lab reproduces the
    registry's numbers exactly), any other set is put in catalog order."""
    s = set(features or [])
    for ds, order in _shipped_orders():
        if ds == dataset and set(order) == s:
            return list(order)
    cat = {f["name"]: i for i, f in enumerate(D.feature_catalog(dataset))}
    return sorted(features or [], key=lambda f: cat.get(f, len(cat)))


def _key(config: dict) -> str:
    """Canonical cache key (and the config that is actually trained): features in canonical_features() order;
    empty optional fields (rows, require) are left out, so older keys stay valid."""
    canon = dict(config, features=canonical_features(config["dataset"], config.get("features")))
    for opt in ("rows", "require"):
        if not canon.get(opt):
            canon.pop(opt, None)
    if canon.get("require"):
        canon["require"] = sorted(canon["require"])
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
    for col, val in (config.get("rows") or {}).items():
        df = df[df[col] == val]
    df = df[df[target].notna()]
    if config.get("exclude_flagged"):
        corrected = df["flags"].fillna("").map(lambda f: any(p.startswith("F") for p in f.split(" | ") if p))
        notes.append(f"{int(corrected.sum())} rows with ETL corrections excluded")
        df = df[~corrected]
    if config.get("require"):  # ablation: every feature set is scored on the same rows
        df = df.dropna(subset=[f for f in config["require"] if f in df.columns])
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
    if ds == "cfig" and "o2_frac" in config["features"] and (config.get("rows") or {}).get("flame_type") != "normal":
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


# ------------------------------------------------------------------ extra analyses, cached (#59)
_EXTRA: dict[str, str] = {}  # (kind | config key) -> JSON, computed in this process
_EXTRA_MAX = 64


def _extra(kind: str, config: dict, fn) -> dict:
    """Disk (precomputed) -> this process -> compute. Results are JSON so callers can't mutate the cache."""
    hit = _disk().get(_disk_key(kind, config))
    if hit is None:
        k = f"{kind}|{_key(config)}"
        if k not in _EXTRA:
            if len(_EXTRA) >= _EXTRA_MAX:
                _EXTRA.pop(next(iter(_EXTRA)))
            _EXTRA[k] = json.dumps(fn(json.loads(_key(config))), default=float)
        hit = _EXTRA[k]
    return json.loads(hit)


def _peek_extra(kind: str, config: dict) -> dict | None:
    hit = _disk().get(_disk_key(kind, config)) or _EXTRA.get(f"{kind}|{_key(config)}")
    return json.loads(hit) if hit else None


def _refuse_leaks(config: dict) -> None:
    refused = check_features(config["dataset"], config["target"], config["features"])
    if refused and not config.get("leakage_demo"):
        raise FeatureRefused(refused)


def positive_class(dataset: str, target: str) -> str | None:
    """The class a probability is about; None for regression targets."""
    if target_type(dataset, target) != "classification":
        return None
    if (dataset, target) in POSITIVE:
        return POSITIVE[(dataset, target)]
    classes = D.DATASETS[dataset]["targets"][target].get("classes") or []
    return class_label(dataset, target, classes[0]) if classes else None


def _score_key(task: str) -> str:
    return "balanced_accuracy" if task == "classification" else "r2"


# ---- leaderboard: every algorithm on the current features, rows and CV
def leaderboard_configs(config: dict) -> list[dict]:
    """One config per algorithm. The current model keeps its parameters, the others use DEFAULT_PARAMS;
    logistic regression / k-NN / ridge are always scaled (they need it, trees don't care)."""
    task = target_type(config["dataset"], config["target"])
    algos = CLASSIFIERS if task == "classification" else REGRESSORS
    cw = config.get("class_weight") or ("balanced" if task == "classification" else None)
    out = []
    for a in algos:
        c = dict(config, model=a,
                 params=dict(config.get("params") or {}) if a == config["model"] else dict(DEFAULT_PARAMS[a]))
        if a in ("logreg", "knn", "ridge"):
            c["scale"] = True
        if task == "classification":
            c["class_weight"] = None if a == "knn" else (config.get("class_weight") if a == config["model"] else cw)
        out.append(c)
    return out


def _board(config: dict, results: list[dict]) -> dict:
    task = target_type(config["dataset"], config["target"])
    sk = _score_key(task)
    rows = []
    for c, r in zip(leaderboard_configs(config), results):
        row = {"model": c["model"], "label": MODEL_LABELS[c["model"]], "params": c["params"],
               "score": r["metrics"][sk], "seconds": r["seconds"], "from_cache": bool(r.get("from_cache")),
               "is_current": c["model"] == config["model"], "shipped": is_shipped(c), "config": c}
        if task == "classification":
            row["accuracy"] = r["metrics"]["accuracy"]
            row["beats_baseline"] = r["metrics"]["accuracy"]["mean"] > r["baseline"]["accuracy"]["mean"]
        else:
            row["mae"] = r["metrics"]["mae"]
            row["beats_baseline"] = r["metrics"]["r2"]["mean"] > r["baseline"]["r2"]["mean"]
        rows.append(row)
    rows.sort(key=lambda d: -d["score"]["mean"])
    ok = [r for r in rows if r["beats_baseline"]] or rows
    base = results[0]["baseline"]
    return {"task": task, "score": sk.replace("_", " ").replace("r2", "R²"), "rows": rows,
            "baseline": {"label": "Baseline: " + base["name"], "score": base[sk],
                         **({"accuracy": base["accuracy"]} if task == "classification" else {"mae": base["mae"]})},
            "pick": ok[0]["model"], "rule": SHIP_RULE if task == "classification" else
            "must beat predicting the mean (R² > 0); among those, the highest cross-validated R² wins",
            "n_rows": results[0]["n_rows"], "cv": results[0]["cv"]["scheme"]}


def leaderboard(config: dict, progress=None) -> dict:
    """Cross-validate every algorithm (each run is cached like run()). progress(i, n, label) after each model."""
    _refuse_leaks(config)
    cfgs, results = leaderboard_configs(config), []
    for i, c in enumerate(cfgs):
        results.append(run(c))
        if progress:
            progress(i + 1, len(cfgs), MODEL_LABELS[c["model"]])
    return _board(config, results)


def peek_leaderboard(config: dict) -> dict | None:
    """The leaderboard if every algorithm's run is already available — never trains."""
    if check_features(config["dataset"], config["target"], config["features"]):
        return None
    results = [peek(c) for c in leaderboard_configs(config)]
    return _board(config, results) if all(results) else None


# ---- ablation: what does feature engineering add?
FEATURE_SETS = {"raw": "Raw only", "both": "Raw + engineered", "engineered": "Engineered only"}


def ablation_configs(config: dict) -> dict[str, dict]:
    """Three feature sets from the catalog (non-leaking, non-constant in the selected rows), scored with the
    current model, parameters and CV on the same rows (`require` = the union). PI-derived columns are left out."""
    ds, target = config["dataset"], config["target"]
    df = D.load(ds)
    for col, val in (config.get("rows") or {}).items():
        df = df[df[col] == val]
    df = df[df[target].notna()]
    usable = [f for f in D.feature_catalog(ds) if f["kind"] in ("raw", "engineered") and f["name"] != target
              and f["name"] in df.columns and not check_features(ds, target, [f["name"]])
              and df[f["name"]].nunique(dropna=True) > 1]
    raw = [f["name"] for f in usable if f["kind"] == "raw"]
    eng = [f["name"] for f in usable if f["kind"] == "engineered"]
    sets = {"raw": raw, "both": raw + eng, "engineered": eng}
    return {k: dict(config, features=v, require=raw + eng, leakage_demo=False) for k, v in sets.items() if v}


def _abl(config: dict, results: dict[str, dict]) -> dict:
    task = target_type(config["dataset"], config["target"])
    sk = _score_key(task)
    any_r = next(iter(results.values()))
    return {"task": task, "score": sk.replace("_", " ").replace("r2", "R²"), "model": MODEL_LABELS[config["model"]],
            "bars": [{"set": k, "label": FEATURE_SETS[k], "features": r["config"]["features"],
                      "score": r["metrics"][sk], "n_rows": r["n_rows"]} for k, r in results.items()],
            "baseline": {"label": "Baseline", "score": any_r["baseline"][sk]}, "n_rows": any_r["n_rows"]}


def ablation(config: dict, progress=None) -> dict:
    cfgs, results = ablation_configs(config), {}
    for i, (k, c) in enumerate(cfgs.items()):
        results[k] = run(c)
        if progress:
            progress(i + 1, len(cfgs), FEATURE_SETS[k])
    return _abl(config, results)


def peek_ablation(config: dict) -> dict | None:
    cfgs = ablation_configs(config)
    results = {k: peek(c) for k, c in cfgs.items()}
    return _abl(config, results) if results and all(results.values()) else None


# ---- what the model learned: partial dependence
PD_GRID, PD_GRID_2D, PD_TOP = 20, 14, 4
_PRESSURE = ("pressure_kPa", "ln_pressure")


def _pd_grid(x: pd.Series, n: int) -> np.ndarray:
    u = np.unique(x.dropna().astype(float))
    return u if len(u) <= 8 else np.linspace(u.min(), u.max(), n)


def _pd_predict(model, X: pd.DataFrame, task: str, pos, variants: list[dict]) -> list[float]:
    """Mean prediction over all training rows for each variant ({feature: value} overrides), in ONE predict call."""
    blocks = [X.assign(**v) for v in variants]
    big = pd.concat(blocks, ignore_index=True)
    if task == "classification":
        idx = list(model.classes_).index(pos)
        p = model.predict_proba(big)[:, idx]
    else:
        p = model.predict(big)
    n = len(X)
    return [float(p[i * n:(i + 1) * n].mean()) for i in range(len(variants))]


def _pd_pairs(features: list[str]) -> list[tuple[str, str]]:
    pairs = []
    if "o2_frac" in features:
        if "x_co2" in features:
            pairs.append(("o2_frac", "x_co2"))
        p = next((f for f in _PRESSURE if f in features), None)
        if p:
            pairs.append(("o2_frac", p))
    return pairs


def _partial_dependence(config: dict) -> dict:
    _refuse_leaks(config)
    model, X, y, meta, task = _fitted(_key(config))
    ds, target = config["dataset"], config["target"]
    pos = positive_class(ds, target)
    cats = set(_categorical(ds, config["features"]))
    order = [d["feature"] for d in run(config)["permutation_importance"]]
    nums = [f for f in order if f not in cats and X[f].nunique() > 1]
    curves = []
    for f in nums[:PD_TOP]:
        grid = _pd_grid(X[f], PD_GRID)
        ys = _pd_predict(model, X, task, pos, [{f: g} for g in grid])
        curves.append({"feature": f, "x": np.round(grid, 5).tolist(), "y": np.round(ys, 4).tolist(),
                       "rug": np.round(X[f].astype(float).to_numpy(), 5).tolist(),
                       "display": "exp" if f == "ln_pressure" else None})
    maps = []
    for a, b in _pd_pairs(config["features"]):
        ga, gb = _pd_grid(X[a], PD_GRID_2D), _pd_grid(X[b], PD_GRID_2D)
        z = _pd_predict(model, X, task, pos, [{a: va, b: vb} for vb in gb for va in ga])
        maps.append({"x": a, "y": b, "xs": np.round(ga, 5).tolist(), "ys": np.round(gb, 5).tolist(),
                     "z": np.round(np.array(z).reshape(len(gb), len(ga)), 4).tolist(),
                     "points": np.round(X[[a, b]].astype(float).to_numpy(), 5).tolist(),
                     "display_y": "exp" if b == "ln_pressure" else None})
    unit = D.DATASETS[ds]["targets"][target].get("unit", "")
    return {"task": task, "of": f"P({pos})" if pos else f"predicted {target} ({unit})", "positive": pos,
            "curves": curves, "maps": maps, "n_rows": int(len(X)),
            "skipped_categorical": [f for f in order if f in cats],
            "pairs_missing": [] if len(maps) == 2 else
            [p for p in ("O₂ × CO₂", "O₂ × pressure")
             if p not in {f"O₂ × {'CO₂' if m['y'] == 'x_co2' else 'pressure'}" for m in maps}]}


def partial_dependence(config: dict) -> dict:
    """1-D curves for the 4 most important numeric features (with the training values as a rug) and 2-D maps
    for O2 × CO2 and O2 × pressure when those features are selected. Model trained on all rows of the config."""
    return _extra("pd", config, _partial_dependence)


def peek_partial_dependence(config: dict) -> dict | None:
    return _peek_extra("pd", config)


# ---- calibration + error analysis (out-of-fold predictions)
CAL_BINS = 5  # same binning as analysis.registry's reliability chart, so shipped models match the Models hub


def _diagnostics(config: dict) -> dict:
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    _refuse_leaks(config)
    ds, target = config["dataset"], config["target"]
    task = target_type(ds, target)
    X, y, meta, _, _ = _frame(config)
    folds, seed = int(config["cv"]["folds"]), int(config["cv"]["seed"])
    pipe = _pipeline(config, task)
    meta_cols = [c for c in ("test_id", "date_iso", "flags") if c in meta.columns]

    def conditions(i):
        return {f: (X.loc[i, f] if isinstance(X.loc[i, f], str) else
                    (None if pd.isna(X.loc[i, f]) else round(float(X.loc[i, f]), 4))) for f in config["features"]}

    if task == "regression":
        pred = cross_val_predict(pipe, X, y, cv=KFold(folds, shuffle=True, random_state=seed))
        res = (y.to_numpy(dtype=float) - pred)
        worst = np.argsort(-np.abs(res))[:10]
        unit = D.DATASETS[ds]["targets"][target].get("unit", "")
        return {"task": task, "calibration": None, "unit": unit, "n_rows": int(len(X)),
                "errors": [{**{c: meta.loc[i, c] for c in meta_cols}, **conditions(i),
                            "actual": round(float(y.iloc[i]), 4), "predicted": round(float(pred[i]), 4),
                            "error": round(float(res[i]), 4)} for i in worst]}

    yl = _labels(ds, target, y)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, yl, cv=cv, method="predict_proba")
    classes = sorted(yl.unique(), key=str)  # cross_val_predict orders columns like the estimator: sorted labels
    pos = positive_class(ds, target)
    p = proba[:, classes.index(pos)]
    obs = (yl == pos).to_numpy(dtype=float)
    edges = np.linspace(0, 1, CAL_BINS + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum():
            bins.append({"p_predicted": round(float(p[sel].mean()), 3), "p_observed": round(float(obs[sel].mean()), 3),
                         "n": int(sel.sum()), "range": [round(float(lo), 2), round(float(hi), 2)]})
    brier = float(np.mean((p - obs) ** 2))
    prev = float(obs.mean())
    brier_base = prev * (1 - prev)
    ece = float(sum(b["n"] * abs(b["p_predicted"] - b["p_observed"]) for b in bins) / len(p))
    if ece <= 0.05:
        verdict = (f"Yes — well calibrated: out of fold, predicted and observed rates differ by "
                   f"{ece * 100:.0f} percentage points on average, so P({pos}) can be read as a percentage.")
    elif ece <= 0.10:
        verdict = (f"Roughly — predicted and observed rates differ by {ece * 100:.0f} points on average: read "
                   f"P({pos}) as an approximate percentage (±10 points), not an exact one.")
    else:
        verdict = (f"No — predicted and observed rates differ by {ece * 100:.0f} points on average: read P({pos}) "
                   "as a ranking (higher = more likely), not as a percentage.")
    k = {c: i for i, c in enumerate(classes)}
    p_true = proba[np.arange(len(yl)), [k[c] for c in yl]]
    worst = np.argsort(p_true)[:10]
    errors = [{**{c: meta.loc[i, c] for c in meta_cols}, **conditions(i), "actual": yl.iloc[i],
               "predicted": classes[int(np.argmax(proba[i]))], "p_actual": round(float(p_true[i]), 3),
               "p_predicted": round(float(proba[i].max()), 3)} for i in worst]
    return {"task": task, "positive": pos, "n_rows": int(len(X)), "folds": folds,
            "calibration": {"bins": bins, "brier": round(brier, 4), "brier_baseline": round(brier_base, 4),
                            "brier_skill": round(1 - brier / brier_base, 3) if brier_base else None,
                            "ece": round(ece, 4), "verdict": verdict, "of": f"P({pos})",
                            "baseline": f"always predicting the base rate {prev:.0%}"},
            "errors": errors, "n_misclassified": int(sum(classes[int(np.argmax(proba[i]))] != yl.iloc[i]
                                                         for i in range(len(yl))))}


def diagnostics(config: dict) -> dict:
    """Out-of-fold predictions ({folds}-fold, shuffled, the config's seed): reliability bins, Brier score
    vs the base-rate baseline, a one-line verdict, and the 10 worst-predicted tests."""
    return _extra("diagnostics", config, _diagnostics)


def peek_diagnostics(config: dict) -> dict | None:
    return _peek_extra("diagnostics", config)


# ------------------------------------------------------------------ shareable configs (#59)
_PARAM_SPEC = {"n_estimators": (int, 10, 500), "min_samples_leaf": (int, 1, 20), "max_depth": (int, 1, 50),
               "learning_rate": (float, 0.001, 1.0), "C": (float, 1e-4, 1e4), "n_neighbors": (int, 1, 50),
               "alpha": (float, 1e-4, 1e4)}
_MODEL_PARAMS = {"logreg": {"C"}, "random_forest": {"n_estimators", "min_samples_leaf", "max_depth"},
                 "gradient_boosting": {"n_estimators", "learning_rate", "max_depth"}, "knn": {"n_neighbors"},
                 "ridge": {"alpha"}}
MAX_TOKEN = 3000


def encode_config(config: dict) -> str:
    import base64
    keep = {k: config[k] for k in ("dataset", "target", "features", "scale", "impute", "exclude_flagged", "model",
                                   "params", "class_weight", "cv", "leakage_demo", "rows") if config.get(k) is not None}
    raw = json.dumps(keep, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_config(token: str) -> dict:
    """A ?config= value -> a validated config. The token is untrusted input: only known keys and values are
    taken, numbers are bounded, anything else raises ValueError with a readable reason."""
    import base64
    if not token or len(token) > MAX_TOKEN:
        raise ValueError("the config link is empty or too long")
    try:
        d = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode())
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError("the config link is not valid (could not decode it)") from e
    if not isinstance(d, dict):
        raise ValueError("the config link is not valid")
    ds = d.get("dataset")
    if ds not in D.DATASETS:
        raise ValueError(f"unknown dataset {ds!r}")
    target = d.get("target")
    if target not in D.DATASETS[ds]["targets"]:
        raise ValueError(f"unknown target {target!r} for {ds}")
    task = target_type(ds, target)
    cfg = default_config(ds, target)
    names = {f["name"] for f in D.feature_catalog(ds)}
    feats = d.get("features")
    if not isinstance(feats, list) or not feats or not all(isinstance(f, str) and f in names for f in feats):
        raise ValueError("the link's feature list is empty or names features that are not in the catalog")
    cfg["features"] = list(dict.fromkeys(feats))
    model = d.get("model")
    if model not in (CLASSIFIERS if task == "classification" else REGRESSORS):
        raise ValueError(f"model {model!r} is not available for a {task} target")
    cfg["model"] = model
    params = {}
    for k, v in (d.get("params") or {}).items():
        if k not in _MODEL_PARAMS[model]:
            continue
        if v is None and k == "max_depth":
            params[k] = None
            continue
        typ, lo, hi = _PARAM_SPEC[k]
        try:
            v = typ(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"parameter {k} is not a number") from e
        if not lo <= v <= hi:
            raise ValueError(f"parameter {k} = {v} is outside {lo}–{hi}")
        params[k] = v
    cfg["params"] = {**DEFAULT_PARAMS[model], **params}
    cv = d.get("cv") or {}
    try:
        cfg["cv"] = {"folds": int(cv.get("folds", 5)), "repeats": int(cv.get("repeats", 3)), "seed": int(cv.get("seed", 0))}
    except (TypeError, ValueError) as e:
        raise ValueError("the link's cross-validation settings are not numbers") from e
    if not (3 <= cfg["cv"]["folds"] <= 10 and 1 <= cfg["cv"]["repeats"] <= 5 and 0 <= cfg["cv"]["seed"] <= 9999):
        raise ValueError("cross-validation settings out of range (folds 3–10, repeats 1–5, seed 0–9999)")
    cfg["scale"] = bool(d.get("scale", True))
    cfg["impute"] = d.get("impute") if d.get("impute") in ("drop", "median") else "drop"
    cfg["exclude_flagged"] = bool(d.get("exclude_flagged", False))
    cfg["leakage_demo"] = bool(d.get("leakage_demo", False))
    cfg["class_weight"] = ("balanced" if d.get("class_weight") == "balanced" else None) \
        if task == "classification" and model != "knn" else None
    rows = {}
    for col, val in (d.get("rows") or {}).items():
        if val not in ROW_FILTERS.get(ds, {}).get(col, []):
            raise ValueError(f"row filter {col} = {val!r} is not available for {ds}")
        rows[col] = val
    if rows:
        cfg["rows"] = rows
    return cfg


def share_query(model_id: str) -> dict:
    """Query parameters that open the ML Lab on a shipped model, e.g. st.page_link(..., query_params=...)."""
    return {"config": encode_config(shipped_config(model_id))}


def run_record(config: dict, res: dict) -> dict:
    """The run as a downloadable JSON: config, data fingerprint, metrics — enough to reproduce it."""
    import sklearn
    from datetime import datetime, timezone
    keep = ("task", "n_rows", "n_dropped", "metrics", "baseline", "cv", "recall", "class_counts",
            "permutation_importance", "warnings", "leakage_demo")
    return {"tool": "IGNIS ML Lab (classical ML, scikit-learn; no language model)",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": config, "config_token": encode_config(config), "shipped_model": is_shipped(config),
            "data": {"dataset": config["dataset"], "source": D.DATASETS[config["dataset"]]["source"],
                     "fingerprint_sha256_16": data_fingerprint(config["dataset"]),
                     "built_from": "processed/tests.csv via analysis.datasets"},
            "sklearn": sklearn.__version__, "result": {k: res[k] for k in keep if k in res}}


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


def start_configs() -> list[dict]:
    """What the page opens on for each dataset × target (shipped model where there is one)."""
    return [start_config(ds, t) for ds in D.DATASETS for t in D.DATASETS[ds]["targets"]]


def precomputed_configs() -> list[dict]:
    seen, out = set(), []
    for c in start_configs() + [default_config(ds, t) for ds in D.DATASETS for t in D.DATASETS[ds]["targets"]]:
        if _key(c) not in seen:
            seen.add(_key(c))
            out.append(c)
    return out


def precompute() -> None:
    """Compute and store, for every start + default config: the run, learning curve, diagnostics and partial
    dependence; for every start config also the leaderboard and ablation runs. Incremental: entries whose key
    (config + data fingerprint + sklearn version) is unchanged are kept, only missing ones are trained."""
    old, store = _disk(), {}

    def keep(kind, cfg, fn):
        """runs / learning curves are reused while their key matches; diagnostics and partial dependence are
        cheap (one CV pass / one fit) and their output format may change with the code, so they are rebuilt."""
        k = _disk_key(kind, cfg)
        store[k] = (old.get(k) if kind in ("run", "learning_curve") else None) or fn(_key(cfg))

    def extra(fn):
        return lambda key: json.dumps(fn(json.loads(key)), default=float)

    starts = {_key(c) for c in start_configs()}
    for cfg in precomputed_configs():
        t = time.perf_counter()
        keep("run", cfg, _run_cached)
        keep("learning_curve", cfg, _learning_curve_cached)
        keep("diagnostics", cfg, extra(_diagnostics))
        keep("pd", cfg, extra(_partial_dependence))
        if _key(cfg) in starts:
            for c in leaderboard_configs(cfg) + list(ablation_configs(cfg).values()):
                keep("run", c, _run_cached)
        r = json.loads(store[_disk_key("run", cfg)])
        k = "accuracy" if r["task"] == "classification" else "r2"
        print(f"{cfg['dataset']:4s} {cfg['target']:24s} {cfg['model']:14s} n={r['n_rows']:3d}  "
              f"{k} {r['metrics'][k]['mean']:.3f} ± {r['metrics'][k]['sd']:.3f}  baseline {r['baseline'][k]['mean']:.3f}"
              f"  ({time.perf_counter() - t:.1f} s)", flush=True)
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
        for cfg in start_configs():
            r = run(cfg)
            print(cfg["dataset"], cfg["target"], {k: v["mean"] for k, v in r["metrics"].items()},
                  "from cache" if r.get("from_cache") else f"{r['seconds']} s")
