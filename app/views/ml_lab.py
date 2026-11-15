"""ML Lab (issues #47, #59; owned by pc2): data → features → train → compare → inspect → predict.

Strictly classical, cross-validated machine learning on the PSI test tables (scikit-learn) — no LLM.
Backend: ai/mllab.py (pure functions). Data: analysis/datasets.py (main's data layer). The page opens on the
shipped model of each dataset (analysis.registry, the Models hub) and accepts ?config=<token> deep links.
"""
from __future__ import annotations

import json
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai import mllab as M
from analysis import datasets as D
from app import ml_ui, ui

# Colours come from the UI kit (app/ui.py): flame orange is the UI accent, never a data colour. Classes use
# the kit's data colours in fixed order (FLEX outcomes: ui.OUTCOME_COLOURS, as on Predict extinction);
# magnitudes one blue ramp; correlation a blue <-> red diverging pair with a theme-neutral midpoint.
CAT = [ui.BLUE, ui.MAGENTA, ui.AQUA]
SEQ = ui.seq_blue()
DIV = [[0.0, ui.BLUE], [0.5, "#383835" if ui.is_dark() else "#f0efec"], [1.0, ui.RED]]
KIND_LABEL = {"raw": "Raw — set before the test", "engineered": "Engineered — computed from raw columns",
              "pi_derived": "Published by the investigators", "post_test": "Measured during/after the test"}
ROW_LABEL = {"normal": "Normal flames only (shipped models)", "all": "All flames", "inverse": "Inverse flames only"}
PLOT_CFG = {"displayModeBar": False}
# widget keys per model parameter, and the widgets' allowed values (a shared link is snapped onto them)
PARAM_KEYS = {"random_forest": {"n_estimators": "ml_rf_trees", "min_samples_leaf": "ml_rf_leaf", "max_depth": "ml_rf_depth"},
              "gradient_boosting": {"n_estimators": "ml_gb_n", "learning_rate": "ml_gb_lr", "max_depth": "ml_gb_depth"},
              "logreg": {"C": "ml_lr_c"}, "knn": {"n_neighbors": "ml_knn_k"}, "ridge": {"alpha": "ml_ridge_a"}}
GB_LR = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
LR_C = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
RIDGE_A = [0.01, 0.1, 1.0, 10.0, 100.0]
BOUNDS = {"ml_rf_trees": (20, 300), "ml_rf_leaf": (1, 10), "ml_rf_depth": (0, 20), "ml_gb_n": (20, 300),
          "ml_gb_depth": (1, 6), "ml_knn_k": (1, 25), "ml_folds": (3, 10), "ml_repeats": (1, 5), "ml_seed": (0, 9999)}
OPTIONS = {"ml_gb_lr": GB_LR, "ml_lr_c": LR_C, "ml_ridge_a": RIDGE_A}
WIDGET_DEFAULTS = {"ml_rf_trees": 50, "ml_rf_leaf": 2, "ml_rf_depth": 0, "ml_gb_n": 150, "ml_gb_lr": 0.1, "ml_gb_depth": 3,
                   "ml_lr_c": 1.0, "ml_knn_k": 7, "ml_ridge_a": 1.0, "ml_scale": True, "ml_impute": "drop",
                   "ml_exclude": False, "ml_leak_demo": False, "ml_balanced": True, "ml_folds": 5, "ml_repeats": 3,
                   "ml_seed": 0}
ss = st.session_state
ss.setdefault("ml_history", [])


def _metric(where, *args, **kwargs):
    """st.metric in the kit's bordered-card look, keeping the 'vs baseline' delta ui.metric_card lacks."""
    where.metric(*args, border=True, **kwargs)


def _layout(fig, h=300, **kw):
    t = ui.chart_theme()
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), legend_title_text="",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=t["label_fg"]), **kw)
    fig.update_xaxes(gridcolor=ui.GRID)
    fig.update_yaxes(gridcolor=ui.GRID)
    return fig


def _snap(key, v):
    if key in OPTIONS:
        return min(OPTIONS[key], key=lambda o: abs(o - float(v)))
    if key in BOUNDS:
        lo, hi = BOUNDS[key]
        return int(min(max(int(v), lo), hi))
    return v


# ---------------------------------------------------------------------- widget state
def _apply(cfg: dict) -> None:
    """Load a config into every widget (before they are drawn)."""
    ds, t = cfg["dataset"], cfg["target"]
    ss["ml_dataset"], ss[f"ml_target_{ds}"] = ds, t
    for f in D.feature_catalog(ds):
        ss[f"ml_feat_{ds}_{t}_{f['name']}"] = f["name"] in cfg["features"]
    if ds in M.ROW_FILTERS:
        ss[f"ml_rows_{ds}"] = (cfg.get("rows") or {}).get("flame_type", "all")
    ss.update({"ml_scale": bool(cfg["scale"]), "ml_impute": cfg["impute"], "ml_exclude": bool(cfg["exclude_flagged"]),
               "ml_leak_demo": bool(cfg.get("leakage_demo"))})
    task = M.target_type(ds, t)
    ss[f"ml_model_{task}"] = cfg["model"]
    if task == "classification" and cfg["model"] != "knn":
        ss["ml_balanced"] = cfg.get("class_weight") == "balanced"
    for p, key in PARAM_KEYS[cfg["model"]].items():
        v = (cfg.get("params") or {}).get(p, M.DEFAULT_PARAMS[cfg["model"]].get(p))
        ss[key] = _snap(key, 0 if v is None else v)
    for k in ("folds", "repeats", "seed"):
        ss[f"ml_{k}"] = _snap(f"ml_{k}", cfg["cv"][k])


def _on_dataset():
    ds = ss["ml_dataset"]
    _apply(M.start_config(ds, ss.get(f"ml_target_{ds}") or M.DEFAULT_TARGET[ds]))


def _on_target():
    ds = ss["ml_dataset"]
    _apply(M.start_config(ds, ss[f"ml_target_{ds}"]))


def _reset_to_start():
    """on_click: the shipped model of the current dataset × target (or its default config)."""
    ds = ss["ml_dataset"]
    _apply(M.start_config(ds, ss[f"ml_target_{ds}"]))


for _k, _v in WIDGET_DEFAULTS.items():
    ss.setdefault(_k, _v)
if "ml_init" not in ss:
    ss["ml_init"] = True
    token = st.query_params.get("config")
    start = None
    if token:
        try:
            start = M.decode_config(token)
            ss["ml_link_note"] = ("ok", "Opened from a shared link — the configuration below is exactly the one "
                                        "in the link.")
        except ValueError as e:
            ss["ml_link_note"] = ("bad", f"The shared link could not be used ({e}); showing the shipped model instead.")
    _apply(start or M.start_config("flex"))
for _ds in D.DATASETS:
    ss.setdefault(f"ml_target_{_ds}", M.DEFAULT_TARGET[_ds])


# ---------------------------------------------------------------------- page
ui.page_header("ML Lab", "Machine learning on the PSI test tables, **step by step and in the open**: look at the data, "
               "choose and engineer features, train a model with cross-validation, compare algorithms, see what it "
               "learned and where it fails — then use it. It opens on the **shipped model** (the one the Models hub "
               "and the prediction pages use). Classical machine learning only (scikit-learn); **no language model "
               "on this page**.", ui.ICONS["ml_lab"])
if ss.get("ml_link_note"):
    kind, text = ss["ml_link_note"]
    ui.callout("note" if kind == "ok" else "warning", text)

# The page opens on the shipped model's results (#59 item 1); ?tab=compare etc. opens another tab (shared links).
TABS = {"explore": "1 · Explore the data", "features": "2 · Engineer features", "train": "3 · Train & evaluate",
        "compare": "4 · Compare", "learned": "5 · What it learned", "predict": "6 · Predict"}
tab_explore, tab_features, tab_train, tab_compare, tab_inspect, tab_predict = st.tabs(
    list(TABS.values()), default=TABS.get(st.query_params.get("tab", "train"), TABS["train"]))

# ====================================================================== 1. explore
with tab_explore:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        ds = st.radio("Dataset", list(D.DATASETS), key="ml_dataset", horizontal=True, on_change=_on_dataset,
                      format_func=lambda k: D.DATASETS[k]["title"].split(" — ")[0])
    spec = D.DATASETS[ds]
    with c2:
        targets = list(spec["targets"])
        target = st.selectbox("Target — what the model should predict", targets, key=f"ml_target_{ds}",
                              on_change=_on_target, format_func=lambda t: f"{t} ({spec['targets'][t]['type']})")
    tmeta = spec["targets"][target]
    task = tmeta["type"]
    # colour follows the class, never its rank: one fixed colour per class, in the dataset's class order
    CLASS_ORDER = [M.class_label(ds, target, c) for c in tmeta.get("classes", [])]
    CLASS_COLORS = {c: ui.OUTCOME_COLOURS.get(c, CAT[i % len(CAT)]) for i, c in enumerate(CLASS_ORDER)}
    ORDERS = {"class": CLASS_ORDER, "_y": CLASS_ORDER}
    src = spec["source"]
    st.markdown(f"**{spec['title']}** · data: PSI {ui.source_link(src['doi'], src['psi_id'] + ' · DOI ' + src['doi'])}")
    ui.callout("limit", f"**Scope:** {spec['scope']}")
    st.caption(f"Target `{target}`: {tmeta['note']}")

    df = D.load(ds)
    summ = D.summary(ds)
    cat = M.catalog(ds)
    numeric = [f["name"] for f in D.feature_catalog(ds) if f["unit"] != "category" and f["name"] in df.columns]
    m1, m2, m3, m4 = st.columns(4)
    ui.metric_card(m1, "Tests (rows)", summ["rows"])
    ui.metric_card(m2, f"Rows with a known `{target}`", summ["targets"][target]["n"])
    ui.metric_card(m3, "Missing values (all features)", sum(summ["missing"].values()))
    ui.metric_card(m4, "Rows with ETL corrections", summ["rows_with_corrections"])
    ui.provenance_label("computed", "Counts and charts below are computed from the PSI test tables "
                                    "(analysis.datasets, built from processed/tests.csv).")
    st.caption("Small data: a few hundred tests. Every score on this page is cross-validated and shown next to a "
               "baseline that knows nothing about the conditions.")

    y_lab = (df[target].map(lambda v: M.class_label(ds, target, v)) if task == "classification" else df[target])
    if task == "classification":
        bal = y_lab[df[target].notna()].value_counts().rename_axis("class").reset_index(name="tests")
        fig = px.bar(bal, x="tests", y="class", orientation="h", color="class", color_discrete_map=CLASS_COLORS,
                     category_orders=ORDERS, text="tests", title="Target balance")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(_layout(fig, 60 + 45 * len(bal), showlegend=False), width="stretch", config=PLOT_CFG)
    else:
        fig = px.histogram(df.dropna(subset=[target]), x=target, nbins=30, color_discrete_sequence=[CAT[0]],
                           title=f"Target distribution ({tmeta.get('unit', '')})")
        st.plotly_chart(_layout(fig, 260, bargap=0.05), width="stretch", config=PLOT_CFG)

    with st.expander("Each feature, split by the target", expanded=False):
        shown = [f for f in numeric if f != target]
        cols = st.columns(3)
        plot_df = df.assign(_y=y_lab).dropna(subset=["_y"])
        for i, f in enumerate(shown):
            leak = cat[f]["kind"] == "post_test" or target in cat[f].get("leak", [])
            if task == "classification":
                fg = px.histogram(plot_df, x=f, color="_y", barmode="overlay", opacity=0.75, nbins=20,
                                  color_discrete_map=CLASS_COLORS, category_orders=ORDERS)
            else:
                fg = px.scatter(plot_df, x=f, y="_y", opacity=0.6, color_discrete_sequence=[CAT[0]])
                fg.update_yaxes(title_text=target)
            title = f"{f}" + ("  ⚠ leaks the answer" if leak else "")
            cols[i % 3].plotly_chart(_layout(fg, 220, title=dict(text=title, font=dict(size=13)), showlegend=(i == 0)),
                                     width="stretch", config=PLOT_CFG, key=f"ml_hist_{ds}_{target}_{f}")

    cc1, cc2 = st.columns(2)
    with cc1:
        corr = M.correlations(ds, [f for f in numeric if cat[f]["kind"] != "post_test"])
        fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale=DIV, aspect="auto",
                        title="Correlation between features (|r| > 0.95 = redundant)")
        st.plotly_chart(_layout(fig, 430), width="stretch", config=PLOT_CFG)
    with cc2:
        sx = st.selectbox("Scatter: x", numeric, index=numeric.index("o2_frac") if "o2_frac" in numeric else 0,
                          key=f"ml_sx_{ds}")
        sy = st.selectbox("Scatter: y", numeric, index=numeric.index("pressure_kPa") if "pressure_kPa" in numeric else 1,
                          key=f"ml_sy_{ds}")
        sc = df.assign(_y=y_lab).dropna(subset=[sx, sy, "_y"])
        fig = px.scatter(sc, x=sx, y=sy, color="_y" if task == "classification" else None,
                         color_discrete_map=CLASS_COLORS, category_orders=ORDERS, hover_data=["test_id"], opacity=0.75,
                         title=f"{sy} vs {sx}, coloured by {target}" if task == "classification" else f"{sy} vs {sx}")
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color=ui.chart_theme()["neutral_line"])))
        st.plotly_chart(_layout(fig, 340), width="stretch", config=PLOT_CFG)
        if ds == "cfig" and "o2_frac" in (sx, sy):
            st.caption("⚠ `o2_frac` of **inverse** flames is the O₂ in the burner stream, not an atmosphere "
                       "(`o2_role = burner`).")

    ui.section("Feature catalog")
    ui.table(pd.DataFrame([{"feature": f["name"], "kind": f["kind"], "unit": f["unit"], "formula": f["formula"],
                            "why": f["why"],
                            "leaks the answer": ("yes — measured during/after the test" if f["kind"] == "post_test"
                                                 else ("for " + ", ".join(f["leak"]) if f.get("leak") else ""))}
                           for f in D.feature_catalog(ds)]), width="stretch")

# ====================================================================== 2. features
with tab_features:
    st.markdown(f"Choose the inputs for predicting **`{target}`**. The page opened on the shipped model's inputs "
                "(or, where nothing ships, the numeric features known *before* a test starts). Hover a name for its "
                "formula and physical reason.")
    defaults = set(D.default_features(ds, target))
    rows = {}
    if ds in M.ROW_FILTERS:
        choice = st.radio("Rows", ["normal", "all", "inverse"], key=f"ml_rows_{ds}", horizontal=True,
                          format_func=ROW_LABEL.get,
                          help="The shipped CFI-G models use normal flames only: in an inverse flame the O₂ value "
                               "is the burner stream, not an atmosphere.")
        if choice != "all":
            rows = {"flame_type": choice}
    features = []
    for kind in ("raw", "engineered", "pi_derived", "post_test"):
        group = [f for f in D.feature_catalog(ds) if f["kind"] == kind and f["name"] != target]
        if not group:
            continue
        st.markdown(f"**{KIND_LABEL[kind]}**")
        if kind == "post_test":
            st.caption("These are measured during or after the test, so they contain the answer. They are refused "
                       "unless you tick *Show me what leakage does* below.")
        cols = st.columns(3)
        for i, f in enumerate(group):
            key = f"ml_feat_{ds}_{target}_{f['name']}"
            ss.setdefault(key, f["name"] in defaults)
            with cols[i % 3]:
                label = f["name"] + (" (category)" if f["unit"] == "category" else "")
                if st.checkbox(label, key=key, help=f"{f['formula']} — {f['why']}"):
                    features.append(f["name"])
                if kind == "engineered":
                    st.caption(f"= {f['formula']} · {f['why']}")
                if ds == "cfig" and f["name"] == "o2_frac" and rows.get("flame_type") != "normal":
                    st.caption("Inverse flames: burner-stream O₂, not an atmosphere.")

    o1, o2, o3, o4 = st.columns(4)
    scale = o1.checkbox("Scale features (z-score)", key="ml_scale",
                        help="Needed for logistic regression and k-NN; trees don't care.")
    impute = o2.radio("Missing values", ["drop", "median"], key="ml_impute", horizontal=True,
                      help="drop: skip rows with a missing input · median: fill from the training folds only")
    exclude = o3.checkbox("Exclude rows with ETL corrections", key="ml_exclude",
                          help="Rows whose values main's ETL had to correct (flags F1–F10).")
    leak_demo = o4.checkbox("Show me what leakage does", key="ml_leak_demo",
                            help="Allows post-test features so you can see how a leaking model looks too good.")

    refused = M.check_features(ds, target, features)
    if refused and not leak_demo:
        ui.callout("alert", "**Refused — these features leak the answer:** " +
                   "; ".join(f"`{r['feature']}` ({r['reason']})" for r in refused) +
                   ". Untick them, or tick *Show me what leakage does* to see the effect.")
    elif refused:
        ui.callout("alert", "**LEAKAGE DEMO** — the selected features include information measured after the test. "
                            "Scores will look excellent and mean nothing.")

    if features:
        ui.section("Feature matrix preview (first rows)")
        prev = df[["test_id"] + features + [target]].dropna(subset=[target])
        ui.table(prev.head(8), width="stretch")
        mi = M.mutual_information(ds, target, features)
        if mi:
            mdf = pd.DataFrame(mi)
            fig = px.bar(mdf, x="mi", y="feature", orientation="h", color_discrete_sequence=[CAT[0]],
                         title=f"Signal of each feature on its own (mutual information with {target})",
                         labels={"mi": "mutual information (nats)", "feature": ""})
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(_layout(fig, 60 + 26 * len(mdf)), width="stretch", config=PLOT_CFG)
            st.caption("Univariate only: a feature with little signal alone can still help in combination.")
    else:
        ui.callout("warning", "Select at least one feature.")

# ====================================================================== 3. train
with tab_train:
    models = M.CLASSIFIERS if task == "classification" else M.REGRESSORS
    mkey = f"ml_model_{task}"
    if ss.get(mkey) not in models:
        ss[mkey] = "random_forest"
    a1, a2, a3 = st.columns([1.2, 1.4, 1.4])
    with a1:
        model = st.selectbox("Model", models, key=mkey, format_func=M.MODEL_LABELS.get)
        cw = None
        if task == "classification" and model != "knn":
            cw = "balanced" if st.checkbox("Balance the classes", key="ml_balanced",
                                           help="Weights rare outcomes up; otherwise the model can ignore them.") else None
    with a2:
        params = {}
        if model == "random_forest":
            params["n_estimators"] = st.slider("Trees", 20, 300, step=10, key="ml_rf_trees")
            params["min_samples_leaf"] = st.slider("Min. tests per leaf", 1, 10, key="ml_rf_leaf")
            depth = st.slider("Max depth (0 = unlimited)", 0, 20, key="ml_rf_depth")
            params["max_depth"] = depth or None
        elif model == "gradient_boosting":
            params["n_estimators"] = st.slider("Boosting rounds", 20, 300, step=10, key="ml_gb_n")
            params["learning_rate"] = st.select_slider("Learning rate", GB_LR, key="ml_gb_lr")
            params["max_depth"] = st.slider("Tree depth", 1, 6, key="ml_gb_depth")
        elif model == "logreg":
            params["C"] = st.select_slider("Regularisation C (small = simpler)", LR_C, key="ml_lr_c")
        elif model == "knn":
            params["n_neighbors"] = st.slider("Neighbours k", 1, 25, key="ml_knn_k")
        elif model == "ridge":
            params["alpha"] = st.select_slider("Ridge α (large = simpler)", RIDGE_A, key="ml_ridge_a")
    with a3:
        folds = st.slider("CV folds", 3, 10, key="ml_folds")
        repeats = st.slider("CV repeats", 1, 5, key="ml_repeats")
        seed = st.number_input("Random seed", 0, 9999, key="ml_seed")

    config = {"dataset": ds, "target": target, "features": features, "scale": bool(scale), "impute": impute,
              "exclude_flagged": bool(exclude), "model": model, "params": params, "class_weight": cw,
              "cv": {"folds": int(folds), "repeats": int(repeats), "seed": int(seed)}, "leakage_demo": bool(leak_demo)}
    if rows:
        config["rows"] = rows
    blocked = bool((refused and not leak_demo) or not features)
    ship = M.shipped_for(ds, target)
    shipped_id = M.is_shipped(config) if not blocked else None

    b1, b2, _ = st.columns([1, 1.3, 3])
    train = b1.button("Train & cross-validate", type="primary", key="ml_train", disabled=blocked,
                      icon=":material/play_arrow:")
    b2.button("Reset to the shipped model" if ship else "Reset to the defaults", key="ml_reset",
              on_click=_reset_to_start, icon=ui.ICONS["reset"],
              help=(f"Loads the configuration of the shipped model '{ship['spec']['title']}' (Models hub)." if ship
                    else "Loads the default configuration for this target (nothing ships for it)."))
    if refused and not leak_demo:
        ui.callout("alert", "Training is blocked: " + ", ".join(f"`{r['feature']}`" for r in refused) +
                            " would leak the answer (see tab 2).")

    if ship and not shipped_id:
        sm = ship["spec"] or {}
        st.caption(f"Not the shipped configuration. The shipped model for this target is *{sm.get('title', '')}* "
                   f"(accuracy {sm.get('metrics', {}).get('cv_accuracy', float('nan')):.3f}) — "
                   "*Reset to the shipped model* loads it.")

    res = None
    if train and not blocked:
        with st.spinner("Cross-validating… (each fold is trained from scratch)"):
            t0 = time.perf_counter()
            try:
                res = M.run(config)
            except (M.FeatureRefused, ValueError) as e:
                ui.callout("alert", f"Not trained: {e}")
        if res:
            ss["ml_history"].append({
                "#": len(ss["ml_history"]) + 1, "dataset": ds, "target": target, "model": M.MODEL_LABELS[model],
                "features": len(features), "feature list": ", ".join(features),
                **({"accuracy": res["metrics"]["accuracy"]["mean"], "balanced acc.": res["metrics"]["balanced_accuracy"]["mean"],
                    "baseline acc.": res["baseline"]["accuracy"]["mean"]} if task == "classification" else
                   {"R²": res["metrics"]["r2"]["mean"], "MAE": res["metrics"]["mae"]["mean"],
                    "baseline R²": res["baseline"]["r2"]["mean"]}),
                "rows": res["n_rows"], "CV": res["cv"]["scheme"], "leakage demo": "⚠ yes" if res["leakage_demo"] else "",
                "shipped": "yes" if shipped_id else "",
                "seconds": 0.0 if res.get("from_cache") else round(time.perf_counter() - t0, 1)})
    elif not blocked:
        res = M.peek(config)
        if res is None:
            ui.callout("note", "Settings changed — press **Train & cross-validate** to see how this configuration performs.")

    if res:
        if res["leakage_demo"]:
            ui.callout("alert", "**LEAKAGE DEMO** — these scores use information measured after the test "
                                "and are not a real model.")
        if shipped_id:
            sm = M.registry_spec(shipped_id) or {}
            met = sm.get("metrics", {})
            ui.callout("note", f"**This is the shipped model** — *{sm.get('title', shipped_id)}* on the Models hub "
                               f"(`{shipped_id}`). Registry: accuracy {met.get('cv_accuracy', float('nan')):.3f} vs "
                               f"{met.get('baseline_accuracy', float('nan')):.3f} baseline, n = {met.get('n_train')}. "
                               f"The Lab re-runs the same cross-validation below: "
                               f"{res['metrics']['accuracy']['mean']:.3f} vs {res['baseline']['accuracy']['mean']:.3f}.",
                       icon=":material/verified:")
        how = ("precomputed for this exact configuration and data (processed/ai/mllab_runs.json)"
               if res.get("from_cache") else f"trained just now in {res['seconds']} s")
        subset = f" · rows: {ROW_LABEL[rows['flame_type']].lower()}" if rows else ""
        ui.provenance_label("ml", f"{res['n_rows']} tests used ({res['n_dropped']} left out: missing target/inputs, "
                                  f"excluded or filtered){subset} · {res['cv']['scheme']} = {res['cv']['n_fits']} models · {how}")
        k1, k2, k3 = st.columns(3)
        if task == "classification":
            acc, bacc, base = res["metrics"]["accuracy"], res["metrics"]["balanced_accuracy"], res["baseline"]
            _metric(k1, "Accuracy (CV)", f"{acc['mean']:.3f} ± {acc['sd']:.3f}",
                    f"{acc['mean'] - base['accuracy']['mean']:+.3f} vs baseline")
            _metric(k2, "Balanced accuracy (CV)", f"{bacc['mean']:.3f} ± {bacc['sd']:.3f}",
                    f"{bacc['mean'] - base['balanced_accuracy']['mean']:+.3f} vs baseline")
            _metric(k3, "Baseline: " + base["name"], f"{base['accuracy']['mean']:.3f}",
                    help=f"Balanced accuracy of the baseline: {base['balanced_accuracy']['mean']:.3f} (chance).")
            if acc["mean"] <= base["accuracy"]["mean"]:
                ui.callout("warning", "This model does **not** beat always guessing the most common class on accuracy.")
        else:
            r2, mae, base = res["metrics"]["r2"], res["metrics"]["mae"], res["baseline"]
            unit = tmeta.get("unit", "")
            _metric(k1, "R² (CV)", f"{r2['mean']:.3f} ± {r2['sd']:.3f}", f"{r2['mean'] - base['r2']['mean']:+.3f} vs baseline")
            _metric(k2, f"Mean absolute error (CV, {unit})", f"{mae['mean']:.3f} ± {mae['sd']:.3f}",
                    f"{mae['mean'] - base['mae']['mean']:+.3f} vs baseline", delta_color="inverse")
            _metric(k3, "Baseline: " + base["name"], f"R² {base['r2']['mean']:.3f}", help=f"MAE {base['mae']['mean']:.3f} {unit}")
        for w in res["warnings"]:
            ui.callout("warning", w)

        # ---- share this run (#59 item 7)
        with st.expander("Share this run — link and JSON", icon=":material/share:"):
            token = M.encode_config(config)
            try:
                u = urlsplit(st.context.url or "")
                base_url = f"{u.scheme}://{u.netloc}{u.path}" if u.netloc else "ml_lab"
            except Exception:  # noqa: BLE001 - no browser context (tests)
                base_url = "ml_lab"
            st.caption("Anyone opening this link gets exactly this configuration (copy button on the right):")
            st.code(f"{base_url}?config={token}", language=None, wrap_lines=True)
            st.download_button("Download this run (JSON)", key="ml_download", icon=ui.ICONS["download"],
                               data=json.dumps(M.run_record(config, res), indent=1, default=str),
                               file_name=f"ignis_mllab_{ds}_{target}_{model}.json", mime="application/json",
                               help="Config, data fingerprint, scikit-learn version and all metrics — enough to "
                                    "reproduce the run.")

        g1, g2 = st.columns(2)
        if task == "classification":
            with g1:
                cm = res["confusion_matrix"]
                fig = px.imshow(np.array(cm["counts"]), x=cm["labels"], y=cm["labels"], text_auto=True,
                                color_continuous_scale=SEQ, labels=dict(x="predicted", y="actual", color="tests"),
                                title="Confusion matrix (each test predicted once, out of fold)")
                st.plotly_chart(_layout(fig, 340, coloraxis_showscale=False), width="stretch", config=PLOT_CFG)
            with g2:
                rec = pd.DataFrame({"class": list(res["recall"]), "recall": list(res["recall"].values())})
                fig = px.bar(rec, x="recall", y="class", orientation="h", color="class", text="recall",
                             color_discrete_map=CLASS_COLORS, category_orders=ORDERS, range_x=[0, 1],
                             title="Per-class recall: share of each outcome found")
                fig.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
                st.plotly_chart(_layout(fig, 340, showlegend=False), width="stretch", config=PLOT_CFG)
        imp = pd.DataFrame(res["permutation_importance"])
        with g1 if task == "regression" else st.container():
            fig = go.Figure(go.Bar(x=imp["mean"], y=imp["feature"], orientation="h", marker_color=CAT[0],
                                   error_x=dict(type="data", array=imp["sd"], color="#9a9a96", thickness=1.5)))
            fig.update_yaxes(categoryorder="total ascending")
            _layout(fig, 70 + 26 * len(imp), title=f"Permutation importance: drop in {res['permutation_scoring']} "
                                                   "when a feature is shuffled (held-out folds)",
                    xaxis_title=f"drop in {res['permutation_scoring']}")
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        with g2 if task == "regression" else st.container():
            try:
                lc = M.learning_curve(config)
                lcd = pd.DataFrame({"training tests": lc["train_sizes"] * 2,
                                    "score": lc["train_mean"] + lc["valid_mean"],
                                    "set": ["training"] * len(lc["train_sizes"]) + ["validation"] * len(lc["train_sizes"])})
                fig = px.line(lcd, x="training tests", y="score", color="set", markers=True, color_discrete_sequence=CAT,
                              title=f"Learning curve ({lc['scoring']}): would more tests help?")
                fig.update_traces(line=dict(width=2), marker=dict(size=8))
                st.plotly_chart(_layout(fig, 300), width="stretch", config=PLOT_CFG)
                st.caption("A gap between training and validation = the model memorises; a rising validation "
                           "line = more tests would still help.")
            except M.FeatureRefused:
                pass

    if ss["ml_history"]:
        ui.section("Run history (this session)")
        ui.table(pd.DataFrame(ss["ml_history"]), width="stretch")
        if st.button("Clear history", key="ml_clear", icon=ui.ICONS["clear"]):
            ss["ml_history"] = []
            st.rerun()


def _progress_cb(bar):
    def cb(i, n, label):
        bar.progress(i / n, text=f"{label} done ({i}/{n})")
    return cb


def _fmt(ms: dict) -> str:
    return f"{ms['mean']:.3f} ± {ms['sd']:.3f}"


# ====================================================================== 4. compare
with tab_compare:
    if blocked:
        ui.empty_state("Choose features without leakage first (tab 2).")
    elif leak_demo and refused:
        ui.callout("limit", "No comparisons in the leakage demo: every model would look excellent for the wrong reason.")
    else:
        # ---- leaderboard (#59 item 2)
        ui.section("Compare algorithms")
        st.markdown(f"Every algorithm on **the same {len(features)} features, rows and cross-validation folds** "
                    f"({folds}-fold × {repeats}), sorted by "
                    f"{'balanced accuracy — every outcome counts equally, however rare' if task == 'classification' else 'cross-validated R²'}. "
                    "Logistic regression, k-NN and ridge are always scaled; your current model keeps your settings, "
                    "the others use standard ones.")
        board = M.peek_leaderboard(config)
        if board is None:
            if st.button("Compare algorithms", key="ml_board", type="primary", icon=":material/leaderboard:"):
                bar = st.progress(0.0, text="Cross-validating each algorithm…")
                board = M.leaderboard(config, progress=_progress_cb(bar))
                bar.empty()
            else:
                st.caption("Not computed yet for these settings: it trains every algorithm with the CV above "
                           "(forests take longest — a progress bar shows each model).")
        if board:
            sk = "balanced_accuracy" if task == "classification" else "r2"
            rows_tbl = [{"#": "—", "Model": board["baseline"]["label"], board["score"]: _fmt(board["baseline"]["score"]),
                         **({"Accuracy": _fmt(board["baseline"]["accuracy"])} if task == "classification"
                            else {"MAE": _fmt(board["baseline"]["mae"])}),
                         "Beats baseline": "", "CV time (s)": None, "Note": "knows nothing about the conditions"}]
            for i, r in enumerate(board["rows"], start=1):
                notes = []
                if r["shipped"]:
                    notes.append("ships (Models hub)")
                if r["is_current"]:
                    notes.append("your current model")
                if r["model"] == board["pick"]:
                    notes.append("selection rule's pick")
                rows_tbl.append({"#": str(i), "Model": r["label"], board["score"]: _fmt(r["score"]),
                                 **({"Accuracy": _fmt(r["accuracy"])} if task == "classification" else {"MAE": _fmt(r["mae"])}),
                                 "Beats baseline": "yes" if r["beats_baseline"] else "no",
                                 "CV time (s)": r["seconds"], "Note": " · ".join(notes)})
            ui.table(pd.DataFrame(rows_tbl), width="stretch", key="ml_board_table")
            # bars with ±1 SD, baseline first (muted)
            names = [board["baseline"]["label"]] + [r["label"] for r in board["rows"]]
            vals = [board["baseline"]["score"]["mean"]] + [r["score"]["mean"] for r in board["rows"]]
            sds = [board["baseline"]["score"]["sd"]] + [r["score"]["sd"] for r in board["rows"]]
            colours = [ui.chart_theme()["muted"]] + [ui.BLUE if (r["shipped"] or r["is_current"]) else "#7fb0ea"
                                                     for r in board["rows"]]
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colours,
                                   error_x=dict(type="data", array=sds, color="#9a9a96", thickness=1.5),
                                   text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            _layout(fig, 60 + 42 * len(names), yaxis=dict(autorange="reversed"),
                    xaxis_title=f"{board['score']} (cross-validated, ±1 SD over folds)", showlegend=False)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="ml_board_chart")
            cached = all(r["from_cache"] for r in board["rows"])
            ui.provenance_label("ml", f"{board['n_rows']} tests · {board['cv']} · "
                                      + ("precomputed (processed/ai/mllab_runs.json); CV time = when it was computed"
                                         if cached else "trained in this session"))
            shipped_row = next((r for r in board["rows"] if r["shipped"]), None)
            pick = next(r for r in board["rows"] if r["model"] == board["pick"])
            why = [f"**Selection rule:** a model {board['rule']}."]
            if shipped_row:
                sm = M.registry_spec(shipped_row["shipped"]) or {}
                why.append(f"**Ships:** {shipped_row['label']} — *{sm.get('title', '')}* on the Models hub.")
                if pick is shipped_row:
                    why.append("It is also the rule's pick here.")
                else:
                    gap = pick["score"]["mean"] - shipped_row["score"]["mean"]
                    noise = max(pick["score"]["sd"], shipped_row["score"]["sd"])
                    why.append(f"By the same rule {pick['label']} scores {gap:+.3f} higher here — "
                               + ("within one standard deviation of the folds, i.e. inside the CV noise, so there is "
                                  "no evidence it is better." if abs(gap) <= noise else
                                  "more than one standard deviation: a candidate for the next model version."))
                    if shipped_row["model"] == "random_forest" and ds == "flex":
                        why.append("(analysis/flex_classifier.py compared logistic regression and random forest only.)")
            st.markdown("  \n".join(why))

        # ---- feature-set ablation (#59 item 3)
        ui.section("What does feature engineering add?")
        st.markdown(f"The same model ({M.MODEL_LABELS[model]}, your settings) on three feature sets from the "
                    "catalog — **raw columns only**, **raw + engineered**, **engineered only** — scored on the same "
                    "rows and folds, next to the baseline. Leaking, constant and investigator-computed columns are "
                    "left out.")
        abl = M.peek_ablation(config)
        if abl is None:
            if st.button("Compare feature sets", key="ml_ablation", icon=":material/stacked_bar_chart:"):
                bar = st.progress(0.0, text="Cross-validating each feature set…")
                abl = M.ablation(config, progress=_progress_cb(bar))
                bar.empty()
            else:
                st.caption("Not computed yet for these settings (three cross-validations).")
        if abl:
            names = [abl["baseline"]["label"]] + [b["label"] for b in abl["bars"]]
            vals = [abl["baseline"]["score"]["mean"]] + [b["score"]["mean"] for b in abl["bars"]]
            sds = [abl["baseline"]["score"]["sd"]] + [b["score"]["sd"] for b in abl["bars"]]
            fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[ui.chart_theme()["muted"], ui.BLUE, ui.MAGENTA, ui.AQUA][:len(names)],
                                   error_y=dict(type="data", array=sds, color="#9a9a96", thickness=1.5),
                                   text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
            _layout(fig, 320, yaxis_title=f"{abl['score']} (CV, ±1 SD)", showlegend=False)
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="ml_ablation_chart")
            by = {b["set"]: b["score"]["mean"] for b in abl["bars"]}
            if "raw" in by and "both" in by:
                d = by["both"] - by["raw"]
                st.markdown(f"Adding the engineered features changes {abl['score']} by **{d:+.3f}** "
                            f"({abl['n_rows']} tests). " + ("Small next to the fold-to-fold spread: on this data the "
                            "engineered features mostly re-express what the raw ones already carry."
                            if abs(d) < max(b["score"]["sd"] for b in abl["bars"]) else
                            "Larger than the fold-to-fold spread, so the engineering carries real signal."))
            with st.expander("Features in each set"):
                for b in abl["bars"]:
                    st.markdown(f"**{b['label']}** ({len(b['features'])}): " + ", ".join(f"`{f}`" for f in b["features"]))

# ====================================================================== 5. what it learned
with tab_inspect:
    res_now = M.peek(config) if not blocked else None
    if blocked:
        ui.empty_state("Choose features without leakage first (tab 2).")
    elif res_now is None:
        ui.empty_state("Train this configuration first (tab 3) — the inspection uses its trained model.")
    elif res_now["leakage_demo"]:
        ui.callout("limit", "Nothing to inspect in the leakage demo: the model learned the answer, not the physics.")
    else:
        pdres, diag = M.peek_partial_dependence(config), M.peek_diagnostics(config)
        if pdres is None or diag is None:
            if st.button("Inspect this model", key="ml_inspect", type="primary", icon=":material/search_insights:"):
                with st.spinner("Partial dependence and out-of-fold predictions…"):
                    pdres, diag = M.partial_dependence(config), M.diagnostics(config)
            else:
                st.caption("Not computed yet for these settings: one model on all rows for the curves, one "
                           f"{folds}-fold pass for calibration and errors.")
        if pdres:
            # ---- partial dependence (#59 item 4)
            ui.section("What the model learned")
            pct = pdres["task"] == "classification"
            st.markdown(f"**Partial dependence:** {pdres['of']} as one input moves across its tested range while "
                        "every test keeps its other real values, averaged over all tests. The ticks at the bottom "
                        "are the real tests — where they are sparse, the curve is the model guessing.")

            def _disp(f, xs):
                return ([float(np.exp(x)) for x in xs], "pressure (kPa, log scale in the model)") if f == "ln_pressure" \
                    else (xs, f)
            cols = st.columns(2)
            for i, c in enumerate(pdres["curves"]):
                xs, xlab = _disp(c["feature"], c["x"])
                rug, _ = _disp(c["feature"], c["rug"])
                ys = [y * 100 for y in c["y"]] if pct else c["y"]
                lo = 0 if pct else min(ys)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=ui.BLUE, width=2.5),
                                         marker=dict(size=5), hovertemplate=f"{xlab} %{{x:.3g}} → %{{y:.1f}}<extra></extra>"))
                fig.add_trace(go.Scatter(x=rug, y=[lo] * len(rug), mode="markers", hoverinfo="skip",
                                         marker=dict(symbol="line-ns-open", size=12, color=ui.MAGENTA, opacity=0.35)))
                _layout(fig, 250, showlegend=False, title=dict(text=f"{pdres['of']} vs {xlab}", font=dict(size=13)),
                        xaxis_title=xlab, yaxis_title=("%" if pct else pdres["of"]),
                        yaxis=dict(range=[0, 100]) if pct else {})
                cols[i % 2].plotly_chart(fig, width="stretch", config=PLOT_CFG, key=f"ml_pd_{i}")
            if pdres["skipped_categorical"]:
                st.caption("No curve for category inputs: " + ", ".join(f"`{f}`" for f in pdres["skipped_categorical"]) + ".")
            if pdres["maps"]:
                st.markdown("**Two inputs at once** — dots are the real tests; colour away from the dots is extrapolation.")
                mcols = st.columns(len(pdres["maps"]))
                for i, mp in enumerate(pdres["maps"]):
                    ys_, ylab = _disp(mp["y"], mp["ys"])
                    z = (np.array(mp["z"]) * 100) if pct else np.array(mp["z"])
                    fig = go.Figure(go.Heatmap(x=mp["xs"], y=ys_, z=z, colorscale=SEQ, zmin=0 if pct else None,
                                               zmax=100 if pct else None,
                                               colorbar=dict(title="%" if pct else "", thickness=12),
                                               hovertemplate=f"{mp['x']} %{{x:.3g}}, {ylab} %{{y:.3g}} → %{{z:.0f}}<extra></extra>"))
                    pts = np.array(mp["points"])
                    py = np.exp(pts[:, 1]) if mp["y"] == "ln_pressure" else pts[:, 1]
                    fig.add_trace(go.Scatter(x=pts[:, 0], y=py, mode="markers", hoverinfo="skip",
                                             marker=dict(size=6, color=ui.MAGENTA,
                                                         line=dict(width=1, color=ui.chart_theme()["neutral_line"]))))
                    _layout(fig, 340, showlegend=False, xaxis_title=mp["x"], yaxis_title=ylab,
                            title=dict(text=f"{pdres['of']} over {mp['x']} × {ylab.split(' (')[0]}", font=dict(size=13)))
                    mcols[i].plotly_chart(fig, width="stretch", config=PLOT_CFG, key=f"ml_pd2_{i}")
            if pdres["pairs_missing"]:
                st.caption("Not drawn: " + ", ".join(pdres["pairs_missing"]) +
                           " — select `o2_frac` and the CO₂ / pressure feature in tab 2 to see it.")
            st.caption("A what-if on the model, not a law of physics: engineered duplicates (e.g. `o2_partial_kPa` "
                       "with `o2_frac`) are held at their real values while one input moves.")
        if diag:
            # ---- calibration (#59 item 5)
            cal = diag.get("calibration")
            if cal:
                ui.section("Can the probabilities be read as percentages?")
                c1, c2 = st.columns([1.3, 1])
                ml_ui.reliability_chart({"id": f"lab_{ds}_{target}", "reliability": cal["bins"],
                                         "reliability_of": cal["of"]}, where=c1, key="lab")
                with c2:
                    _metric(st, "Brier score (lower is better)", f"{cal['brier']:.3f}",
                            f"{cal['brier'] - cal['brier_baseline']:+.3f} vs baseline", delta_color="inverse",
                            help=f"Mean squared error of {cal['of']} out of fold. Baseline = {cal['baseline']} "
                                 f"({cal['brier_baseline']:.3f}).")
                    _metric(st, "Calibration error", f"{cal['ece'] * 100:.0f} points",
                            help="Average gap between predicted and observed rate over the bins, weighted by tests.")
                ui.callout("note", f"**{cal['verdict']}**")
                st.caption(f"Out-of-fold predictions ({diag['folds']}-fold, every test predicted by a model that never "
                           "saw it) · bins 20 points wide, as on the Models hub; bigger dot = more tests.")
            # ---- error analysis (#59 item 6)
            ui.section("Where the model is wrong")
            errs = pd.DataFrame(diag["errors"])
            if len(errs):
                psi = D.DATASETS[ds]["source"]["psi_id"]
                if "flags" in errs:
                    errs["⚠ ETL correction"] = errs.pop("flags").fillna("").map(lambda f: f"⚠ {f}" if f else "")
                if "ln_pressure" in errs:  # the model's log-pressure, shown in kPa for people
                    errs["ln_pressure"] = np.exp(errs["ln_pressure"].astype(float)).round(1)
                    errs = errs.rename(columns={"ln_pressure": "pressure (kPa)"})
                errs["Data explorer"] = [f"explorer?psi={psi}&test={t}" for t in errs["test_id"]]
                rename = {"test_id": "Test", "date_iso": "Date", "actual": "Actual", "predicted": "Predicted",
                          "p_actual": "P(actual)", "p_predicted": "P(predicted)", "error": "Error (actual − predicted)"}
                errs = errs.rename(columns=rename)
                if task == "classification":
                    st.markdown(f"The 10 tests the model found **least likely to end the way they really did** "
                                f"(out of fold; {diag['n_misclassified']} of {diag['n_rows']} tests are misclassified "
                                "in this pass). Check them in the Data explorer — some are borderline conditions, "
                                "some may be data issues.")
                else:
                    st.markdown(f"The 10 tests with the **largest prediction error** (out of fold, {diag['unit']}).")
                ui.table(errs, width="stretch", raw=("Data explorer",), key="ml_errors",
                         column_config={"Data explorer": st.column_config.LinkColumn("Data explorer", display_text="open")})
                ui.provenance_label("psi", f"Test rows from the {psi} table {ui.source_link(D.DATASETS[ds]['source']['doi'], 'DOI')} "
                                           "· predictions: ML model, out of fold")

# ====================================================================== 6. predict
with tab_predict:
    res_now = M.peek(config) if not blocked else None
    if blocked:
        ui.empty_state("Choose features without leakage first (tab 2).")
    elif res_now is None:
        ui.empty_state("Train this configuration first (tab 3) — predictions always come with its cross-validated error.")
    elif res_now["leakage_demo"]:
        ui.callout("limit", "No predictions in the leakage demo: the leaking features are only known after the test.")
    else:
        st.markdown(f"Set the inputs. The model ({M.MODEL_LABELS[model]}, trained on {res_now['n_rows']} real tests) "
                    "refuses any value outside the range that was actually tested.")
        ranges = M.input_ranges(config)
        inputs, cols = {}, st.columns(3)
        for i, f in enumerate(features):
            rng = ranges[f]
            with cols[i % 3]:
                if isinstance(rng[0], str):
                    inputs[f] = st.selectbox(f, rng, key=f"ml_in_{ds}_{f}")
                else:
                    lo, hi = rng
                    span = (hi - lo) or abs(hi) or 1.0
                    med = float(df[f].median())
                    med = min(max(med, lo), hi)
                    step = float(10 ** np.floor(np.log10(span / 100))) if span > 0 else 0.01
                    inputs[f] = st.slider(f"{f} (tested {lo:.4g}–{hi:.4g})", float(lo - 0.25 * span),
                                          float(hi + 0.25 * span), med, step, key=f"ml_in_{ds}_{f}")
        if any(cat[f]["kind"] == "engineered" for f in features):
            st.caption("Engineered inputs are set directly here — keep them consistent with the raw ones "
                       "(e.g. `o2_partial_kPa` = `o2_frac` × `pressure_kPa`).")
        pr = M.predict(config, inputs)
        ui.provenance_label("ml", f"{M.MODEL_LABELS[model]} estimate — cross-validated on real PSI tests, not a "
                                  "physical simulation.")
        if pr["refused"]:
            ui.callout("limit", f"**No prediction:** {pr['refused']}")
        elif task == "classification":
            probs = pd.DataFrame({"class": list(pr["probs"]), "probability": list(pr["probs"].values())})
            p1, p2 = st.columns([1, 1.3])
            _metric(p1, "Most likely", pr["predicted"], f"{pr['probs'][pr['predicted']]:.0%}", delta_color="off")
            p1.caption(f"Cross-validated accuracy of this configuration: {pr['cv']['accuracy']['mean']:.3f} "
                       f"(baseline {pr['cv']['baseline_accuracy']:.3f}).")
            fig = px.bar(probs, x="probability", y="class", orientation="h", color="class", text="probability",
                         color_discrete_map=CLASS_COLORS, category_orders=ORDERS, range_x=[0, 1], title="Class probabilities")
            fig.update_traces(texttemplate="%{x:.0%}", textposition="outside", cliponaxis=False)
            p2.plotly_chart(_layout(fig, 220, showlegend=False), width="stretch", config=PLOT_CFG)
        else:
            _metric(st, f"Predicted {target}", f"{pr['value']:.3g} {pr['unit']}",
                    f"± {pr['cv']['mae']['mean']:.3g} {pr['unit']} typical error (CV MAE)", delta_color="off")
        ui.section("The 5 most similar real tests")
        ui.table(pd.DataFrame(pr["nearest"]), width="stretch")
        st.caption("Distance over the chosen numeric inputs after standardising each one. Check how the real "
                   "tests ended before trusting a prediction.")
