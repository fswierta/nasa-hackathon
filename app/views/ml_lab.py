"""ML Lab (issue #47, owned by pc2): data exploration → feature engineering → model → prediction.

Strictly classical, cross-validated machine learning on the PSI test tables (scikit-learn) — no LLM.
Backend: ai/mllab.py (pure functions). Data: analysis/datasets.py (main's data layer).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai import mllab as M
from analysis import datasets as D
from app import data

# Colours: categorical identity uses the app's established outcome palette (views/predict.py), in fixed
# order; magnitudes use one blue ramp; correlation uses a blue <-> red diverging pair with a grey midpoint.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#9b6bd6", "#d6a12a"]
SEQ = [[0.0, "#eef4fc"], [0.5, "#7fb0ea"], [1.0, "#0b3d7a"]]
DIV = [[0.0, "#2a78d6"], [0.5, "#383835"], [1.0, "#e0484b"]]
KIND_LABEL = {"raw": "Raw — set before the test", "engineered": "Engineered — computed from raw columns",
              "pi_derived": "Published by the investigators", "post_test": "Measured during/after the test"}
PLOT_CFG = {"displayModeBar": False}
ss = st.session_state
ss.setdefault("ml_history", [])


def _layout(fig, h=300, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), legend_title_text="", **kw)
    return fig


def _reset_to_shipped():
    """on_click: load the configuration behind analysis.flex_classifier into the widgets."""
    cfg = M.shipped_config()
    ss["ml_dataset"] = cfg["dataset"]
    ss[f"ml_target_{cfg['dataset']}"] = cfg["target"]
    for f in D.feature_catalog(cfg["dataset"]):
        ss[f"ml_feat_{cfg['dataset']}_{cfg['target']}_{f['name']}"] = f["name"] in cfg["features"]
    ss.update({"ml_scale": cfg["scale"], "ml_impute": cfg["impute"], "ml_exclude": cfg["exclude_flagged"],
               "ml_leak_demo": False, "ml_model_classification": cfg["model"],
               "ml_rf_trees": cfg["params"]["n_estimators"], "ml_rf_leaf": cfg["params"]["min_samples_leaf"],
               "ml_rf_depth": 0, "ml_balanced": True, "ml_folds": cfg["cv"]["folds"],
               "ml_repeats": cfg["cv"]["repeats"], "ml_seed": cfg["cv"]["seed"]})


st.title("🧪 ML Lab")
st.markdown("Machine learning on the PSI test tables, **step by step and in the open**: look at the data, choose "
            "and engineer features, train a model with cross-validation, then use it — with every setting "
            "changeable. Classical machine learning only (scikit-learn); **no language model on this page**.")

tab_explore, tab_features, tab_train, tab_predict = st.tabs(
    ["1 · Explore the data", "2 · Engineer features", "3 · Train & evaluate", "4 · Predict"])

# ====================================================================== 1. explore
with tab_explore:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        ds = st.radio("Dataset", list(D.DATASETS), key="ml_dataset", horizontal=True,
                      format_func=lambda k: D.DATASETS[k]["title"].split(" — ")[0])
    spec = D.DATASETS[ds]
    with c2:
        targets = list(spec["targets"])
        target = st.selectbox("Target — what the model should predict", targets, key=f"ml_target_{ds}",
                              format_func=lambda t: f"{t} ({spec['targets'][t]['type']})")
    tmeta = spec["targets"][target]
    task = tmeta["type"]
    # colour follows the class, never its rank: one fixed colour per class, in the dataset's class order
    # (FLEX outcome: extinction blue, disruption orange, completion green — as on the Predict page)
    CLASS_ORDER = [M.class_label(ds, target, c) for c in tmeta.get("classes", [])]
    CLASS_COLORS = {c: CAT[i % len(CAT)] for i, c in enumerate(CLASS_ORDER)}
    ORDERS = {"class": CLASS_ORDER, "_y": CLASS_ORDER}
    src = spec["source"]
    st.markdown(f"**{spec['title']}** · data: PSI [{src['psi_id']} · DOI {src['doi']}]({data.doi_url(src['doi'])})")
    st.info(f"**Scope:** {spec['scope']}", icon="🎯")
    st.caption(f"Target `{target}`: {tmeta['note']}")

    df = D.load(ds)
    summ = D.summary(ds)
    cat = M.catalog(ds)
    numeric = [f["name"] for f in D.feature_catalog(ds) if f["unit"] != "category" and f["name"] in df.columns]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tests (rows)", summ["rows"])
    m2.metric(f"Rows with a known `{target}`", summ["targets"][target]["n"])
    m3.metric("Missing values (all features)", sum(summ["missing"].values()))
    m4.metric("Rows with ETL corrections", summ["rows_with_corrections"])
    st.caption("Small data: a few hundred tests. Every score on this page is cross-validated and shown next to a "
               "baseline that knows nothing about the conditions.")

    y_lab = (df[target].map(lambda v: M.class_label(ds, target, v)) if task == "classification" else df[target])
    if task == "classification":
        bal = y_lab[df[target].notna()].value_counts().rename_axis("class").reset_index(name="tests")
        fig = px.bar(bal, x="tests", y="class", orientation="h", color="class", color_discrete_map=CLASS_COLORS, category_orders=ORDERS,
                     text="tests", title="Target balance")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(_layout(fig, 60 + 45 * len(bal), showlegend=False), width="stretch", config=PLOT_CFG)
    else:
        fig = px.histogram(df.dropna(subset=[target]), x=target, nbins=30, color_discrete_sequence=[CAT[0]],
                           title=f"Target distribution ({tmeta.get('unit', '')})")
        st.plotly_chart(_layout(fig, 260, bargap=0.05), width="stretch", config=PLOT_CFG)

    with st.expander("Each feature, split by the target", expanded=True):
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
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#1a1a19")))
        st.plotly_chart(_layout(fig, 340), width="stretch", config=PLOT_CFG)
        if ds == "cfig" and "o2_frac" in (sx, sy):
            st.caption("⚠ `o2_frac` of **inverse** flames is the O₂ in the burner stream, not an atmosphere "
                       "(`o2_role = burner`).")

    st.markdown("##### Feature catalog")
    st.dataframe(pd.DataFrame([{"feature": f["name"], "kind": f["kind"], "unit": f["unit"], "formula": f["formula"],
                                "why": f["why"],
                                "leaks the answer": ("yes — measured during/after the test" if f["kind"] == "post_test"
                                                     else ("for " + ", ".join(f["leak"]) if f.get("leak") else ""))}
                               for f in D.feature_catalog(ds)]), hide_index=True, width="stretch")

# ====================================================================== 2. features
with tab_features:
    st.markdown(f"Choose the inputs for predicting **`{target}`**. Defaults are the numeric features that are known "
                "*before* a test starts. Hover a name for its formula and physical reason.")
    defaults = set(D.default_features(ds, target))
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
                if ds == "cfig" and f["name"] == "o2_frac":
                    st.caption("Inverse flames: burner-stream O₂, not an atmosphere.")

    o1, o2, o3, o4 = st.columns(4)
    scale = o1.checkbox("Scale features (z-score)", value=True, key="ml_scale",
                        help="Needed for logistic regression and k-NN; trees don't care.")
    impute = o2.radio("Missing values", ["drop", "median"], key="ml_impute", horizontal=True,
                      help="drop: skip rows with a missing input · median: fill from the training folds only")
    exclude = o3.checkbox("Exclude rows with ETL corrections", value=False, key="ml_exclude",
                          help="Rows whose values main's ETL had to correct (flags F1–F10).")
    leak_demo = o4.checkbox("Show me what leakage does", value=False, key="ml_leak_demo",
                            help="Allows post-test features so you can see how a leaking model looks too good.")

    refused = M.check_features(ds, target, features)
    if refused and not leak_demo:
        st.error("**Refused — these features leak the answer:** " +
                 "; ".join(f"`{r['feature']}` ({r['reason']})" for r in refused) +
                 ". Untick them, or tick *Show me what leakage does* to see the effect.", icon="🚫")
    elif refused:
        st.error("**LEAKAGE DEMO** — the selected features include information measured after the test. "
                 "Scores will look excellent and mean nothing.", icon="🧪")

    if features:
        st.markdown("##### Feature matrix preview (first rows)")
        prev = df[["test_id"] + features + [target]].dropna(subset=[target])
        st.dataframe(prev.head(8), hide_index=True, width="stretch")
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
        st.warning("Select at least one feature.")

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
            cw = "balanced" if st.checkbox("Balance the classes", value=True, key="ml_balanced",
                                           help="Weights rare outcomes up; otherwise the model can ignore them.") else None
    with a2:
        params = {}
        if model == "random_forest":
            params["n_estimators"] = st.slider("Trees", 20, 300, M.DEFAULT_PARAMS[model]["n_estimators"], 10, key="ml_rf_trees")
            params["min_samples_leaf"] = st.slider("Min. tests per leaf", 1, 10, 2, key="ml_rf_leaf")
            depth = st.slider("Max depth (0 = unlimited)", 0, 20, 0, key="ml_rf_depth")
            params["max_depth"] = depth or None
        elif model == "gradient_boosting":
            params["n_estimators"] = st.slider("Boosting rounds", 20, 300, 150, 10, key="ml_gb_n")
            params["learning_rate"] = st.select_slider("Learning rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.1, key="ml_gb_lr")
            params["max_depth"] = st.slider("Tree depth", 1, 6, 3, key="ml_gb_depth")
        elif model == "logreg":
            params["C"] = st.select_slider("Regularisation C (small = simpler)", [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                                           1.0, key="ml_lr_c")
        elif model == "knn":
            params["n_neighbors"] = st.slider("Neighbours k", 1, 25, 7, key="ml_knn_k")
        elif model == "ridge":
            params["alpha"] = st.select_slider("Ridge α (large = simpler)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0,
                                               key="ml_ridge_a")
    with a3:
        folds = st.slider("CV folds", 3, 10, 5, key="ml_folds")
        repeats = st.slider("CV repeats", 1, 5, 3, key="ml_repeats")
        seed = st.number_input("Random seed", 0, 9999, 0, key="ml_seed")

    config = {"dataset": ds, "target": target, "features": features, "scale": bool(scale), "impute": impute,
              "exclude_flagged": bool(exclude), "model": model, "params": params, "class_weight": cw,
              "cv": {"folds": int(folds), "repeats": int(repeats), "seed": int(seed)}, "leakage_demo": bool(leak_demo)}
    blocked = (refused and not leak_demo) or not features
    b1, b2, _ = st.columns([1, 1.3, 3])
    train = b1.button("▶ Train & cross-validate", type="primary", key="ml_train", disabled=bool(blocked))
    b2.button("↺ Reset to the shipped model", key="ml_reset", on_click=_reset_to_shipped,
              help="Loads the configuration behind the Predict-extinction page (analysis.flex_classifier).")
    if refused and not leak_demo:
        st.error("Training is blocked: " + ", ".join(f"`{r['feature']}`" for r in refused) +
                 " would leak the answer (see tab 2).", icon="🚫")

    res = None
    if train and not blocked:
        with st.spinner("Cross-validating… (each fold is trained from scratch)"):
            t0 = time.perf_counter()
            try:
                res = M.run(config)
            except (M.FeatureRefused, ValueError) as e:
                st.error(f"Not trained: {e}")
        if res:
            ss["ml_history"].append({
                "#": len(ss["ml_history"]) + 1, "dataset": ds, "target": target, "model": M.MODEL_LABELS[model],
                "features": len(features), "feature list": ", ".join(features),
                **({"accuracy": res["metrics"]["accuracy"]["mean"], "balanced acc.": res["metrics"]["balanced_accuracy"]["mean"],
                    "baseline acc.": res["baseline"]["accuracy"]["mean"]} if task == "classification" else
                   {"R²": res["metrics"]["r2"]["mean"], "MAE": res["metrics"]["mae"]["mean"],
                    "baseline R²": res["baseline"]["r2"]["mean"]}),
                "rows": res["n_rows"], "CV": res["cv"]["scheme"], "leakage demo": "⚠ yes" if res["leakage_demo"] else "",
                "seconds": 0.0 if res.get("from_cache") else round(time.perf_counter() - t0, 1)})
    elif not blocked:
        res = M.peek(config)
        if res is None:
            st.info("Settings changed — press **Train & cross-validate** to see how this configuration performs.")

    if res:
        if res["leakage_demo"]:
            st.markdown("<div style='background:#7a1010;color:#fff;padding:10px 14px;border-radius:6px;"
                        "font-weight:700'>LEAKAGE DEMO — these scores use information measured after the test "
                        "and are not a real model.</div>", unsafe_allow_html=True)
        how = ("precomputed for this exact configuration and data (processed/ai/mllab_runs.json)"
               if res.get("from_cache") else f"trained just now in {res['seconds']} s")
        st.caption(f"{res['n_rows']} tests used ({res['n_dropped']} dropped: missing target/inputs or excluded) · "
                   f"{res['cv']['scheme']} = {res['cv']['n_fits']} models · {how}")
        k1, k2, k3 = st.columns(3)
        if task == "classification":
            acc, bacc, base = res["metrics"]["accuracy"], res["metrics"]["balanced_accuracy"], res["baseline"]
            k1.metric("Accuracy (CV)", f"{acc['mean']:.3f} ± {acc['sd']:.3f}",
                      f"{acc['mean'] - base['accuracy']['mean']:+.3f} vs baseline")
            k2.metric("Balanced accuracy (CV)", f"{bacc['mean']:.3f} ± {bacc['sd']:.3f}",
                      f"{bacc['mean'] - base['balanced_accuracy']['mean']:+.3f} vs baseline")
            k3.metric("Baseline: " + base["name"], f"{base['accuracy']['mean']:.3f}",
                      help=f"Balanced accuracy of the baseline: {base['balanced_accuracy']['mean']:.3f} (chance).")
            if acc["mean"] <= base["accuracy"]["mean"]:
                st.warning("This model does **not** beat always guessing the most common class on accuracy.")
        else:
            r2, mae, base = res["metrics"]["r2"], res["metrics"]["mae"], res["baseline"]
            unit = tmeta.get("unit", "")
            k1.metric("R² (CV)", f"{r2['mean']:.3f} ± {r2['sd']:.3f}", f"{r2['mean'] - base['r2']['mean']:+.3f} vs baseline")
            k2.metric(f"Mean absolute error (CV, {unit})", f"{mae['mean']:.3f} ± {mae['sd']:.3f}",
                      f"{mae['mean'] - base['mae']['mean']:+.3f} vs baseline", delta_color="inverse")
            k3.metric("Baseline: " + base["name"], f"R² {base['r2']['mean']:.3f}", help=f"MAE {base['mae']['mean']:.3f} {unit}")
        for w in res["warnings"]:
            st.warning(w, icon="⚠️")

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
        st.markdown("##### Run history (this session)")
        st.dataframe(pd.DataFrame(ss["ml_history"]), hide_index=True, width="stretch")
        if st.button("Clear history", key="ml_clear"):
            ss["ml_history"] = []
            st.rerun()

# ====================================================================== 4. predict
with tab_predict:
    res_now = M.peek(config) if not blocked else None
    if blocked:
        st.info("Choose features without leakage first (tab 2).")
    elif res_now is None:
        st.info("Train this configuration first (tab 3) — predictions always come with its cross-validated error.")
    elif res_now["leakage_demo"]:
        st.warning("No predictions in the leakage demo: the leaking features are only known after the test.")
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
                    step = float(10 ** np.floor(np.log10(span / 100))) if span > 0 else 0.01
                    inputs[f] = st.slider(f"{f} (tested {lo:.4g}–{hi:.4g})", float(lo - 0.25 * span),
                                          float(hi + 0.25 * span), med, step, key=f"ml_in_{ds}_{f}")
        if any(cat[f]["kind"] == "engineered" for f in features):
            st.caption("Engineered inputs are set directly here — keep them consistent with the raw ones "
                       "(e.g. `o2_partial_kPa` = `o2_frac` × `pressure_kPa`).")
        pr = M.predict(config, inputs)
        if pr["refused"]:
            st.error(f"**No prediction:** {pr['refused']}", icon="🚫")
        elif task == "classification":
            probs = pd.DataFrame({"class": list(pr["probs"]), "probability": list(pr["probs"].values())})
            p1, p2 = st.columns([1, 1.3])
            p1.metric("Most likely", pr["predicted"], f"{pr['probs'][pr['predicted']]:.0%}", delta_color="off")
            p1.caption(f"Cross-validated accuracy of this configuration: {pr['cv']['accuracy']['mean']:.3f} "
                       f"(baseline {pr['cv']['baseline_accuracy']:.3f}).")
            fig = px.bar(probs, x="probability", y="class", orientation="h", color="class", text="probability",
                         color_discrete_map=CLASS_COLORS, category_orders=ORDERS, range_x=[0, 1], title="Class probabilities")
            fig.update_traces(texttemplate="%{x:.0%}", textposition="outside", cliponaxis=False)
            p2.plotly_chart(_layout(fig, 220, showlegend=False), width="stretch", config=PLOT_CFG)
        else:
            st.metric(f"Predicted {target}", f"{pr['value']:.3g} {pr['unit']}",
                      f"± {pr['cv']['mae']['mean']:.3g} {pr['unit']} typical error (CV MAE)", delta_color="off")
        st.markdown("##### The 5 most similar real tests")
        st.dataframe(pd.DataFrame(pr["nearest"]), hide_index=True, width="stretch")
        st.caption("Distance over the chosen numeric inputs after standardising each one. Check how the real "
                   "tests ended before trusting a prediction.")
