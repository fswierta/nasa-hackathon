"""Every number the submission states must match processed/key_numbers.json (#38).
Run from repo root:  py -m pytest etl/tests -q"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", *sorted((ROOT / "submission").glob("*.md"))]
NUM = r"([\d,]+(?:\.\d+)?)"

# (regex with one or two number groups, key(s), scale applied to the document's number)
CHECKS = [
    (rf"{NUM} test records", ["test_records"], 1),
    (rf"indexed {NUM} PDF", ["documents_indexed"], 1),
    (rf"{NUM} PDF/Word documents", ["documents_indexed"], 1),
    (rf"{NUM} passages", ["passages"], 1),
    (rf"\*\*{NUM}\*\* files", ["psi_files"], 1),
    (rf"~?\*?\*?{NUM} TB", ["psi_raw_tb"], 1),
    (rf"\*\*{NUM}\*\* publications", ["publications"], 1),
    (rf"\*\*{NUM}\*\* combustion investigations", ["investigations"], 1),
    (rf"\*\*{NUM}\*\* flown", ["flight_investigations"], 1),
    (rf"tested only up to \**{NUM} ?% O2", ["max_solid_o2_pct"], 1),
    (rf"highest \*?atmosphere\*? O2 tested: {NUM} ?%", ["max_ambient_o2_pct"], 1),
    (rf"{NUM} ?% accuracy vs {NUM} ?% baseline", ["classifier_cv_accuracy_pct", "classifier_baseline_pct"], 1),
    (rf"accuracy {NUM} vs {NUM}", ["classifier_cv_accuracy_pct", "classifier_baseline_pct"], 100),
    (rf"R² ≈ {NUM}, n = {NUM}", ["burning_rate_r2", "burning_rate_n_train"], 1),
    (rf"{NUM} clusters", ["similarity_clusters"], 1),
    (rf"recovers {NUM} of {NUM}", ["similarity_related_found", "similarity_related_total"], 1),
    (rf"forest \({NUM} tests", ["classifier_n_train"], 1),
    # goes-out-vs-burns-completely model and retrieval benchmark (#67, #68)
    (rf"accuracy {NUM} ?% \(baseline {NUM} ?%\)", ["goes_out_accuracy_pct", "goes_out_baseline_pct"], 1),
    (rf"\({NUM} ?% vs {NUM} ?% baseline, ROC-AUC", ["goes_out_accuracy_pct", "goes_out_baseline_pct"], 1),
    (rf"recall {NUM} ?% → {NUM} ?%", ["completion_recall_3class_pct", "completion_recall_pct"], 1),
    (rf"finds {NUM} ?% of complete burns vs {NUM} ?%", ["completion_recall_pct", "completion_recall_3class_pct"], 1),
    (rf"finds {NUM} ?% at {NUM} ?% precision", ["completion_recall_pct", "completion_precision_pct"], 1),
    (rf"Benchmark \({NUM} questions", ["retrieval_questions"], 1),
    (rf"{NUM} questions with the source fixed", ["retrieval_questions"], 1),
]


def _close(doc_value: float, true_value: float, written: str) -> bool:
    decimals = len(written.split(".")[1]) if "." in written else 0
    return abs(doc_value - true_value) <= 0.5 * 10 ** -decimals + 1e-9


# Numbers that come out of a trained model: they move slightly with scikit-learn / numpy versions (#56),
# so a rebuild is compared with a tolerance; everything else must match exactly.
MODEL_KEYS = {"classifier_cv_accuracy_pct": 3, "classifier_baseline_pct": 0.01, "burning_rate_r2": 0.05,
              "gas_ignition_accuracy_pct": 3, "gas_self_extinguish_accuracy_pct": 3,
              "gas_ignition_baseline_pct": 0.01, "gas_self_extinguish_baseline_pct": 0.01,
              "heptane_30pct_o2_goes_out_pct": 8, "heptane_30pct_o2_co2_for_90pct": 0.1,
              "similarity_clusters": 1, "similarity_related_found": 2,
              "goes_out_accuracy_pct": 3, "goes_out_baseline_pct": 0.01, "completion_recall_3class_pct": 8,
              "completion_recall_pct": 8, "completion_precision_pct": 10, "completion_recall_conservative_pct": 10,
              "goes_out_roc_auc": 0.05}


def test_rebuild_matches_committed_key_numbers():
    """The committed file is the source of truth for docs; a rebuild on this machine must reproduce it."""
    import sys
    sys.path.insert(0, str(ROOT / "etl"))
    import build_key_numbers
    committed = json.loads((ROOT / "processed" / "key_numbers.json").read_text(encoding="utf-8"))
    rebuilt = build_key_numbers.build()
    assert set(rebuilt) == set(committed), set(rebuilt) ^ set(committed)
    exact, drift = [], []
    for key, entry in committed.items():
        a, b = entry["value"], rebuilt[key]["value"]
        if key in MODEL_KEYS:
            if abs(a - b) > MODEL_KEYS[key]:
                drift.append(f"{key}: committed {a}, rebuilt {b}")
        elif a != b:
            exact.append(f"{key}: committed {a}, rebuilt {b}")
    assert not exact, "data changed - run `py etl/build_key_numbers.py` and commit: " + "; ".join(exact)
    assert not drift, ("model numbers differ beyond tolerance - library versions? run `pip install -r requirements.txt` "
                       "(analysis/requirements.txt pins the ML stack): " + "; ".join(drift))


def test_documents_agree_with_key_numbers():
    k = json.loads((ROOT / "processed" / "key_numbers.json").read_text(encoding="utf-8"))
    problems, checked = [], 0
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for pattern, keys, scale in CHECKS:
            for m in re.finditer(pattern, text):
                for written, key in zip(m.groups(), keys):
                    checked += 1
                    value = float(written.replace(",", "")) * scale
                    true = k[key]["value"]
                    w = written.replace(",", "")
                    if scale != 1:  # 0.765 * 100 -> compare at the precision written
                        w = f"{value:.{max(len(w.split('.')[1]) - 2, 0) if '.' in w else 0}f}"
                    if not _close(value, true, w):
                        problems.append(f"{doc.name}: '{m.group(0)}' -> {key} is {true}")
    assert not problems, "\n".join(problems)
    assert checked >= 10, f"only {checked} numbers matched - patterns out of date?"


def test_key_numbers_are_sane():
    k = json.loads((ROOT / "processed" / "key_numbers.json").read_text(encoding="utf-8"))
    assert k["investigations"]["value"] == 24
    assert k["solid_tests_near_lunar_base"]["value"] == 0
    assert k["max_ambient_o2_pct"]["value"] <= 41  # burner O2 (85 %) must not leak in
    assert k["classifier_cv_accuracy_pct"]["value"] > k["classifier_baseline_pct"]["value"]
