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
]


def _close(doc_value: float, true_value: float, written: str) -> bool:
    decimals = len(written.split(".")[1]) if "." in written else 0
    return abs(doc_value - true_value) <= 0.5 * 10 ** -decimals + 1e-9


def test_documents_agree_with_key_numbers():
    import sys
    sys.path.insert(0, str(ROOT / "etl"))
    import build_key_numbers
    build_key_numbers.main()
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
