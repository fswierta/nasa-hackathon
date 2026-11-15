"""processed/key_numbers.json — every headline number of the project, computed in one place (#38).

The app (Home/About), submission/*.md and README must use these values; etl/tests/test_key_numbers.py
fails when a document states a number that disagrees. Each entry: {"value", "text" (display form),
"source" (where it is computed from)}.

Run from the repo root after the ETL / analysis steps:  py etl/build_key_numbers.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "processed"
sys.path.insert(0, str(ROOT))


def _load(path):
    return json.loads((P / path).read_text(encoding="utf-8"))


def build() -> dict:
    invs = _load("investigations.json")
    tests = list(csv.DictReader((P / "tests.csv").open(encoding="utf-8")))
    cov = _load("coverage.json")
    sim = _load("analysis/similarity.json")
    flex = _load("analysis/flex_boundary.json")["model_card"]
    chunks = [json.loads(l) for l in (P / "ai" / "chunks.jsonl").open(encoding="utf-8")]
    emb = _load("ai/embeddings_meta.json")
    from ai.model import model_card as rate_card
    rate = rate_card()

    hab = {h["name"]: h for h in cov["habitats"]}
    solid_ids = {i["id"] for i in invs if i["fuel_phase"] == "solid"}
    solid_rows = [t for t in tests if t["investigation_id"] in solid_ids and t["row_kind"] == "test"]
    solid_o2 = [float(x) for t in solid_rows for x in (t["o2_frac"], t["o2_frac_max"]) if x]
    amb_o2 = [float(x) for t in tests if t.get("o2_role") != "burner" for x in (t["o2_frac"], t["o2_frac_max"]) if x]
    docs = {c["file"] for c in chunks if c["file"].lower().endswith((".pdf", ".docx"))}
    flags = Counter(part.split(":")[0] for t in tests for part in filter(None, t["flags"].split(" | ")))
    chosen = flex["cv"][flex["model"]]

    def n(value, text, source):
        return {"value": value, "text": text, "source": source}

    k = {
        "investigations": n(len(invs), f"{len(invs)}", "processed/investigations.json"),
        "flight_investigations": n(sum(i["type"] == "flight" for i in invs), None, "investigations.json type"),
        "tier_a_investigations": n(sum(i["relevance_tier"] == "A" for i in invs), None, "investigations.json relevance_tier"),
        "experiment_tables_parsed": n(len({t["source_file"] for t in tests}), None, "tests.csv source_file"),
        "test_records": n(sum(t["row_kind"] != "report_test" for t in tests),
                          f"{sum(t['row_kind'] != 'report_test' for t in tests):,}", "processed/tests.csv rows from the PSI tables"),
        "report_test_rows": n(sum(t["row_kind"] == "report_test" for t in tests), None,
                              "processed/tests.csv rows taken from PSI reports (BRE Final Report 2022), flag R1"),
        "records_with_o2_and_pressure": n(cov["totals"]["rows_with_o2_and_pressure"], None, "coverage.json totals"),
        "psi_files": n(sum(i["n_files"] or 0 for i in invs), f"{sum(i['n_files'] or 0 for i in invs):,}", "investigations.json n_files"),
        "psi_raw_tb": n(round(sum(i["total_size_gb"] or 0 for i in invs) / 1000, 1),
                        f"{sum(i['total_size_gb'] or 0 for i in invs) / 1000:.1f} TB", "investigations.json total_size_gb"),
        "publications": n(sum(len(i["publications"]) for i in invs), None, "investigations.json publications"),
        "documents_indexed": n(len(docs), f"{len(docs)}", "ai/chunks.jsonl distinct PDF/DOCX files"),
        "passages": n(len(chunks), f"{len(chunks):,}", "ai/chunks.jsonl"),
        "unique_vectors": n(emb["n_vectors"], f"{emb['n_vectors']:,}", "ai/embeddings_meta.json"),
        "tests_near_lunar_base": n(hab["Lunar base"]["n_tests_near"], None, "coverage.json habitats (ambient O2 only)"),
        "solid_tests_near_lunar_base": n(hab["Lunar base"]["by_fuel_phase"]["solid"], None, "coverage.json habitats"),
        "tests_near_exploration_atm": n(hab["Exploration atmosphere"]["n_tests_near"], None, "coverage.json habitats"),
        "max_solid_o2_pct": n(round(max(solid_o2) * 100, 1), f"{max(solid_o2) * 100:.1f} %", "tests.csv solid-fuel rows"),
        "max_ambient_o2_pct": n(round(max(amb_o2) * 100, 1), f"{max(amb_o2) * 100:.0f} %", "tests.csv o2_role != burner"),
        "classifier_n_train": n(flex["n_train"], None, "analysis flex model card"),
        "classifier_cv_accuracy_pct": n(round(chosen["cv_accuracy"] * 100, 1), f"{chosen['cv_accuracy'] * 100:.1f} %", "analysis flex model card"),
        "classifier_baseline_pct": n(round(flex["baseline_accuracy"] * 100, 1), f"{flex['baseline_accuracy'] * 100:.1f} %", "analysis flex model card"),
        "burning_rate_r2": n(round(rate["k"]["cv_r2"], 2), f"{rate['k']['cv_r2']:.2f}", "ai.model.model_card()['k']"),
        "burning_rate_n_train": n(rate["k"]["n_train"], None, "ai.model.model_card()['k']"),
        "similarity_clusters": n(sim["cluster_quality"]["k"], None, "analysis/similarity.json"),
        "similarity_related_found": n(sim["validation"]["found_in_top5_neighbours"], None, "analysis/similarity.json validation"),
        "similarity_related_total": n(sim["validation"]["psi_related_pairs"], None, "analysis/similarity.json validation"),
        "rows_with_corrections": n(sum(1 for t in tests if any(p.startswith("F") for p in t["flags"].split(" | ") if p)),
                                   None, "tests.csv flags F*"),
        "corrections_by_fix": n(dict(sorted(flags.items())), None, "tests.csv flags"),
        "solid_material_known_pct": n(round(100 * sum(1 for t in solid_rows if t["fuel"]) / len(solid_rows)),
                                      None, "tests.csv solid-fuel test rows with fuel"),
        "data_gaps": n(len(cov["gaps"]), None, "coverage.json gaps"),
    }
    # fire-scenario engine (analysis/scenario.py): gas-flame models + suppression evidence
    from analysis.scenario import gas_model_card, suppressant_scan, suppression_evidence, assess
    gas = gas_model_card()["models"]
    rich = suppression_evidence()["table"][-1]
    hept = assess({"fuel_class": "droplet", "fuel": "heptane", "o2_frac": 0.30, "pressure_kPa": 101.3})
    scan = suppressant_scan("heptane", 0.30, 101.3, "CO2")
    k.update({
        "gas_ignition_accuracy_pct": n(round(gas["hot_ignition"]["cv_accuracy"] * 100, 1), f"{gas['hot_ignition']['cv_accuracy'] * 100:.1f} %", "analysis.scenario gas model card (CFI-G normal flames)"),
        "gas_ignition_baseline_pct": n(round(gas["hot_ignition"]["baseline_accuracy"] * 100, 1), f"{gas['hot_ignition']['baseline_accuracy'] * 100:.1f} %", "analysis.scenario gas model card"),
        "gas_self_extinguish_accuracy_pct": n(round(gas["self_extinguished"]["cv_accuracy"] * 100, 1), f"{gas['self_extinguished']['cv_accuracy'] * 100:.1f} %", "analysis.scenario gas model card"),
        "gas_self_extinguish_baseline_pct": n(round(gas["self_extinguished"]["baseline_accuracy"] * 100, 1), f"{gas['self_extinguished']['baseline_accuracy'] * 100:.1f} %", "analysis.scenario gas model card"),
        "gas_n_train": n(gas["hot_ignition"]["n_train"], None, "CFI-G normal-flame tests"),
        "o2_rich_went_out_n2_only_pct": n(round(rich["N2 only"]["share_went_out"] * 100), f"{rich['N2 only']['share_went_out'] * 100:.0f} % (n={rich['N2 only']['n']})", "FLEX tests above 22 % O2, nitrogen only"),
        "o2_rich_went_out_co2_pct": n(round(rich["CO2"]["share_went_out"] * 100), f"{rich['CO2']['share_went_out'] * 100:.0f} % (n={rich['CO2']['n']})", "FLEX tests above 22 % O2 with CO2"),
        "o2_rich_went_out_he_pct": n(round(rich["He"]["share_went_out"] * 100), f"{rich['He']['share_went_out'] * 100:.0f} % (n={rich['He']['n']})", "FLEX tests above 22 % O2 with He"),
        "heptane_30pct_o2_goes_out_pct": n(round(hept["prediction"]["p_goes_out"] * 100), None, "analysis.scenario.assess: heptane droplet, 30 % O2, 101.3 kPa"),
        "heptane_30pct_o2_co2_for_90pct": n(scan["first_fraction_reaching_90pct"], None, "analysis.scenario.suppressant_scan: CO2 discharge fraction reaching P(goes out) >= 0.9"),
    })
    bench_path = P / "ai" / "benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        hyb = next(v for name, v in bench["summary"].items() if name.startswith("hybrid"))
        k.update({
            "retrieval_questions": n(bench["n_questions"], None, "ai/benchmark.json"),
            "retrieval_source_hit1_pct": n(round(hyb["hit@1"] * 100), f"{hyb['hit@1'] * 100:.0f} %", "ai/benchmark.json hybrid hit@1 (right source ranked first)"),
            "retrieval_answer_top3_pct": n(round(hyb["answer@3"] * 100), f"{hyb['answer@3'] * 100:.0f} %", "ai/benchmark.json hybrid answer@3 (passage with the answer in top 3)"),
        })
    return k


def main() -> None:
    k = build()
    (P / "key_numbers.json").write_text(json.dumps(k, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote processed/key_numbers.json")
    for name, v in k.items():
        if name != "corrections_by_fix":
            print(f"  {name:32s} {v['value']}")


if __name__ == "__main__":
    main()
