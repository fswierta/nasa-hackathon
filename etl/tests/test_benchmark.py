"""The retrieval benchmark result shipped in the repo must be complete and honest (#32).
(The benchmark itself runs with `py -m ai.benchmark`; this test only reads its output.)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_file_is_complete():
    b = json.loads((ROOT / "processed" / "ai" / "benchmark.json").read_text(encoding="utf-8"))
    assert b["n_questions"] == len(b["questions"]) == 15 and 0 < b["n_named"] < 15   # named and unnamed questions
    assert len(b["rows"]) == 15 * len(b["summary"])
    for name, s in b["summary"].items():
        assert 0 <= s["answer@3"] <= s["hit@3"] <= 1, name          # the strict metric can never exceed the lenient one
    hybrid = next(v for k, v in b["summary"].items() if k.startswith("hybrid"))
    assert hybrid["hit@3"] >= 0.8                                    # the search used by Ask finds the right source


def test_readme_states_the_measured_numbers():
    b = json.loads((ROOT / "processed" / "ai" / "benchmark.json").read_text(encoding="utf-8"))
    hybrid = next(v for k, v in b["summary"].items() if k.startswith("hybrid"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"first in {hybrid['hit@1'] * 100:.0f} %" in readme and f"top 3 for {hybrid['answer@3'] * 100:.0f} %" in readme
