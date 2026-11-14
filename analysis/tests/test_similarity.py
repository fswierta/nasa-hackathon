"""Run from repo root:  py -m pytest analysis/tests -q   (first run downloads the ~80 MB MiniLM model once)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import similarity as sim  # noqa: E402


def _data() -> dict:
    return json.loads(sim.OUT_FILE.read_text(encoding="utf-8"))


def _top(d, pid, k):
    return [n["id"] for n in d["neighbors"][pid][:k]]


def test_build_writes_contract():
    sim.main()
    d = _data()
    assert {"method", "points", "clusters", "neighbors"} <= set(d)
    assert len(d["points"]) == 24
    cluster_ids = {c["id"] for c in d["clusters"]}
    members = [m for c in d["clusters"] for m in c["members"]]
    assert sorted(members) == sorted(p["id"] for p in d["points"])  # every investigation in exactly one cluster
    for p in d["points"]:
        assert p["cluster"] in cluster_ids and isinstance(p["x"], float) and isinstance(p["y"], float)
    for c in d["clusters"]:
        assert c["label"] and c["dominant_fuel_phase"]["share"] > 0


def test_neighbours_are_well_formed():
    d = _data()
    for pid, nbrs in d["neighbors"].items():
        assert len(nbrs) == sim.K_NEIGHBOURS
        assert pid not in {n["id"] for n in nbrs}
        scores = [n["score"] for n in nbrs]
        assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
        assert all(isinstance(n["shared"], list) for n in nbrs)


def test_known_families_are_neighbours():
    d = _data()
    assert "PSI-25" in _top(d, "PSI-26", 2)                       # BASS <-> BASS-II
    assert {"PSI-99", "PSI-100"} <= set(_top(d, "PSI-98", 3))      # Saffire I/II/III
    assert "PSI-68" in _top(d, "PSI-69", 3)                       # FLEX <-> FLEX-2
    assert _top(d, "PSI-102", 1) == ["PSI-101"]                   # SAME <-> SAME-R


def test_recovers_most_psi_related_links():
    v = _data()["validation"]
    assert v["psi_related_pairs"] >= 15
    assert v["share"] >= 0.6


def test_similarity_matrix_is_symmetric():
    import numpy as np
    m = np.array(_data()["similarity_matrix"]["combined"])
    assert m.shape == (24, 24) and np.allclose(m, m.T) and np.allclose(np.diag(m), 1.0)


def test_shared_facts_name_concrete_overlaps():
    invs = {i["id"]: i for i in json.loads(sim.INV_FILE.read_text(encoding="utf-8"))}
    facts = sim.shared_facts(invs["PSI-26"], invs["PSI-25"])
    assert "PSI lists them as related investigations" in facts
    assert "same hardware: MSG flow duct" in facts
    assert any(f.startswith("shared fuels:") and "PMMA" in f for f in facts)
