"""Persisted models (#64 item 8). Run: python -m pytest ai/tests/test_model_store.py -q"""
import inspect

from ai import model_store as MS


def test_committed_models_are_current():
    """processed/models/*.joblib match today's data, model code and library versions; otherwise run
    `python -m ai.model_store --build` (the app would silently retrain on every cold start)."""
    assert MS.status() == {name: "current" for name in MS.BUILDERS}


def test_decorator_loads_when_current_and_trains_when_anything_changed(monkeypatch, tmp_path):
    monkeypatch.setattr(MS, "STORE", tmp_path / "models")
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(MS, "ROOT", tmp_path)
    calls = []

    def builder():
        calls.append(1)
        return {"model": len(calls)}
    wrapped = MS.persisted("toy", inputs=("data.csv",))(builder)
    assert wrapped() == {"model": 1} and len(calls) == 1          # nothing stored: trains
    MS.save("toy", MS.fingerprint(inspect.getsourcefile(builder), ("data.csv",)), {"model": "stored"})
    assert wrapped() == {"model": "stored"} and len(calls) == 1   # current file: loaded, no training
    data.write_text("a,b\n1,3\n", encoding="utf-8")                # the data changed: never served stale
    assert wrapped() == {"model": 2} and len(calls) == 2
    (tmp_path / "models" / "toy.joblib").write_bytes(b"not a pickle")
    assert wrapped() == {"model": 3}                               # unreadable file: trains


def test_stored_flex_model_predicts_like_the_registry():
    """The stored 3-class model and goes-out model are the trained objects (same classes, valid probabilities)."""
    import numpy as np
    from analysis import flex_classifier as F
    _, key_file = MS._builder("flex_fitted")
    model, rows, card = MS.load("flex_fitted", MS.fingerprint(key_file))
    bmodel, thr = MS.load("flex_burnout", MS.fingerprint(MS._builder("flex_burnout")[1]))
    X = np.array([F.features(r) for r in rows[:5]])
    assert sorted(model.classes_) == sorted(F.CLASSES) and np.allclose(model.predict_proba(X).sum(1), 1)
    assert 0 < thr < 1 and bmodel.predict_proba(X).shape == (5, 2) and card["n_train"] == len(rows)
