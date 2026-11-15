"""Persisted trained models (issue #64 item 8): no training on a Streamlit Cloud cold start.

The app's models (FLEX outcome + goes-out forests, CFI-G gas forests) used to be trained - with their
cross-validation - in every new process: ~45 s before the first model page answered, again after every sleep.
`@persisted(name)` stores the trained result in processed/models/<name>.joblib (committed) and loads it when
its fingerprint still matches; otherwise it trains exactly as before. Nothing is ever served stale.

    from ai.model_store import persisted

    @lru_cache(maxsize=1)
    @persisted("flex_fitted")              # under the lru_cache: the file is read once per process
    def _fitted(): ...

    python -m ai.model_store --build       # (re)train every model in BUILDERS and write the files
    python -m ai.model_store --check       # which stored models are current

Fingerprint = SHA-256 of the input data (processed/tests.csv), of the source of the module that defines the
builder and of analysis/datasets.py, plus the scikit-learn / numpy / joblib versions (a pickle from another
version is never loaded). Change any of them and the model is retrained on first use until `--build` is re-run
(ai/tests/test_model_store.py fails as a reminder). The files are our own build output, loaded with joblib
(pickle) only from this repository.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import sys
import time
from pathlib import Path

from .config import ROOT

STORE = ROOT / "processed" / "models"
DEFAULT_INPUTS = ("processed/tests.csv", "analysis/datasets.py")
# name -> "module:function" of every persisted builder (what --build trains; the decorator uses the same names)
BUILDERS = {
    "flex_fitted": "analysis.flex_classifier:_fitted",            # 3-class FLEX ending model + its CV card
    "flex_burnout": "analysis.flex_classifier:_fitted_burnout",   # 2-class P(goes out) model + threshold
    "gas_models": "analysis.scenario:_gas_models",                # CFI-G hot-ignition / self-extinguish forests
}
_STATS = {"loaded": [], "trained": []}  # what happened in this process (for the app / tests)


def _versions() -> str:
    import joblib
    import numpy
    import sklearn
    return f"sklearn {sklearn.__version__} | numpy {numpy.__version__} | joblib {joblib.__version__}"


def fingerprint(module_file: str | Path, inputs=DEFAULT_INPUTS) -> str:
    h = hashlib.sha256(_versions().encode())
    for p in [*(ROOT / i for i in inputs), Path(module_file)]:
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()[:24]


def _path(name: str) -> Path:
    return STORE / f"{name}.joblib"


def load(name: str, key: str):
    """The stored value if its key matches, else None (missing, other data / code / versions, unreadable)."""
    import joblib
    p = _path(name)
    if not p.exists():
        return None
    try:
        blob = joblib.load(p)
    except Exception:  # noqa: BLE001 - a broken or foreign file just means: train
        return None
    return blob["value"] if isinstance(blob, dict) and blob.get("key") == key else None


def save(name: str, key: str, value) -> Path:
    import joblib
    STORE.mkdir(parents=True, exist_ok=True)
    p = _path(name)
    joblib.dump({"key": key, "name": name, "versions": _versions(), "value": value}, p, compress=3)
    return p


def persisted(name: str, inputs=DEFAULT_INPUTS):
    """Decorator for a zero-argument model builder: load processed/models/<name>.joblib when current, else build."""
    def deco(fn):
        module_file = inspect.getsourcefile(fn)

        @functools.wraps(fn)
        def wrapper():
            key = fingerprint(module_file, inputs)
            value = load(name, key)
            if value is not None:
                _STATS["loaded"].append(name)
                return value
            _STATS["trained"].append(name)
            return fn()
        wrapper.persisted_name = name
        return wrapper
    return deco


def _builder(name: str):
    mod_name, fn_name = BUILDERS[name].split(":")
    fn = getattr(importlib.import_module(mod_name), fn_name)
    raw = inspect.unwrap(fn)  # past lru_cache and @persisted: always really train here
    return raw, inspect.getsourcefile(raw)


def status() -> dict[str, str]:
    out = {}
    for name in BUILDERS:
        _, module_file = _builder(name)
        key = fingerprint(module_file)
        out[name] = "current" if load(name, key) is not None else ("stale" if _path(name).exists() else "missing")
    return out


def build(names=None) -> None:
    for name in names or BUILDERS:
        raw, module_file = _builder(name)
        t = time.perf_counter()
        value = raw()
        p = save(name, fingerprint(module_file), value)
        print(f"{name:14s} trained in {time.perf_counter() - t:5.1f} s -> {p.relative_to(ROOT)} "
              f"({p.stat().st_size / 1e6:.1f} MB)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="*", metavar="NAME", help="train and store (all when no name is given)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.build is not None:
        build(a.build or None)
    for name, st in status().items():
        print(f"{name:14s} {st}")
    if a.check and any(v != "current" for v in status().values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
