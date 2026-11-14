"""Run from repo root:  py -m pytest analysis/tests -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import datasets as ds  # noqa: E402


def test_shapes_and_targets():
    flex, cfig = ds.load("flex"), ds.load("cfig")
    assert len(flex) == 274 and len(cfig) == 333
    assert set(flex["outcome"]) == {"extinction", "disruption", "completion"}
    assert flex["went_out"].notna().sum() == 213            # disrupted tests excluded
    assert set(cfig["hot_ignition"].dropna()) == {0.0, 1.0}
    assert cfig.loc[cfig["hot_ignition"] != 1.0, "self_extinguished"].isna().all()  # only defined if it ignited


def test_every_catalog_feature_exists_and_is_documented():
    for name in ds.DATASETS:
        df = ds.load(name)
        for f in ds.feature_catalog(name):
            assert f["name"] in df.columns, (name, f["name"])
            assert f["kind"] in {"raw", "engineered", "pi_derived", "post_test"} and f["why"] and f["formula"]


def test_default_features_never_leak():
    for name, spec in ds.DATASETS.items():
        post = {f["name"] for f in ds.feature_catalog(name) if f["kind"] == "post_test"}
        for target in spec["targets"]:
            feats = ds.default_features(name, target)
            assert feats and not (set(feats) & post) and target not in feats


def test_engineered_features_follow_their_formula():
    flex = ds.load("flex").dropna(subset=["o2_frac", "pressure_kPa"])
    assert ((flex["o2_partial_kPa"] - flex["o2_frac"] * flex["pressure_kPa"]).abs() < 1e-9).all()
    assert ((flex["x_co2"] + flex["x_he"] + flex["x_n2"] + flex["o2_frac"] - 1).abs() < 0.02).mean() > 0.95
    cfig = ds.load("cfig")
    assert set(cfig.loc[cfig["flame_type"] == "inverse", "o2_role"]) == {"burner"}
