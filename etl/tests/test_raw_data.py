"""The NASA source files are read-only: any change (e.g. a CSV re-saved by Excel with ';' separators, which
silently dropped 120 PSI-10 tests on 14 Nov) fails here. Restore with `git checkout bc2ab8c -- <file>`.
Run from repo root:  py -m pytest etl/tests -q"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "project-data-1-pc1" / "data"


def test_raw_psi_files_are_unchanged():
    manifest = json.loads((ROOT / "etl" / "raw_manifest.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in manifest.items()
               if hashlib.sha256((DATA / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest]
    assert not changed, f"raw NASA files were modified (restore them from git): {changed}"


def test_experimental_tables_are_comma_separated():
    for p in (DATA / "experimental_tables").glob("*.csv"):
        header = p.read_bytes().split(b"\n", 1)[0]
        assert header.count(b",") > header.count(b";"), f"{p.name}: separator changed (opened and saved in Excel?)"
