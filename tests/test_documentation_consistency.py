"""Regression checks for the published benchmark suite and software metadata."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RAW = {
    "msls-val", "tokyo24/7", "gardens-point",
    "amstertime", "spedtest", "17-places",
}
EXPECTED_DISPLAY = (
    "MSLS, Tokyo 24/7, Gardens Point, AmsterTime, SPEDTest, and 17-Places"
)
DOC_PATHS = (
    "README.md", "docs/RESULTS.md", "docs/DATA_SOURCES.md",
    "docs/METHODOLOGY.md",
)


def test_evaluation_cohorts_match_documentation():
    path = ROOT / "results/original_evaluation/final_dataset_effects_and_discordance_v1.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for k in {"50", "100"}:
        names = {row["dataset"] for row in rows if row["k"] == k}
        assert names == EXPECTED_RAW, (k, names)
    for relpath in DOC_PATHS:
        content = (ROOT / relpath).read_text(encoding="utf-8")
        assert EXPECTED_DISPLAY in content, relpath
        for incorrect in ("Pittsburgh30k", "Oxford RobotCar", "VPAir"):
            assert incorrect not in content, (relpath, incorrect)


def test_release_citation_and_public_package_metadata():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "10.5281/zenodo.23882077" in readme
    assert "once registered" not in readme
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.1"' in metadata
    assert "internal staging package" not in metadata
