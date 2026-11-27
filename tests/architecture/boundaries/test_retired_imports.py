from pathlib import Path

from scripts.check_internal_imports import find_obsolete_imports


def test_no_retired_dataset_or_analysis_imports():
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    findings = find_obsolete_imports(root)
    assert not findings, "\n" + "\n".join(findings)
