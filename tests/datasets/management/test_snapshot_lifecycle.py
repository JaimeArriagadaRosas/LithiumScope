from pathlib import Path

import pytest

from lithiumscope.datasets.management.snapshots import (
    dataset_implementation_revision, list_snapshots, materialize_snapshot, verify_snapshot,
)


def test_snapshot_inventory_and_explicit_recovery(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Li\n10\n")
    root = tmp_path / "versions"
    def transform(inputs, output):
        output.write_bytes(inputs["source"].read_bytes())
    def run(**kwargs):
        return materialize_snapshot(root, {"source": source}, policy={}, transform_revision="test",
                                    transform=transform, **kwargs)
    first = run()
    assert list_snapshots(root)[0]["state"] == "verified"
    first.data_path.write_text("corrupted")
    assert list_snapshots(root)[0]["state"] == "corrupted"
    with pytest.raises(ValueError, match="integrity"):
        run()
    repaired = run(recover_corrupt=True)
    assert repaired.version_id == first.version_id
    assert not repaired.reused
    assert verify_snapshot(repaired.directory)
    assert next((root / ".quarantine").iterdir()).joinpath("dataset.csv").read_text() == "corrupted"
    assert len(list_snapshots(root)) == 1


def test_implementation_fingerprint_detects_code_changes_not_location(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "module.py").write_text("x = 1\n")
    first = dataset_implementation_revision("test", dataset_root=a)
    assert first == dataset_implementation_revision("test", dataset_root=b)
    (b / "module.py").write_text("x = 2\n")
    assert first != dataset_implementation_revision("test", dataset_root=b)
