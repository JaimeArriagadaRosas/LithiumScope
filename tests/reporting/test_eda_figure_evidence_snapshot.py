"""The report may only copy verified descriptive evidence."""
from pathlib import Path

import pytest
from PIL import Image

from lithiumscope.core.hashing import file_sha256
from lithiumscope.reporting.figures import eda_figure_snapshot as snapshot


def figure(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return file_sha256(path)


def evidence_fixture(root):
    folder = root / "results/model_1/eda/run_a"
    digest = figure(folder / "one.png", "blue")
    research = root / "results/model_1/candidate_eda/source_a/run_b"
    digest_research = figure(research / "two.png", "green")
    return folder, {
        "state": "CURRENT_SELECTION_EVIDENCE",
        "models": {"model_1": {"eda": {"run_id": "run_a", "figures": {"one.png": digest}}},
                   "model_2": {}},
        "research_sources": [{"source": "source_a", "run_id": "run_b",
                              "figures": {"two.png": digest_research}}],
    }


def test_verified_eda_and_research_figures_are_copied(tmp_path):
    folder, evidence = evidence_fixture(tmp_path)
    destination = tmp_path / "snapshot"
    source_hash = file_sha256(folder / "one.png")
    copied = snapshot.snapshot_eda_figures(evidence, destination, root=tmp_path)
    assert {path.name for path in copied} == {"eda_model_1_one.png", "research_source_a_two.png"}
    assert evidence["figure_snapshot"] == {path.name: file_sha256(path) for path in copied}
    assert file_sha256(folder / "one.png") == source_hash


def test_unverified_missing_and_outside_figures_are_not_copied(tmp_path):
    folder, evidence = evidence_fixture(tmp_path)
    wrong = figure(folder / "wrong.png", "red")
    outside = figure(folder.parent / "outside.png", "yellow")
    evidence["models"]["model_1"]["eda"]["figures"].update({
        "wrong.png": "0" * 64, "missing.png": wrong, "../outside.png": outside,
    })
    copied = snapshot.snapshot_eda_figures(evidence, tmp_path / "snapshot", root=tmp_path)
    assert len(copied) == 2
    assert not any("wrong" in path.name or "outside" in path.name for path in copied)
    assert snapshot.snapshot_eda_figures({"state": "HISTORICAL_COMPARISON"},
                                         tmp_path / "unused", root=tmp_path) == []


def test_copy_drift_is_rejected_and_corrupted_target_removed(tmp_path, monkeypatch):
    folder, evidence = evidence_fixture(tmp_path)
    def altered_copy(source, target):
        Path(target).write_bytes(b"corrupted")
    monkeypatch.setattr(snapshot.shutil, "copyfile", altered_copy)
    destination = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="changed during snapshot"):
        snapshot.snapshot_eda_figures(evidence, destination, root=tmp_path)
    assert not list(destination.iterdir())
