from pathlib import Path

import lithiumscope.datasets.provisioner as provisioner


def test_dataset_inspection_does_not_download(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        provisioner,
        "DATA_DIR",
        tmp_path / "data",
    )
    monkeypatch.setattr(
        provisioner,
        "PROJECT_ROOT",
        tmp_path,
    )

    statuses = provisioner.inspect_required_datasets()

    assert len(statuses) == 2
    assert all(
        status.ready is False
        for status in statuses
    )
