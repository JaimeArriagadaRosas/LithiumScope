import pandas as pd

from lithiumscope.datasets.manifest import build_tabular_manifest, write_manifest


def test_dataset_manifest_captures_hash_and_shape(tmp_path):
    path = tmp_path / "dataset.csv"
    frame = pd.DataFrame({"Li_icpms": [1.0, 2.0], "SiO2": [50.0, 51.0]})
    frame.to_csv(path, index=False)

    manifest = build_tabular_manifest(
        path,
        dataset_name="test",
        model_group="model_1",
        stage="test",
        frame=frame,
    )
    destination = write_manifest(manifest, tmp_path / "manifest.json")

    assert len(manifest.source_sha256) == 64
    assert manifest.rows == 2
    assert manifest.columns == 2
    assert destination.exists()
