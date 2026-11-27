"""Integrated entry points preserve source evidence before inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.prediction.execution import complete, session


def test_single_image_snapshot_keeps_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "RESULTS_DIR", tmp_path / "results")
    model_1 = tmp_path / "samples.csv"
    pd.DataFrame({"case_id": ["a"], "SiO2": [50.0]}).to_csv(model_1, index=False)
    model_2 = tmp_path / "image.npy"
    np.save(model_2, np.ones((6, 32, 32), dtype=np.float32))
    sidecar = model_2.with_suffix(".npy.json")
    sidecar.write_text(json.dumps({"scene_id": "example"}), encoding="utf-8")
    monkeypatch.setattr(complete, "_run_integrated", lambda **kwargs: kwargs)
    result = complete.run_complete_prediction(model_1, model_2)
    root = result["session"].root
    copied = Path(result["model_2_cases"].iloc[0]["image_path"])
    assert copied != model_2 and copied.is_relative_to(root)
    assert file_sha256(copied) == file_sha256(model_2)
    assert copied.with_suffix(".npy.json").read_bytes() == sidecar.read_bytes()
    info = result["input_info"]["model_2"]
    assert info["snapshot_sha256"] == file_sha256(copied)
    assert info["sidecars"][copied.with_suffix(".npy.json").relative_to(root).as_posix()] == file_sha256(sidecar)


def test_manifest_snapshot_resolves_original_relative_path(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "RESULTS_DIR", tmp_path / "results")
    model_1 = tmp_path / "samples.csv"
    pd.DataFrame({"case_id": ["a"], "SiO2": [50.0]}).to_csv(model_1, index=False)
    image = tmp_path / "image.npy"
    np.save(image, np.ones((6, 32, 32), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"case_id": ["a"], "image_path": ["image.npy"]}).to_csv(manifest, index=False)
    real_snapshot = complete.snapshot_input

    def snapshot(source, destination):
        digest = real_snapshot(source, destination)
        if source == manifest:
            pd.DataFrame({"case_id": ["changed"], "image_path": ["missing.npy"]}).to_csv(source, index=False)
        return digest

    monkeypatch.setattr(complete, "snapshot_input", snapshot)
    monkeypatch.setattr(complete, "_run_integrated", lambda **kwargs: kwargs)
    result = complete.run_complete_prediction(model_1, manifest)
    assert result["model_1_input"]["SiO2"].tolist() == [50.0]
    cases = result["model_2_cases"]
    assert cases.iloc[0]["case_id"] == "a"
    assert Path(cases.iloc[0]["image_path"]) == image
    assert cases.iloc[0]["sentinel_status"] == "ready"
