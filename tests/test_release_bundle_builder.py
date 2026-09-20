import json
from pathlib import Path
import zipfile

import joblib

from lithiumscope.core.hashing import file_sha256
import lithiumscope.results.release as release_module


def _artifact(root: Path, model_group: str, run_id: str) -> Path:
    directory = root / model_group / "trained" / run_id
    directory.mkdir(parents=True)
    model_path = directory / "model.joblib"
    joblib.dump({"model_group": model_group}, model_path)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "model_group": model_group,
                "run_id": run_id,
                "algorithm": "dummy",
                "model_sha256": file_sha256(model_path),
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_build_release_bundle_contains_only_manifest_and_model_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    models = tmp_path / "models"
    results = tmp_path / "results"
    monkeypatch.setattr(release_module, "MODELS_DIR", models)
    monkeypatch.setattr(release_module, "RESULTS_DIR", results)

    _artifact(models, "model_1", "training_a")
    _artifact(models, "model_2", "training_b")

    manifest_dir = results / "release_candidates"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "candidate.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suggested_tag": "lithiumscope-training_20260919_191124_m0300",
                "model_1": {"run_id": "training_a"},
                "model_2": {"run_id": "training_b"},
            }
        ),
        encoding="utf-8",
    )

    archive = release_module.build_release_bundle(
        manifest_path
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())

    assert "release_manifest.json" in names
    assert (
        "model_1/training_a/model.joblib"
        in names
    )
    assert (
        "model_2/training_b/model.joblib"
        in names
    )
    assert not any(
        name.startswith("src/")
        for name in names
    )
