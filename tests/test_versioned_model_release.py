import json
from pathlib import Path
import zipfile

import joblib

import lithiumscope.distribution.installer as installer
from lithiumscope.core.hashing import file_sha256
from lithiumscope.distribution.github_releases import (
    ModelRelease,
    parse_model_releases,
)
import lithiumscope.persistence.active_models as active_models
from lithiumscope.persistence.load_model import latest_model_path
import lithiumscope.persistence.load_model as load_model_module


def test_release_parser_only_selects_training_assets():
    payload = [
        {
            "tag_name": "lithiumscope-training_20260919_191124_m0300",
            "name": "training",
            "draft": False,
            "published_at": "2026-09-20T15:00:00Z",
            "html_url": "https://example.invalid/release",
            "assets": [
                {
                    "name": "lithiumscope-training_20260919_191124_m0300-artifacts.zip",
                    "browser_download_url": "https://example.invalid/model.zip",
                    "size": 123,
                    "digest": "sha256:abc",
                }
            ],
        },
        {
            "tag_name": "v1",
            "name": "product version",
            "draft": False,
            "assets": [
                {
                    "name": "v1-artifacts.zip",
                    "browser_download_url": "https://example.invalid/v1.zip",
                    "size": 10,
                }
            ],
        },
    ]

    releases = parse_model_releases(
        payload,
        asset_suffix="-artifacts.zip",
        tag_prefix="lithiumscope-training_",
    )

    assert len(releases) == 1
    assert releases[0].tag_name.endswith("_m0300")


def test_active_model_is_preferred(monkeypatch, tmp_path: Path):
    project = tmp_path
    models = project / "models"
    active_path = models / "active_models.json"
    trained = models / "model_1" / "trained"
    older = trained / "older" / "model.joblib"
    selected = trained / "selected" / "model.joblib"
    older.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    selected.write_bytes(b"selected")

    monkeypatch.setattr(active_models, "PROJECT_ROOT", project)
    monkeypatch.setattr(active_models, "MODELS_DIR", models)
    monkeypatch.setattr(active_models, "ACTIVE_MODELS_PATH", active_path)
    monkeypatch.setattr(load_model_module, "MODELS_DIR", models)

    active_models.set_active_model(
        "model_1",
        selected,
        source={"type": "test"},
    )

    assert latest_model_path("model_1") == selected


def _write_artifact(
    root: Path,
    model_group: str,
    run_id: str,
) -> Path:
    directory = root / run_id
    directory.mkdir(parents=True)
    model_path = directory / "model.joblib"
    joblib.dump(
        {
            "algorithm": "dummy",
            "model_group": model_group,
        },
        model_path,
    )
    metadata = {
        "model_group": model_group,
        "algorithm": "dummy",
        "run_id": run_id,
        "model_sha256": file_sha256(model_path),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return directory


def test_installer_accepts_legacy_timestamp_manifest(monkeypatch, tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    m1 = _write_artifact(source, "model_1", "training_a")
    m2 = _write_artifact(source, "model_2", "training_b")
    tag = "lithiumscope-training_20260919_191124_m0300"
    manifest = {
        "schema_version": 1,
        "suggested_tag": tag,
        "model_1": {"run_id": "training_a"},
        "model_2": {"run_id": "training_b"},
    }
    (source / "20260920T150618Z.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.write(
            source / "20260920T150618Z.json",
            arcname="20260920T150618Z.json",
        )
        for directory in (m1, m2):
            for path in directory.rglob("*"):
                if path.is_file():
                    bundle.write(
                        path,
                        arcname=str(
                            Path(directory.name)
                            / path.relative_to(directory)
                        ),
                    )

    installed_root = tmp_path / "installed_models"
    monkeypatch.setattr(
        installer,
        "MODELS_DIR",
        installed_root,
    )
    monkeypatch.setattr(
        active_models,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        active_models,
        "MODELS_DIR",
        installed_root,
    )
    monkeypatch.setattr(
        active_models,
        "ACTIVE_MODELS_PATH",
        installed_root / "active_models.json",
    )

    release = ModelRelease(
        tag_name=tag,
        release_name=tag,
        published_at=None,
        html_url="https://example.invalid/release",
        asset_name="models-artifacts.zip",
        asset_url="https://example.invalid/models.zip",
        asset_size=archive.stat().st_size,
        asset_digest=None,
    )
    result = installer.install_release_bundle(
        archive,
        release,
        asset_sha256=file_sha256(archive),
    )

    assert {
        item.model_group
        for item in result.models
    } == {"model_1", "model_2"}
    assert (
        installed_root
        / "model_1"
        / "trained"
        / "training_a"
        / "model.joblib"
    ).is_file()
    assert (
        installed_root
        / "model_2"
        / "trained"
        / "training_b"
        / "model.joblib"
    ).is_file()
    registry = json.loads(
        (
            installed_root
            / "active_models.json"
        ).read_text(encoding="utf-8")
    )
    assert set(registry["models"]) == {
        "model_1",
        "model_2",
    }


def test_safe_extract_blocks_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "bad")

    destination = tmp_path / "extract"
    destination.mkdir()

    try:
        installer._safe_extract(
            archive,
            destination,
        )
    except RuntimeError as exc:
        assert "ruta insegura" in str(exc)
    else:
        raise AssertionError(
            "Unsafe ZIP path should have been rejected"
        )
