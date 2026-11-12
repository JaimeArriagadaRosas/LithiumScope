"""Separated install_release_bundle responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.distribution.installer import InstallationResult, ModelRelease, Path

def install_release_bundle(
    archive: Path,
    release: ModelRelease,
    *,
    asset_sha256: str,
) -> InstallationResult:
    from lithiumscope.distribution.installer import InstallationResult, MODELS_DIR, ModelRelease, Path, _discover_artifacts, _prepare_installation, _release_manifest, _safe_extract, _validate_deserialization, _validate_manifest_against_artifacts, datetime, json, os, set_active_models, shutil, tempfile, timezone
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".release_extract_",
        dir=MODELS_DIR,
    ) as temporary:
        extracted = Path(temporary)
        _safe_extract(
            archive,
            extracted,
        )
        manifest = _release_manifest(
            extracted,
            release,
        )
        artifacts = _discover_artifacts(
            extracted
        )
        _validate_manifest_against_artifacts(
            manifest,
            artifacts,
        )
        _validate_deserialization(
            artifacts
        )

        prepared = [
            _prepare_installation(
                model_group,
                artifact_dir,
                metadata,
            )
            for model_group, (
                artifact_dir,
                metadata,
            ) in sorted(
                artifacts.items()
            )
        ]

        committed: list[Path] = []
        try:
            for _, staging, destination in prepared:
                if staging is None:
                    continue
                os.replace(
                    staging,
                    destination,
                )
                committed.append(
                    destination
                )
        except Exception:
            for destination in reversed(
                committed
            ):
                shutil.rmtree(
                    destination,
                    ignore_errors=True,
                )
            for _, staging, _ in prepared:
                if (
                    staging is not None
                    and staging.exists()
                ):
                    shutil.rmtree(
                        staging,
                        ignore_errors=True,
                    )
            raise

    installed = tuple(
        item
        for item, _, _ in prepared
    )
    active_paths = {
        item.model_group: item.model_path
        for item in installed
    }
    set_active_models(
        active_paths,
        source={
            "type": "github_release",
            "tag": release.tag_name,
            "asset": release.asset_name,
            "release_url": release.html_url,
            "asset_sha256": asset_sha256,
        },
    )

    receipt = {
        "schema_version": 1,
        "release_tag": release.tag_name,
        "release_name": release.release_name,
        "release_url": release.html_url,
        "asset_name": release.asset_name,
        "asset_sha256": asset_sha256,
        "installed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "models": {
            item.model_group: {
                "run_id": item.run_id,
                "algorithm": item.algorithm,
                "model_path": str(
                    item.model_path
                ),
                "model_sha256": (
                    item.model_sha256
                ),
            }
            for item in installed
        },
    }
    receipt_path = (
        MODELS_DIR
        / "installed_release.json"
    )
    temporary_receipt = (
        receipt_path.with_suffix(
            ".json.tmp"
        )
    )
    temporary_receipt.write_text(
        json.dumps(
            receipt,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_receipt.replace(
        receipt_path
    )

    return InstallationResult(
        release_tag=release.tag_name,
        asset_name=release.asset_name,
        asset_sha256=asset_sha256,
        models=installed,
        receipt_path=receipt_path,
    )
