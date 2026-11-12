"""Isolated responsibility with unchanged original API."""
from __future__ import annotations

def _active_artifact_release_entry(
    model_group: str,
) -> dict | None:
    from lithiumscope.results.release import active_model_path, active_release_candidate, json, read_active_models
    catalog_entry = active_release_candidate(
        model_group
    )
    if catalog_entry is not None:
        return {
            **catalog_entry,
            "release_provenance": "local_run_catalog",
            "reused_from_previous_release": False,
        }

    model_path = active_model_path(
        model_group
    )
    if model_path is None:
        return None
    metadata_path = (
        model_path.parent / "metadata.json"
    )
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    registry = read_active_models()
    registry_entry = (
        registry.get("models", {})
        .get(model_group, {})
    )
    source = (
        registry_entry.get("source", {})
        if isinstance(
            registry_entry,
            dict,
        )
        else {}
    )
    return {
        "model_group": model_group,
        "run_id": str(
            metadata.get(
                "run_id",
                model_path.parent.name,
            )
        ),
        "winner": metadata.get(
            "variant_id",
            metadata.get("algorithm"),
        ),
        "algorithm": metadata.get("algorithm"),
        "dataset_sha256": metadata.get(
            "dataset_sha256"
        ),
        "git_commit": metadata.get(
            "git_commit"
        ),
        "model_sha256": metadata.get(
            "model_sha256"
        ),
        "release_candidate": True,
        "release_provenance": "active_installed_artifact",
        "reused_from_previous_release": (
            source.get("type")
            == "github_release"
        ),
        "source_release": source.get("tag"),
        "source_release_asset": source.get(
            "asset"
        ),
    }
