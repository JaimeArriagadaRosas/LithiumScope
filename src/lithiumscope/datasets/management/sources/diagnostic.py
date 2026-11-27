"""Explicit diagnostic source acquisition, separate from read-only preflight."""
from __future__ import annotations

from pathlib import Path


def prepare_diagnostic_source(model_group, *, acquire=False):
    from lithiumscope.datasets.management.sources.local import local_training_source
    try:
        return local_training_source(model_group)
    except FileNotFoundError:
        if not acquire:
            raise
    from lithiumscope.core.config import load_config
    config=load_config(model_group)
    if model_group=="model_1":
        georoc=config["data_sources"].get("georoc",{})
        if georoc.get("enabled") and georoc.get("frozen_combined_sha256"):
            raise FileNotFoundError("Frozen M1 snapshot must be restored; no reconstruction")
        from lithiumscope.runtime.dataset_provisioning import require_model_1_dataset
        return require_model_1_dataset()
    if model_group=="model_2":
        training=config["training"]
        if training.get("expansion",{}).get("enabled"):
            from lithiumscope.runtime.dataset_provisioning import require_model_2_expanded_manifest
            return require_model_2_expanded_manifest()
        if training.get("frozen_manifest_sha256"):
            raise FileNotFoundError("Frozen M2 manifest must be restored; no replacement")
        from lithiumscope.runtime.dataset_provisioning import require_model_2_dataset
        return require_model_2_dataset()
    raise ValueError("Unknown model")
