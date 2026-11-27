"""Read-only diagnostic resolution with explicit injected acquisition authorization."""
from __future__ import annotations

from lithiumscope.core.logging.session import get_logger


def prepare_diagnostic_source(model_group, *, acquire=False, acquire_source=None):
    from lithiumscope.datasets.management.sources.local import local_training_source
    logger = get_logger("datasets.management.sources.diagnostic")
    try:
        return local_training_source(model_group)
    except FileNotFoundError:
        if not acquire:
            logger.info("Diagnostic acquisition not authorized model=%s", model_group)
            raise
    from lithiumscope.core.config import load_config
    config = load_config(model_group)
    if model_group == "model_1":
        georoc = config["data_sources"].get("georoc", {})
        if georoc.get("enabled") and georoc.get("frozen_combined_sha256"):
            raise FileNotFoundError("Frozen M1 snapshot must be restored; no reconstruction")
    elif model_group == "model_2":
        training = config["training"]
        if not training.get("expansion", {}).get("enabled") and training.get("frozen_manifest_sha256"):
            raise FileNotFoundError("Frozen M2 manifest must be restored; no replacement")
    else:
        raise ValueError("Unknown model")
    if acquire_source is None:
        raise RuntimeError("Authorized acquisition requires an explicit provisioning callback")
    logger.info("Diagnostic acquisition authorized model=%s", model_group)
    acquire_source(model_group)
    # Verify the configured source, not an arbitrary path returned by a provider.
    return local_training_source(model_group)
