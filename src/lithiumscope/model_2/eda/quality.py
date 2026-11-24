"""Attach frozen patch QA evidence without removing outer test observations."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training.v4_inputs import spectral_v4_enabled, spectral_v4_paths, combine_v4_quality_mask


def patch_quality(path: Path, config: dict, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = config.get("quality_training", {})
    if not cfg.get("enabled", False):
        return pd.DataFrame(index=manifest.index), {"enabled": False}
    inventory_path = PROJECT_ROOT / str(cfg["inventory_path"])
    mask, audit = training_quality_mask(path, inventory_path=inventory_path,
                                        minimum_joint_valid=float(cfg.get("minimum_joint_valid_fraction", .95)))
    inventory = pd.read_csv(inventory_path).iloc[:len(manifest)].reset_index(drop=True)
    columns = ["sample_id", "image_sha256", "sixband_joint_valid_fraction", "minimum_band_valid_fraction",
               "center_joint_valid_fraction", "has_shared_nodata_mask", "source_cohort"]
    result = inventory[[name for name in columns if name in inventory]].copy()
    if spectral_v4_enabled(config):
        _, _, red_path = spectral_v4_paths(path, config)
        mask, audit = combine_v4_quality_mask(mask, path, config)
        red = pd.read_csv(red_path)
        if not np.array_equal(red["sample_id"].astype(str), result["sample_id"].astype(str)):
            raise ValueError("Red-edge quality rows do not align")
        if "original_patch_sha256" in red and not np.array_equal(red["original_patch_sha256"], result["image_sha256"]):
            raise ValueError("Red-edge original patch hash differs from verified six-band inventory")
        result["rededge_joint_valid_fraction"] = red["rededge_joint_valid_fraction"]
    result["train_eligible"] = mask
    return result, audit
