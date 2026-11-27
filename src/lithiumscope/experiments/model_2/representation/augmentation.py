"""Historical pixel-domain research perturbation and feature extraction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from lithiumscope.model_2.preparation.imagery import load_imagery
from lithiumscope.model_2.preparation.image_preprocessing import preprocess_image
from lithiumscope.model_2.preparation.features.spectral import extract_spectral_features

def augment_image_training_only(
    cube: np.ndarray, *, seed: int = 42, brightness: float = 1.04,
    noise_fraction: float = 0.02, missing_fraction: float = 0.03,
) -> np.ndarray:
    """Small, bounded pixel-domain perturbation, not a substitute for SCL masks."""
    if not 0.0 < brightness <= 1.25:
        raise ValueError("Unexpected brightness factor")
    if not 0 <= noise_fraction <= 0.1 or not 0 <= missing_fraction <= 0.1:
        raise ValueError("Invalid augmentation intensity")
    original = np.asarray(cube, dtype=np.float32)
    if original.ndim != 3 or original.shape[0] != 6:
        raise ValueError("Expected 6-band Sentinel-2 patch")
    rng = np.random.default_rng(seed)
    value = original.copy()
    std = np.nanstd(value, axis=(1, 2), keepdims=True)
    scale = np.maximum(std, 0.005) * noise_fraction
    value = (value * brightness + rng.normal(
        0, 1, size=value.shape
    ).astype(np.float32) * scale).astype(np.float32)
    value = np.clip(value, 0, 1.6)
    value[rng.random(value.shape) < missing_fraction] = np.nan
    return value

def make_augmentation_table(
    manifest: pd.DataFrame,
    bundle: dict,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    features = list(bundle["features"])
    names = tuple(bundle["band_names"])
    rows = []
    for idx, item in enumerate(manifest.itertuples(index=False)):
        image_path = Path(item.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = load_imagery(image_path)
        if image.shape[0] != len(names):
            raise ValueError("Sentinel band mismatch: "+str(image_path))
        disturbed = augment_image_training_only(image, seed=seed+idx)
        clean = preprocess_image(
            disturbed, normalize_per_band=bool(bundle.get("normalize_per_band",False))
        )
        result = extract_spectral_features(clean, band_names=names)
        rows.append(result)
        if (idx+1)%100==0:
            print("M2_AUG_FEATURES",idx+1,"/",len(manifest),flush=True)
    table=pd.DataFrame(rows).reindex(columns=features)
    if table.isna().any().any() or not np.isfinite(table.to_numpy(dtype=float)).all():
        raise RuntimeError("Augmented training features are not all finite.")
    return table
