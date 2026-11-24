"""Version-aware inference inputs. Never substitute absent spectral descriptors."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.preparation.imagery import load_imagery
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES
from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame


class SpectralAbstention(ValueError):
    def __init__(self, status: str, reason: str, **details):
        super().__init__(reason)
        self.status, self.details = status, details


def _source_metadata(path: Path) -> dict:
    for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if sidecar.is_file():
            try:
                value = json.loads(sidecar.read_text(encoding="utf-8"))
                if value.get("collection") != "sentinel-2-l2a" or tuple(value.get("bands", [])) != DEFAULT_BAND_NAMES:
                    break
                lat, lon = float(value["latitude"]), float(value["longitude"])
                scene = value["scene_id"]
                if not (-90 <= lat <= 90 and -180 <= lon <= 180 and isinstance(scene, str) and scene.strip()):
                    break
                expected = value.get("image_sha256") or value.get("patch_sha256")
                if expected and expected != file_sha256(path):
                    break
                return value
            except (ValueError, KeyError, TypeError, OSError):
                break
    raise SpectralAbstention("ABSTAIN_MISSING_SCENE_PROVENANCE",
                             "V4 requires same-scene Sentinel-2 provenance and coordinates.")


def prepare_spectral_input(path: Path, bundle: dict) -> tuple[pd.DataFrame, dict]:
    version = int(bundle.get("feature_extractor_version", 3))
    if version not in {1, 2, 3, 4}:
        raise SpectralAbstention("ABSTAIN_UNSUPPORTED_EXTRACTOR", "Unsupported spectral extractor version.")
    raw = load_imagery(path)
    bands = tuple(bundle.get("band_names", bundle.get("base_band_names", DEFAULT_BAND_NAMES)))
    if raw.shape[0] != len(bands) or raw.shape[1] < 2 or raw.shape[2] < 2:
        raise SpectralAbstention("ABSTAIN_INVALID_IMAGE_SCHEMA", "Input patch does not match the trained band schema.")
    coverage = float(np.isfinite(raw).all(axis=0).mean())
    cutoff = bundle.get("quality_min_joint_valid")
    if not np.isfinite(raw).any(axis=(1, 2)).all() or (cutoff is not None and coverage < float(cutoff)):
        raise SpectralAbstention("ABSTAIN_LOW_SPECTRAL_COVERAGE", "No rank: insufficient valid Sentinel-2 pixel coverage.",
                                 joint_valid_fraction=coverage, required_joint_valid_fraction=cutoff)
    expected = list(bundle["features"])
    if not expected or len(expected) != len(set(expected)):
        raise SpectralAbstention("ABSTAIN_INVALID_FEATURE_SCHEMA", "Invalid or duplicate model feature contract.")
    frame = image_to_feature_frame(path, band_names=bands,
                                   normalize_per_band=bool(bundle.get("normalize_per_band", False)))
    diagnostics = {"joint_valid_fraction": coverage, "feature_extractor_version": version}
    if version == 4:
        from lithiumscope.tools.acquisition.sentinel2.rededge import cached_rededge_sample
        from lithiumscope.model_2.preparation.features.rededge import append_rededge_features

        if bands != DEFAULT_BAND_NAMES or len(expected) != 170:
            raise SpectralAbstention("ABSTAIN_INVALID_FEATURE_SCHEMA", "V4 requires the 126 + 44 feature contract.")
        metadata = _source_metadata(path)
        try:
            red, evidence = cached_rededge_sample(
                SimpleNamespace(sample_id=f"inference_{file_sha256(path)[:24]}_{metadata['scene_id']}",
                                sentinel_scene_id=metadata["scene_id"], latitude=metadata["latitude"],
                                longitude=metadata["longitude"], image_path=str(path)),
                output=RESULTS_DIR / "predictions" / "spectral_cache_v4", min_joint_valid=.05)
        except Exception as exc:
            raise SpectralAbstention("ABSTAIN_REDEDGE_UNAVAILABLE", str(exc)) from exc
        diagnostics.update(evidence)
        required = .95 if cutoff is None else float(cutoff)
        if evidence["rededge_joint_valid_fraction"] < required:
            raise SpectralAbstention("ABSTAIN_LOW_REDEDGE_COVERAGE", "Red-edge QA below required coverage.", **diagnostics)
        frame = pd.DataFrame([append_rededge_features(frame.iloc[0].to_dict(), red)])
    missing = [name for name in expected if name not in frame]
    if missing:
        raise SpectralAbstention("ABSTAIN_MISSING_FEATURES", "Expected spectral descriptors are absent.", missing_features=missing)
    if not np.isfinite(frame[expected].to_numpy(float)).all():
        raise SpectralAbstention("ABSTAIN_NONFINITE_FEATURES", "Spectral predictors are not finite.")
    return frame[expected], diagnostics
