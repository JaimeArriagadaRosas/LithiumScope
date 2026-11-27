import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.data import feature_cache as cache


def fixture(tmp_path):
    image = tmp_path / "patch.npy"
    np.save(image, np.ones((1, 2, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["a"], "image_path": [str(image)],
                  "Li_icpms": [5.0], "spatial_group": ["g"]}).to_csv(manifest, index=False)
    config = {"training": {"lithium_column": "Li_icpms"}, "imagery": {"bands": ["red"]}}
    return image, manifest, config


def test_cache_detects_tampering_and_source_drift(tmp_path):
    image, manifest, cfg = fixture(tmp_path)
    source, identity = cache.source_identity(manifest, cfg)
    frame = pd.DataFrame([{name: "g" if name == "spatial_group" else 5.0 for name in identity["columns"]}])
    path = tmp_path / "features.csv"
    cache.write_cache(path, frame, source, cfg, identity)
    assert cache.read_cache(path, source, cfg, identity) is not None
    path.write_text("broken")
    assert cache.read_cache(path, source, cfg, identity) is None
    cache.write_cache(path, frame, source, cfg, identity)
    np.save(image, np.full((1, 2, 2), 2, dtype=np.float32))
    _, new_identity = cache.source_identity(manifest, cfg)
    assert new_identity != identity
    assert cache.read_cache(path, source, cfg, new_identity) is None
    cfg["imagery"]["normalize_per_band"] = True
    assert cache.source_identity(manifest, cfg)[1] != new_identity


def test_cache_rejects_missing_or_duplicate_id_and_target(tmp_path):
    _, manifest, cfg = fixture(tmp_path)
    frame = pd.read_csv(manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        cache.validate_manifest(pd.concat([frame, frame]), cfg)
    frame.loc[0, "Li_icpms"] = -1
    with pytest.raises(ValueError, match="concentration"):
        cache.validate_manifest(frame, cfg)


def test_cache_rejects_incomplete_pair(tmp_path):
    _, manifest, cfg = fixture(tmp_path)
    source, identity = cache.source_identity(manifest, cfg)
    path = tmp_path / "features.csv"
    path.write_text("uncommitted")
    assert cache.read_cache(path, source, cfg, identity) is None
