import json

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.data.checkpoint_contract import acquisition_signature, reconcile_checkpoint
from lithiumscope.model_2.preparation.patch_contract import make_patch_contract


def test_reconcile_checkpoint_updates_labels_and_rejects_removed_corrupt_and_incompatible(tmp_path):
    imagery = {"bands": ["red", "green", "blue", "nir", "swir16", "swir22"],
               "patch_pixels": 32, "patch_size_m": 640, "datetime": "2020/2021"}
    samples = pd.DataFrame({"sample_id": ["current", "bad"], "Li_icpms": [99., 5.],
                            "latitude": [-20., -20.], "longitude": [-70., -70.],
                            "spatial_group": ["new", "new"]})
    rows = []
    for sid in ("current", "bad", "removed"):
        path = tmp_path / (sid + ".npy")
        np.save(path, np.ones((6, 32, 32), dtype=np.float32))
        scl = path.with_suffix(".scl.npy")
        np.save(scl, np.full((32, 32), 5.))
        meta = {**make_patch_contract(), "scl_sha256": file_sha256(scl),
                "acquisition_geometry": "registered_metric_grid_v2",
                "scientific_patch_qa": {"scientifically_accepted_proxy": True}, "sample_id": sid, "latitude": -20., "longitude": -70.,
                "scene_id": "scene", "image_sha256": file_sha256(path),
                "acquisition_signature": acquisition_signature(imagery)}
        path.with_suffix(".json").write_text(json.dumps(meta))
        rows.append({"sample_id": sid, "Li_icpms": 1., "latitude": -20., "longitude": -70.,
                     "spatial_group": "old", "image_path": str(path), "sentinel_scene_id": "scene"})
        if sid == "bad":
            path.write_bytes(b"corrupt")
    checkpoint = pd.DataFrame(rows)
    ready, rejected = reconcile_checkpoint(checkpoint, samples, imagery, root=tmp_path)
    assert ready.sample_id.tolist() == ["current"]
    assert ready.Li_icpms.tolist() == [99.]
    assert ready.spatial_group.tolist() == ["new"]
    assert len(rejected) == 2
    ready, rejected = reconcile_checkpoint(checkpoint, samples, {**imagery, "datetime": "2022/2023"}, root=tmp_path)
    assert ready.empty and len(rejected) == 3
