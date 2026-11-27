import json
import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.sources.model_2 import require_expanded_manifest
from lithiumscope.datasets.validation.chemistry.external_geochemistry import review_independence


def test_expanded_manifest_is_bound_to_evidence(tmp_path):
    path = tmp_path / "expanded.csv"
    frame = pd.DataFrame({"sample_id": ["a", "b"], "Li_icpms": [10, 20],
                          "spatial_group": ["g1", "g2"], "image_path": ["a.npy", "b.npy"]})
    frame.to_csv(path, index=False)
    settings = {"expansion": {"output_manifest_path": "expanded.csv", "state_dir": "."}}
    evidence = tmp_path / "expansion_summary.json"
    evidence.write_text(json.dumps({"new_manifest_sha256": file_sha256(path), "total": 2}))
    assert require_expanded_manifest(settings, tmp_path) == path
    frame.loc[1, "Li_icpms"] = 21
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="evidence mismatch"):
        require_expanded_manifest(settings, tmp_path)


def test_independence_review_never_certifies_absence_of_overlap():
    reference = pd.DataFrame({"source": ["s"], "sample_id": ["a"], "latitude": [-30.],
                              "longitude": [-70.], "publication_id": ["doi:1"]})
    candidates = pd.DataFrame({"source": ["s", "t", "t"], "sample_id": ["a", "b", "c"],
                               "latitude": [-30., -30., -10.], "longitude": [-70., -70., -60.],
                               "publication_id": ["doi:1", "doi:2", "doi:3"],
                               "sample_date": ["2020-01-01"]*3})
    result = review_independence(candidates, reference, spatial_buffer_km=1)
    assert result.state.tolist() == ["known_overlap", "review_required", "no_overlap_detected"]
    assert not result.independence_certified.any()
