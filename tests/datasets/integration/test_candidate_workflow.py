import json
import pandas as pd
import pytest

from lithiumscope.datasets.management.candidates import audit_candidate, build_georoc_candidates
from lithiumscope.datasets.management.snapshots import verify_snapshot
from lithiumscope.datasets.validation.contracts import candidate_policy


def test_candidate_workflow_preserves_source_and_all_decisions(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"Li_icpms": [10., -1., 0.], "Latitude": [-30.]*3, "Longitude": [-70.]*3,
                  "source_li_unit": ["ppm"]*3, "source_li_method": ["ICP-MS"]*3}).to_csv(source, index=False)
    original = source.read_bytes()
    result = audit_candidate(source, root=tmp_path / "versions", policy=candidate_policy("model_1"))
    assert source.read_bytes() == original
    manifest = verify_snapshot(result.directory)
    assert set(manifest["artifacts"]) == {"dataset.csv", "quality_report.json", "admitted.csv", "quarantine.csv", "excluded.csv"}
    report = json.loads((result.directory / "quality_report.json").read_text(encoding="utf-8"))
    assert report["decisions"] == {"admitted": 1, "quarantine": 1, "excluded": 1}
    assert audit_candidate(source, root=tmp_path / "versions", policy=candidate_policy("model_1")).reused
    (result.directory / "quality_report.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify_snapshot(result.directory)


def test_georoc_candidates_preserve_separate_sources(tmp_path):
    paths = [tmp_path / name for name in ("a.csv", "b.csv")]
    for path in paths:
        path.write_text("MATERIAL,LI(PPM),LONGITUDE,LATITUDE\nWR,10,-70,-30\n")
    result = build_georoc_candidates(paths, root=tmp_path / "cache",
                policy=candidate_policy("model_1"), minimum_predictors=0)
    frame = pd.read_csv(result.data_path)
    assert len(frame) == 2 and frame.audit_observation_id.is_unique
    assert set(frame.source_file) == {"a.csv", "b.csv"}
    assert len(pd.read_csv(result.directory / "quarantine.csv")) == 2
