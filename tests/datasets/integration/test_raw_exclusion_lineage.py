import json
import pandas as pd

from lithiumscope.datasets.management.candidates import build_georoc_candidates
from lithiumscope.datasets.management.snapshots import verify_snapshot
from lithiumscope.datasets.validation.contracts import QualityPolicy, candidate_policy
from lithiumscope.datasets.validation.quality_report import assess_quality


def test_raw_exclusions_have_explicit_reasons_and_positions(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("MATERIAL,LI(PPM),LONGITUDE,LATITUDE\nWR,10,-70,-30\nGL,11,-70,-30\nWR,-1,-70,-30\nWR,12,-70,-100\n")
    root = tmp_path / "cache"
    result = build_georoc_candidates([source], root=root, policy=candidate_policy("model_1"), minimum_predictors=0)
    assert len(pd.read_csv(result.data_path)) == 1
    folder = next(p for p in (root / "sources").iterdir() if (p / "manifest.json").is_file())
    verify_snapshot(folder)
    excluded = pd.read_csv(folder / "raw_exclusions.csv")
    assert excluded.source_row_position.tolist() == [1, 2, 3]
    assert excluded.reason.tolist() == ["incompatible_material", "invalid_lithium", "invalid_coordinates"]
    audit = json.loads((folder / "ingestion_audit.json").read_text(encoding="utf-8"))
    assert sum(audit["rejection_counts"].values()) == 3


def test_identity_conflicts_are_quarantined_not_deleted():
    frame = pd.DataFrame({"Li_icpms": [10., 11., 12.], "source": ["a", "a", "b"], "sample": ["x", "x", "x"]})
    report = assess_quality(frame, QualityPolicy("model_1", require_coordinates=False,
                                                identity_columns=("source", "sample")))
    assert report.decisions.decision.tolist() == ["quarantine", "quarantine", "admitted"]
