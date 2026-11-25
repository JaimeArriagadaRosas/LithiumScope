import pandas as pd
import pytest

from lithiumscope.core.pipeline_audit import PipelineAudit


def test_invalid_exclusions_do_not_partially_mutate_lineage():
    audit = PipelineAudit(dataset_sha256="a"*64)
    raw = pd.DataFrame({"source_dataset": ["s"]*3, "Li_icpms": [1., 2., 3.]})
    audit.capture("raw", raw)
    with pytest.raises(ValueError, match="empty exclusion reason"):
        audit.capture("filtered", raw.iloc[[2]], exclusion_reasons=[
            {"row_index": 0, "reason": "invalid"}, {"row_index": 1, "reason": ""}])
    assert len(audit.records) == 1
    assert not audit.exclusions
    audit.capture("filtered", raw.iloc[[2]], exclusion_reasons=[
        {"row_index": 0, "reason": "invalid"}, {"row_index": 1, "reason": "invalid"}])
    assert len(audit.exclusions) == 2
    assert len({r["source_row_id"] for r in audit.exclusions}) == 2


def test_hash_failure_keeps_previous_stage(monkeypatch):
    from lithiumscope.core import pipeline_audit as module
    audit = PipelineAudit(dataset_sha256="a"*64)
    raw = pd.DataFrame({"Li_icpms": [1., 2.]})
    audit.capture("raw", raw)
    def fail(_):
        raise ValueError("cannot hash")
    monkeypatch.setattr(module, "_digest", fail)
    with pytest.raises(ValueError, match="cannot hash"):
        audit.capture("next", raw.iloc[[1]], exclusion_reasons=[{"row_index": 0, "reason": "invalid"}])
    assert len(audit.records) == 1 and not audit.exclusions
