from types import SimpleNamespace

import pandas as pd

from lithiumscope.model_1.validation.observation_audit import audit_prepared_observations


def test_m1_source_audit_retains_historical_cohort_and_flags_conflicts():
    frame = pd.DataFrame({"source_dataset": ["a", "a", "b"], "Sample": ["A", "A", "A"],
                          "Li_icpms": [10., 90., 10.], "SiO2": [60., 60., 60.],
                          "Longitude": [-70.] * 3, "Latitude": [-20.] * 3})
    original = frame.copy()
    prepared = SimpleNamespace(frame=frame, target="Li_icpms",
                               schema=SimpleNamespace(numeric=["SiO2"], categorical=[]))
    audit, summary = audit_prepared_observations(prepared)
    assert summary["review_conflicts"] == 2
    assert summary["cross_source_coordinate_matches"] == 3
    assert not summary["cohort_mutated"]
    assert audit.decision.tolist() == ["quarantine", "quarantine", "retain"]
    pd.testing.assert_frame_equal(frame, original)
