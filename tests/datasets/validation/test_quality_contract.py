import json

import numpy as np
import pandas as pd
import pytest

from lithiumscope.datasets.validation.contracts import QualityPolicy, candidate_policy
from lithiumscope.datasets.validation.quality_report import assess_quality, write_quality_report
from lithiumscope.datasets.validation.chemistry.composition import assess_composition
from lithiumscope.datasets.validation.spatial.precision import assess_precision


def test_quality_report_partitions_without_mutating_source(tmp_path):
    frame = pd.DataFrame({"Li_icpms": [5000., -1., 20., 10.], "Latitude": [-30.]*4,
                          "Longitude": [-70.]*4, "source_li_unit": ["ppm", "ppm", "unknown", "ppm"],
                          "source_li_method": ["ICP-MS"]*4, "SiO2": [50., 50., 50., 110.]}, index=[5, 6, 7, 8])
    original = frame.copy(deep=True)
    report = assess_quality(frame, candidate_policy("model_1"))
    assert report.decisions.decision.tolist() == ["admitted", "excluded", "quarantine", "quarantine"]
    assert sum(map(len, report.partition(frame).values())) == len(frame)
    pd.testing.assert_frame_equal(frame, original)
    data = json.loads(write_quality_report(report, tmp_path).read_text(encoding="utf-8"))
    assert data["decisions"] == {"admitted": 1, "quarantine": 2, "excluded": 1}
    assert data["scientific_independence_certified"] is False


def test_precision_does_not_certify_unknown_or_reversed_bounds():
    frame = pd.DataFrame({"Latitude": [-30., -30., -30.], "Latitude_min": [-31., -29., np.nan],
                          "Latitude_max": [-29., -31., np.nan], "precision_m": [5., 5., np.nan]})
    result = assess_precision(frame, max_uncertainty_m=10.)
    assert result.precision_status.tolist() == ["reported", "bounds_review", "unknown"]


def test_composition_never_sums_partial_measurements():
    frame = pd.DataFrame({"SiO2": [60., 60., 101.], "Al2O3": [20., np.nan, 20.]})
    result = assess_composition(frame, ("SiO2", "Al2O3"))
    assert result.composition_status.tolist() == ["complete", "incomplete", "invalid_component"]
    assert result.oxide_total_wt_pct.iloc[0] == 80.
    assert result.oxide_total_wt_pct.iloc[1:].isna().all()


def test_m2_precision_and_crs_are_not_assumed():
    frame = pd.DataFrame({"Li_icpms": [1.], "Latitude": [-30.], "Longitude": [-70.],
                          "image_path": ["a.npy"], "source_li_unit": ["ppm"]})
    report = assess_quality(frame, candidate_policy("model_2"))
    assert report.decisions.decision.iloc[0] == "quarantine"
    assert {"unknown_precision", "unverified_crs"} <= set(report.decisions.reason.iloc[0])
    with pytest.raises(ValueError):
        QualityPolicy("model_1", predictor_columns=("Li_icpms",))
