"""Tests of evidence gates: no automatic scientific promotion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.model_2.spatial.leakage_control import threshold
from lithiumscope.experiments.data_lab.source_studies.georoc.screening.mamani_overlap import near_full_mamani


def test_threshold_uses_only_supplied_inner_examples():
    assert 0 <= threshold(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8])) <= 1


def test_geochemical_collocation_does_not_certify_independence():
    mamani = pd.DataFrame({
        "Sample": ["X"] * 800,
        "Latitude (Y)": [-22.0] * 800,
        "Longitude (X)": [-68.0] * 800,
        "Li_icpms": [15.0] * 800,
        "Geochemistry Reference": ["test"] * 800,
    })
    candidates = pd.DataFrame({"latitude": [-22.], "longitude": [-68.], "li_ppm": [15.]})
    result = near_full_mamani(candidates, mamani)
    assert bool(result.possible_same_coordinate_li.iloc[0])
    assert not bool(result.independent_ground_truth_approved.iloc[0])


def test_label_and_group_never_used_as_features():
    code = (next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "src/lithiumscope/experiments/model_2/spatial/leakage_control.py").read_text(encoding="utf-8")
    assert 'drop(columns=["Li_icpms","spatial_group"]' in code


@pytest.mark.local_artifacts
@pytest.mark.m2
@pytest.mark.training
def test_cv_artifact_has_no_external_accuracy_claim():
    report = Path("results/scientific_simulations/leakage_safe_m2/experiment_manifest.json")
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["rows"] == 706
        assert data["folds"] == 5
        assert data["external_accuracy"] is None
