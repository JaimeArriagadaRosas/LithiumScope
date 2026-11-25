import numpy as np
import pandas as pd
import pytest

from lithiumscope.prediction.analysis import ensure_case_ids, concordance_table
from lithiumscope.prediction.analysis.model_selection_evidence import gather_selection_evidence, M1_RUNS, M2_RUNS


def test_duplicate_identifiers_do_not_collide_with_existing_suffixes():
    frame = pd.DataFrame({"case_id": ["a", "a", "a__2", "a", "a__3"]})
    ids = ensure_case_ids(frame)["case_id"]
    assert ids.is_unique
    assert ids.iloc[0] == "a"
    assert ids.iloc[2] == "a__2"
    assert ids.iloc[4] == "a__3"


def test_undefined_predictions_do_not_become_negative_concordance():
    frame = pd.DataFrame({"Li_icpms_predicted": [np.nan, 5., -1., 20.],
                          "prospectivity_score": [.3, np.inf, .2, .8]})
    result = concordance_table(frame, lithium_threshold_ppm=10.)
    assert result["concordance"].tolist() == ["no_comparable"]*3 + ["concordante_alta"]
    with pytest.raises(ValueError, match="threshold"):
        concordance_table(frame, lithium_threshold_ppm=10., model_2_probability_threshold=np.nan)


def test_annex_is_optional_for_other_pairs_and_missing_archive(tmp_path):
    result = gather_selection_evidence({"run_id": "new_m1"}, {"run_id": "new_m2"}, root=tmp_path)
    assert result["state"] == "NOT_APPLICABLE_TO_SELECTED_PAIR"
    result = gather_selection_evidence({"run_id": M1_RUNS[2]}, {"run_id": M2_RUNS[0]}, root=tmp_path)
    assert result["state"] == "HISTORICAL_EVIDENCE_UNAVAILABLE"
    assert not result["model_2_external_independent_holdout"]
