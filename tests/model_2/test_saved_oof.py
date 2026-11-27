"""New training cohorts do not depend on a historical dataset size."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.evaluation.saved_oof import load_predictions


def test_new_cohort_with_three_folds(tmp_path):
    n = 12
    manifest = pd.DataFrame({"sample_id": [f"S{i}" for i in range(n)],
                             "Li_icpms": np.arange(n, dtype=float)})
    df = pd.DataFrame({"sample_id": manifest.sample_id, "li_ppm": manifest.Li_icpms,
                       "outer_fold": np.repeat([1, 2, 3], 4),
                       "y_true": np.arange(n) % 2,
                       "prospectivity_score": np.linspace(.1, .9, n)})
    path = tmp_path / "oof_predictions_new.csv"
    df.to_csv(path, index=False)
    scores, folds, truth = load_predictions(manifest, tmp_path)
    assert len(scores["new"]) == n and set(folds) == {1, 2, 3}
    assert np.array_equal(truth, df.y_true)
    df.loc[0, "sample_id"] = "different"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sample identity"):
        load_predictions(manifest, tmp_path)


@pytest.mark.parametrize("column,value", [("outer_fold", 1.5), ("y_true", .5),
                                           ("prospectivity_score", 1.1)])
def test_invalid_oof_values_rejected(tmp_path, column, value):
    df = pd.DataFrame({"li_ppm": [1., 2., 3.], "outer_fold": [1., 2., 3.],
                       "y_true": [0., 1., 0.], "prospectivity_score": [.1, .8, .2]})
    df.loc[0, column] = value
    df.to_csv(tmp_path / "oof_predictions_new.csv", index=False)
    with pytest.raises(ValueError, match="Invalid OOF"):
        load_predictions(pd.DataFrame({"Li_icpms": [1., 2., 3.]}), tmp_path)
