import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.prediction.predictor import score_model_2_image
from lithiumscope.prediction.integrated_score_model_2_cases import _score_model_2_cases
from lithiumscope.model_2.preparation.features.spectral import extract_spectral_features


class NeverFit:
    def predict_proba(self, frame):
        pytest.fail("Invalid input must never reach estimator")


def test_missing_spectral_features_abstain_in_both_routes(tmp_path):
    p = tmp_path / "patch.npy"
    np.save(p, np.ones((6, 4, 4), dtype=np.float32))
    bundle = {"features": ["unavailable"], "estimator": NeverFit()}
    payload, detail = score_model_2_image(p, bundle=bundle)
    assert payload["prospectivity_score"] is None
    assert detail["missing_features"] == ["unavailable"]
    result, audit = _score_model_2_cases(pd.DataFrame({"case_id": ["a"], "image_path": [str(p)]}), bundle)
    assert result.empty
    assert audit["failed_cases"] == 1 and audit["missing_feature_counts"] == {"unavailable": 1}


def test_missing_profile_is_insufficient_not_ok(tmp_path):
    p = tmp_path / "patch.npy"
    np.save(p, np.ones((6, 4, 4), dtype=np.float32))
    class Score:
        def predict_proba(self, frame):
            return np.array([[.2, .8]])
    bundle = {"features": list(extract_spectral_features(np.ones((6, 4, 4)))),
              "estimator": Score()}
    result, _ = score_model_2_image(p, bundle=bundle)
    assert result["prospectivity_score"] == .8
    assert result["applicability_warning"] == "INSUFFICIENT_PROFILE"


def test_v4_without_provenance_never_scores(tmp_path):
    from lithiumscope.model_2.preparation.features.rededge import extract_rededge_v4
    p = tmp_path / "patch.npy"
    np.save(p, np.ones((6, 4, 4), dtype=np.float32))
    features = list(extract_spectral_features(np.ones((6, 4, 4)))) + list(extract_rededge_v4(np.ones((4, 4, 4)))[0])
    result, detail = score_model_2_image(p, bundle={"feature_extractor_version": 4, "features": features, "estimator": NeverFit()})
    assert result["quality_status"] == "ABSTAIN_MISSING_SCENE_PROVENANCE"
