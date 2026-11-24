import numpy as np
import pandas as pd
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame


class Estimator:
    def predict(self, frame):
        return np.full(len(frame), 10.0)


def test_prediction_does_not_call_padding_generated_data():
    schema = FeatureSchema(["Na2O", "K2O", "MgO", "Alkali_Sum"], [], 2)
    raw = pd.DataFrame({"Na2O": [3.], "K2O": [4.]})
    bundle = {"schema": schema, "algorithm": "dummy", "estimator": Estimator(), "applicability_profile": {}}
    out, audit = predict_model_1_frame(raw, bundle=bundle)
    assert audit["generated_during_preparation"] == ["Alkali_Sum"]
    assert audit["missing_expected_columns"] == ["MgO"]
    assert audit["prepared_missing_counts"]["MgO"] == 1
    assert out.applicability_warning.tolist() == ["INSUFFICIENT_PROFILE"]
    assert out.out_of_training_range_fraction.isna().all()


def test_incomplete_or_invalid_profile_is_not_certified():
    schema = FeatureSchema(["Na2O", "K2O"], [], 2)
    raw = pd.DataFrame({"Na2O": [3.], "K2O": [4.]})
    bundle = {"schema": schema, "algorithm": "dummy", "estimator": Estimator(),
              "applicability_profile": {"Na2O": {"q01": 0., "q99": 10.}}}
    out, _ = predict_model_1_frame(raw, bundle=bundle)
    assert out.applicability_warning.iloc[0] == "INSUFFICIENT_PROFILE"
    bundle["applicability_profile"]["K2O"] = {"q01": 0., "q99": 10.}
    out, _ = predict_model_1_frame(raw, bundle=bundle)
    assert out.applicability_warning.iloc[0] == "OK"


def test_historical_import_paths_remain_joblib_compatible():
    import pickle
    import importlib
    module = importlib.import_module("lithiumscope.model_1.steps.step_08_preprocessing")
    assert module.FeatureSchema is FeatureSchema
    # Protocol 0 GLOBAL resolves the original qualified class name.
    assert pickle.loads(b"clithiumscope.model_1.steps.step_08_preprocessing\nFeatureSchema\n.") is FeatureSchema
    from lithiumscope.model_1.training.algorithms.tabnet import TabNetRegressorAdapter
    assert pickle.loads(b"clithiumscope.model_1.training.tabnet\nTabNetRegressorAdapter\n.") is TabNetRegressorAdapter
