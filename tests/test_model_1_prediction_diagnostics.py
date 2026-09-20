import pandas as pd

import lithiumscope.model_1.prediction.predictor as predictor


class _Schema:
    numeric = ("a", "derived")
    categorical = ()


class _Estimator:
    def predict(self, frame):
        return [1.0] * len(frame)


def test_prediction_diagnostics_distinguish_generated_features(monkeypatch):
    bundle = {
        "schema": _Schema(),
        "algorithm": "dummy",
        "estimator": _Estimator(),
        "applicability_profile": {},
    }

    def fake_prepare(frame, algorithm, schema):
        output = frame.copy()
        output["derived"] = output["a"] * 2
        return output

    monkeypatch.setattr(
        predictor,
        "prepare_prediction_data",
        fake_prepare,
    )
    monkeypatch.setattr(
        predictor,
        "applicability_fraction",
        lambda frame, profile: pd.Series([0.0] * len(frame)),
    )

    _, diagnostics = predictor.predict_model_1_frame(
        pd.DataFrame({"a": [2.0]}),
        bundle=bundle,
    )

    assert diagnostics["missing_raw_input_columns"] == ["derived"]
    assert diagnostics["generated_during_preparation"] == ["derived"]
    assert diagnostics["missing_expected_columns"] == []
