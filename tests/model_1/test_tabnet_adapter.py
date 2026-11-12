from lithiumscope.model_1.training.tabnet import TabNetRegressorAdapter


def test_tabnet_adapter_exposes_internal_validation_and_final_refit_flags():
    estimator = TabNetRegressorAdapter(
        seed=42,
        max_epochs=100,
        patience=10,
        validation_fraction=0.2,
        refit_full=True,
    )

    params = estimator.get_params()

    assert params["validation_fraction"] == 0.2
    assert params["refit_full"] is True
