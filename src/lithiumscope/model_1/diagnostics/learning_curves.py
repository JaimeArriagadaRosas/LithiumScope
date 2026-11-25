"""Model-specific learning curve inputs using canonical development partitions."""
from __future__ import annotations

from lithiumscope.core.config import load_config
from lithiumscope.datasets.management.local_sources import local_training_source


def regression_inputs():
    from lithiumscope.model_1.pipeline import prepare_training_data
    from lithiumscope.model_1.validation.spatial import materialize_regression_splits
    from lithiumscope.model_1.training.factory import build_pipeline
    from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator

    config = load_config("model_1")
    source = local_training_source("model_1")
    data = prepare_training_data(source)
    validation = config["validation"]
    seed = int(validation["random_seed"])
    splits, _ = materialize_regression_splits(data.x, data.y, data.groups,
                                               int(validation["outer_folds"]), seed, require_groups=True)
    def build_fit(device):
        def fit(algorithm, fold, x, y, groups):
            estimator = build_pipeline(algorithm, device, seed + fold, data.schema, {},
                                       final_fit=False, target_transform="identity")
            return fit_regression_estimator(estimator, x, y, groups)
        return fit
    return {"x": data.x, "y": data.y, "groups": data.groups, "splits": splits,
            "kind": "regression", "algorithms": config["model"]["algorithms"],
            "policy": {"configuration": config, "hyperparameters": "predeclared_factory_defaults",
                       "target_transform": "identity", "optimization_performed": False},
            "fit_builder": build_fit, "predict": lambda model, x: model.predict(x)}
