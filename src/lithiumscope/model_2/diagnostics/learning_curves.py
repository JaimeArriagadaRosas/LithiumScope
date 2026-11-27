"""Model-specific learning curve inputs using canonical development partitions."""
from __future__ import annotations

from lithiumscope.core.config import load_config
from lithiumscope.datasets.management.sources.local import local_training_source
from pathlib import Path
from types import SimpleNamespace


def classification_inputs(preflight_directory: Path):
    from lithiumscope.model_2.training.execution.scientific_data import prepare_scientific_data
    from lithiumscope.model_2.training.calibration.operating import fit_policy
    from lithiumscope.model_2.training.calibration.policy import calibration_policy

    config = load_config("model_2")
    source = local_training_source("model_2")
    model = config["model"]
    training = config["training"]
    validation = config["validation"]
    target = training["lithium_column"]
    group = training["spatial_group_column"]
    seed = int(model["random_seed"])
    # Reuse canonical preparation, including the frozen 706/783 outer split rule.
    preflight_directory.mkdir(parents=True, exist_ok=False)
    context = SimpleNamespace(manifests=preflight_directory)
    _, frame, groups, _, _, mask, folds, _, x, _, _ = prepare_scientific_data(
        config, context, model.get("fixed_threshold_ppm"), target, source,
        float(model["high_lithium_quantile"]), int(validation["folds"]), seed, group)
    mode = calibration_policy(config["optimization"])
    def build_fit(device):
        def fit(algorithm, fold, train, labels, train_groups):
            return fit_policy(algorithm, device, seed + fold, {}, train, labels, train_groups,
                              int(validation["inner_folds"]), mode)
        return fit
    return {"x": x, "y": frame[target], "groups": groups,
            "splits": [(f.train_idx, f.test_idx) for f in folds], "kind": "classification",
            "eligible": mask, "endpoints": [f.label_threshold_ppm for f in folds],
            "algorithms": model["algorithms"],
            "policy": {"configuration": config, "hyperparameters": "predeclared_factory_defaults",
                       "calibration": mode, "score_threshold": .5, "optimization_performed": False},
            "fit_builder": build_fit, "predict": lambda model, x: model.predict_proba(x)[:, 1]}
