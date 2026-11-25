"""Explicit local runner for both six-family learning-curve diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.local_sources import local_training_source
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.experiments.shared.curve_suite import run_curve_suite


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


def classification_inputs(preflight_directory: Path):
    from lithiumscope.model_2.training.execution.competition_prepare_scientific_data import _prepare_scientific_data
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
    _, frame, groups, _, _, mask, folds, _, x, _, _ = _prepare_scientific_data(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("model_1", "model_2"))
    parser.add_argument("--execute", action="store_true", help="Fit diagnostic curves; never save a model.")
    parser.add_argument("--prefer-gpu", action="store_true")
    args = parser.parse_args()
    run = RESULTS_DIR / "diagnostics" / "grouped_learning_curves" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:12])
    if args.model == "model_1":
        inputs = regression_inputs()
    else:
        inputs = classification_inputs(run / "preparation")
    device = detect_device(prefer_gpu=args.prefer_gpu)
    fit = inputs.pop("fit_builder")(device)
    report = run_curve_suite(**inputs, device=device, fit=fit, output=run / "curves", execute=args.execute)
    print(f"Curvas {args.model}: {report['state']} — {run}")


if __name__ == "__main__":
    main()
