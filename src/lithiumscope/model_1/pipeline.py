"""Model 1 execution entry points and environment/source resolution."""
from __future__ import annotations


def run_training(device, *, dataset=None):
    """M1 owns environment/source resolution and delegates scientific competition."""
    from lithiumscope.runtime.preboot import require_full_training_environment
    from lithiumscope.runtime.dataset_provisioning import require_model_1_dataset
    from lithiumscope.model_1.training.competition import run_model_1_competition
    require_full_training_environment("model_1")
    source = dataset if dataset is not None else require_model_1_dataset(include_georoc=True)
    return run_model_1_competition(source, device)


def run_independent_evaluation(phase, root, **kwargs):
    """Separate registered holdout phases; no automatic activation."""
    from lithiumscope.evaluation.holdout.workflow import run_holdout_phase
    return run_holdout_phase("model_1",phase,root,**kwargs)
