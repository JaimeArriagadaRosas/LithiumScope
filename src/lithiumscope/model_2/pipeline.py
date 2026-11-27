"""Model 2 training and independent evaluation entry points."""
from lithiumscope.core.config import load_config


def run_training(device, *, manifest=None):
    from lithiumscope.runtime.preboot import require_full_training_environment
    from lithiumscope.runtime.dataset_provisioning import require_model_2_expanded_manifest
    from lithiumscope.model_2.training.prospective import train_with_prospective_evidence

    require_full_training_environment("model_2")
    source = manifest if manifest is not None else require_model_2_expanded_manifest()
    return train_with_prospective_evidence(source, device, load_config("model_2"))


def run_independent_evaluation(phase, root, **kwargs):
    """Separate registered holdout phases; no automatic activation."""
    from lithiumscope.evaluation.holdout.workflow import run_holdout_phase
    return run_holdout_phase("model_2",phase,root,**kwargs)
