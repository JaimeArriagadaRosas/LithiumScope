"""Explicit phase boundary for prospective evaluation; never expose labels during scoring."""
from __future__ import annotations
from pathlib import Path


def run_holdout_phase(model_group, phase, root, **kwargs):
    if model_group not in {"model_1","model_2"}:
        raise ValueError("Unknown model group")
    root=Path(root)
    from lithiumscope.evaluation.holdout.registry import register, verified_registration
    if phase=="register":
        if "model_group" in kwargs or "labels_path" in kwargs:
            raise ValueError("Registration accepts only the custodian's label commitment")
        return register(root,model_group=model_group,**kwargs)
    if phase not in {"score","evaluate"}:
        raise ValueError("Unknown prospective phase")
    record,_=verified_registration(root)
    if record["model_group"]!=model_group:
        raise ValueError("Registered holdout belongs to another model")
    if phase=="score":
        if "labels_path" in kwargs:
            raise ValueError("Scoring must not receive labels")
        from lithiumscope.evaluation.holdout.scoring import score_registered
        return score_registered(root,**kwargs)
    from lithiumscope.evaluation.holdout.evaluation import evaluate_registered
    return evaluate_registered(root,**kwargs)
