"""Deterministic factorial designs for tuning-budget and preprocessing comparisons."""
from __future__ import annotations

from lithiumscope.experiments.shared.paired_suite import Arm


def budget_arms(algorithms, x, *, eligible=None, seeds=(42,43,44), attempts=(20,100)):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid experiment seeds")
    if len(attempts) != 2 or any(type(n) is not int or n < 1 for n in attempts) or attempts[0] >= attempts[1]:
        raise ValueError("Expected two increasing optimization attempt budgets")
    arms = []
    for algorithm in algorithms:
        for seed in seeds:
            reference = f"{algorithm}_seed{seed}_{attempts[0]}_no_pruning"
            for count in attempts:
                for pruning in (False,True):
                    name = f"{algorithm}_seed{seed}_{count}_{'pruning' if pruning else 'no_pruning'}"
                    arms.append(Arm(name, algorithm, None if name == reference else reference, x,
                                    {"attempts":count,"pruning":pruning,"seed":seed}, eligible))
    return arms


def choose_algorithms(configured, requested):
    if not requested:
        return tuple(configured)
    if len(set(requested)) != len(requested) or not set(requested) <= set(configured):
        raise ValueError("Requested algorithm not configured or duplicated")
    return tuple(requested)
