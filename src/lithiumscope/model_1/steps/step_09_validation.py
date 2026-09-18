from __future__ import annotations

from sklearn.model_selection import KFold


def nested_cv(
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_seed: int = 42,
) -> tuple[KFold, KFold]:
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=random_seed)
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    return outer, inner
