"""Training-only grouped calibration partitions; no random fallback."""
import numpy as np
from sklearn.model_selection import GroupKFold

def calibration_splits(x, y, groups, n_splits: int = 3):
    if groups.nunique() < n_splits:
        return None
    splits = list(GroupKFold(n_splits=n_splits).split(x, groups=groups))
    for tr, va in splits:
        if len(np.unique(np.asarray(y)[tr])) < 2 or len(np.unique(np.asarray(y)[va])) < 2:
            return None
        if set(groups.iloc[tr]).intersection(set(groups.iloc[va])):
            raise RuntimeError("Group leakage during probability calibration")
    return splits
