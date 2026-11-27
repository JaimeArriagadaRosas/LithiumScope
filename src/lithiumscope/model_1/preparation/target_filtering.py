from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.validation.chemistry.concentration import valid_concentration

logger = get_logger("model_1.target_filtering")


from lithiumscope.datasets.validation.tabular.schema import resolve_target as resolve_target


def filter_target(
    frame: pd.DataFrame,
    candidates: list[str],
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
    *,
    zero_policy: str = "retain",
) -> tuple[pd.DataFrame, str]:
    """Resolve Li and remove only invalid/missing targets.

    Quantiles are descriptive diagnostics only. They never decide which
    observations enter cross-validation.
    """
    target = resolve_target(frame, candidates)
    result = frame.copy()
    numeric, mask, audit = valid_concentration(result[target], zero_policy=zero_policy)
    exclusions = []
    for idx in result.index[~mask]:
        val = numeric.loc[idx]
        reason = "missing_or_unparseable" if pd.isna(val) else "nonfinite" if not np.isfinite(val) else "negative" if val < 0 else "zero_policy"
        exclusions.append({"row_index": str(idx), "column": target, "original": str(result.loc[idx, target]), "reason": reason})
    result[target] = numeric
    before = len(result)
    result = result.loc[mask].copy()
    result.attrs["target_validation"] = audit
    result.attrs["target_exclusions"] = exclusions
    logger.info("Physical target validation: %s", audit)
    low = float(result[target].quantile(lower_quantile))
    high = float(result[target].quantile(upper_quantile))

    logger.info(
        "Target preparation %s: initial=%d valid=%d diagnostic_q%.3f=%.4f "
        "diagnostic_q%.3f=%.4f trimmed=0",
        target,
        before,
        len(result),
        lower_quantile,
        low,
        upper_quantile,
        high,
    )
    return result, target
