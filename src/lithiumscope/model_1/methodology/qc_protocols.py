"""M1 summaries of shared geochemical quality protocols."""
from __future__ import annotations

import pandas as pd
from lithiumscope.datasets.validation.chemistry.qc_protocols import protocol_masks

__all__ = ["protocol_masks", "protocol_summary"]

def protocol_summary(frame: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    records = []
    source = frame.get(
        "source_dataset", pd.Series("unknown", index=frame.index)
    ).astype(str)
    for label, idx in source.groupby(source).groups.items():
        for protocol in ("A_original", "B_reconstructed", "C_experimental"):
            chosen = masks.loc[idx, protocol]
            records.append({
                "source": label,
                "protocol": protocol,
                "rows": int(chosen.sum()),
                "total": len(idx),
                "additional_vs_A": int(chosen.sum() - masks.loc[idx, "A_original"].sum()),
                "quality_certified": protocol != "C_experimental",
            })
    return pd.DataFrame(records)
