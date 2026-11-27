"""Source adaptation results; adaptation does not certify training eligibility."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class AdaptedSource:
    frame: pd.DataFrame
    evidence: dict

    def __post_init__(self):
        if self.frame.empty or not self.frame.columns.is_unique:
            raise ValueError("Adapted source must have nonempty, unambiguous data")
