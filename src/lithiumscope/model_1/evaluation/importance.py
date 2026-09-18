from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _importance_values(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "get_feature_importance"):
        try:
            return np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            return None
    return None


def save_feature_importance(estimator, destination_table: Path, destination_figure: Path, top_n: int = 25) -> tuple[Path, Path] | None:
    try:
        preprocessor = estimator.named_steps["preprocess"]
        model = estimator.named_steps["model"]
        names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
        values = _importance_values(model)
        if values is None or len(values) != len(names):
            return None
        table = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
        destination_table.parent.mkdir(parents=True, exist_ok=True)
        destination_figure.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination_table, index=False)
        top = table.head(top_n).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["feature"], top["importance"])
        ax.set_title(f"Top {top_n} importancias — modelo ganador")
        ax.set_xlabel("Importancia")
        fig.tight_layout()
        fig.savefig(destination_figure, dpi=160)
        plt.close(fig)
        return destination_table, destination_figure
    except Exception:
        return None
