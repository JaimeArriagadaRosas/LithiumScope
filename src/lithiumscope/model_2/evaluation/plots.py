from __future__ import annotations

from pathlib import Path

from lithiumscope.core.visualization import configure_headless_matplotlib

configure_headless_matplotlib()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_probability_histogram(y_true, probabilities, destination: Path, title: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y": y_true, "p": probabilities})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame.loc[frame.y == 0, "p"], bins=20, alpha=0.6, label="Referencia baja/media")
    ax.hist(frame.loc[frame.y == 1, "p"], bins=20, alpha=0.6, label="Referencia alta")
    ax.set_xlabel("Score de prospectividad")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_roc_pr(y_true, probabilities, destination: Path, title: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axes[0])
    axes[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axes[1])
    axes[1].set_title("Precision-Recall")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def save_competition_chart(table: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("roc_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered["label"], ordered["roc_auc_mean"], xerr=ordered["roc_auc_std"])
    ax.set_xlabel("ROC-AUC CV (mayor es mejor)")
    ax.set_title("Competencia Modelo 2")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
