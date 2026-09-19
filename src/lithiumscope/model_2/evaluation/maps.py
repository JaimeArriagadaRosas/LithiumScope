from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_spatial_score_map(
    manifest_path: Path,
    probabilities,
    destination: Path,
    title: str,
) -> Path | None:
    manifest = pd.read_csv(manifest_path)
    required = {"longitude", "latitude"}
    if not required.issubset(manifest.columns):
        return None

    scores = np.asarray(probabilities, dtype=float)
    if len(manifest) != len(scores):
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        manifest["longitude"],
        manifest["latitude"],
        c=scores,
        alpha=0.75,
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(title)
    fig.colorbar(points, ax=ax, label="Score de prospectividad OOF")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
