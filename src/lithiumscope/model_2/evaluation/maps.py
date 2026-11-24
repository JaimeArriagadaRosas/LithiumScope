from __future__ import annotations

from pathlib import Path

from lithiumscope.core.visualization import configure_headless_matplotlib

configure_headless_matplotlib()

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

    if isinstance(probabilities, pd.DataFrame):
        manifest = manifest.merge(probabilities[["sample_id", "prospectivity_score"]],
                                  on="sample_id", validate="one_to_one", how="left")
        scores = manifest["prospectivity_score"].to_numpy(dtype=float)
        if len(probabilities) != len(manifest) or not np.isfinite(scores).all():
            raise ValueError("OOF map identities do not match manifest")
    else:
        scores = np.asarray(probabilities, dtype=float)
    if len(manifest) != len(scores):
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        manifest["longitude"],
        manifest["latitude"],
        c=scores,
        alpha=0.75, vmin=0, vmax=1,
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(title)
    fig.colorbar(points, ax=ax, label="Score de prospectividad OOF")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination
