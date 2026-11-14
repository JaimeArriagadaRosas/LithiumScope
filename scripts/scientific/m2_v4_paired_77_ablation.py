"""Paired v4 706 vs 783 using ONLY the same historic 706 outer TEST rows.

The 783 arm includes 77 provisional GEOROC rows in training, and carries
different historical v3-tuned fold parameters. NOT a single-factor causal test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT
from scripts.scientific.m2_rededge_v4_competition import ARMS
from scripts.scientific.m2_rededge_v4_report import paired_group_ci


def compare(*, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    base = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    a = base / "comparison_706"
    b = base / "comparison_783"
    for folder in (a, b):
        status = folder / "experiment_manifest.json"
        if (
            not status.is_file()
            or json.loads(status.read_text(encoding="utf-8")).get("status") != "complete_experimental_development_only"
        ):
            raise FileNotFoundError("Both six-algorithm fixed M2 v4 runs must be complete")
    historical = pd.read_csv(root / "data/processed/model_2/training_manifest.csv")
    expanded = pd.read_csv(root / "data/processed/model_2/vnext_expanded/training_manifest.csv")
    if len(historical) != 706 or len(expanded) != 783:
        raise ValueError("Unexpected M2 cohort sizes")
    if not np.array_equal(historical.sample_id.astype(str), expanded.sample_id.iloc[:706].astype(str)):
        raise ValueError("First 706 candidate identities drifted")
    q706 = pd.read_csv(
        root / "results/model_2/runs/training_20261113_133905_m0300/tables/oof_predictions_random_forest.csv"
    )
    q783 = pd.read_csv(
        root / "results/model_2/runs/training_20261113_114607_m0300/tables/oof_predictions_random_forest.csv"
    ).iloc[:706]
    if not np.array_equal(q706.y_true, q783.y_true) or not np.array_equal(q706.outer_fold, q783.outer_fold):
        raise ValueError("Historical test labels/folds are not paired")
    labels = q706.y_true.to_numpy(int)
    groups = historical.spatial_group.to_numpy(str)
    records = []
    for alg in ARMS:
        s706 = np.load(a / f"oof_{alg}.npy", allow_pickle=False)
        s783 = np.load(b / f"oof_{alg}.npy", allow_pickle=False)
        if len(s706) != 706 or len(s783) != 783:
            raise ValueError("Incomplete OOF predictors")
        s783 = s783[:706]
        m706 = {
            "auc": roc_auc_score(labels, s706),
            "ap": average_precision_score(labels, s706),
            "brier": brier_score_loss(labels, s706),
        }
        m783 = {
            "auc": roc_auc_score(labels, s783),
            "ap": average_precision_score(labels, s783),
            "brier": brier_score_loss(labels, s783),
        }
        pair = paired_group_ci(labels, s783, s706, groups, n=400)
        records.append(
            {
                "algorithm": alg,
                **{f"trained_706_{name}": float(value) for name, value in m706.items()},
                **{f"trained_783_{name}": float(value) for name, value in m783.items()},
                "delta_auc_without_77": float(m706["auc"] - m783["auc"]),
                "delta_ap_without_77": float(m706["ap"] - m783["ap"]),
                "delta_brier_without_77": float(m706["brier"] - m783["brier"]),
                **{k: json.dumps(v) for k, v in pair.items()},
                "same_706_test_rows_and_folds": True,
                "different_frozen_v3_hyperparameters_causal_limit": True,
            }
        )
    table = pd.DataFrame(records)
    out = base / "paired_77_ablation"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "six_algorithms_same_706_test.csv", index=False)
    (out / "interpretation.md").write_text(
        "# M2 v4 — ablación 77 GEOROC sobre TEST histórico común\n\n"
        "Compara los mismos 706 TEST OOF y los seis algoritmos, entrenados con "
        "706 o 783 filas. Los hiperparámetros están congelados **desde ejecuciones "
        "v3 distintas**, por lo que **no es una ablación causal estricta** del "
        "efecto de los 77. Los IC95 son bootstrap espacial exploratorio. "
        "No reemplazar release ni modificar etiquetas sin auditoría.\n",
        encoding="utf-8",
    )
    print(
        "M2_V4_PAIRED_77_ABLATION",
        table[
            [
                "algorithm",
                "delta_auc_without_77",
                "delta_ap_without_77",
                "delta_brier_without_77",
            ]
        ]
        .round(5)
        .to_string(index=False),
        flush=True,
    )
    return table


if __name__ == "__main__":
    compare()
