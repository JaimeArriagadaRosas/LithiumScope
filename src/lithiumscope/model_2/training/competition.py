from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.run_context import RunContext
from lithiumscope.model_2.evaluation.plots import save_competition_chart, save_probability_histogram, save_roc_pr
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.training.cv_runner import run_classification_cv
from lithiumscope.model_2.training.factory import create_model, get_label
from lithiumscope.persistence.save_model import save_model_bundle
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook

logger = get_logger("model_2.competition")


@dataclass
class CompetitionOutcome:
    run_dir: Path
    ranking: pd.DataFrame
    winner: str | None
    successful: list[str]
    failed: list[str]


def _ranking_row(result) -> dict:
    folds = result.fold_table
    return {
        "algorithm": result.algorithm,
        "label": result.label,
        "roc_auc_mean": float(folds["roc_auc"].mean()),
        "roc_auc_std": float(folds["roc_auc"].std(ddof=0)),
        "average_precision_mean": float(folds["average_precision"].mean()),
        "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
        "f1_mean": float(folds["f1"].mean()),
        "precision_mean": float(folds["precision"].mean()),
        "recall_mean": float(folds["recall"].mean()),
        "brier_score_mean": float(folds["brier_score"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
    }


def _excel_sheets(ranking: pd.DataFrame, results: dict[str, object], y: pd.Series) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {"ranking": ranking}
    for algorithm, result in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "y_true": y.to_numpy(),
                "prospectivity_score": result.probabilities,
                "predicted_class_0_5": (result.probabilities >= 0.5).astype(int),
            }
        )
    return sheets


def run_model_2_competition(manifest_path: Path, device: DeviceInfo) -> CompetitionOutcome:
    cfg = load_config("model_2")
    algorithms = list(cfg["model"]["algorithms"])
    seed = int(cfg["model"]["random_seed"])
    lithium_column = str(cfg["training"]["lithium_column"])
    spatial_group_column = str(cfg["training"].get("spatial_group_column", "spatial_group"))
    requested_folds = int(cfg["validation"].get("folds", 5))
    q = float(cfg["model"]["high_lithium_quantile"])
    context = RunContext.create("model_2")

    print("\n=== MODELO 2 · COMPETENCIA DE ALGORITMOS ===")
    frame = load_model_2_training_frame(manifest_path)
    threshold = float(frame[lithium_column].quantile(q))
    y = (frame[lithium_column] >= threshold).astype(int)
    groups = frame[spatial_group_column] if spatial_group_column in frame.columns else None
    x = frame.drop(columns=[column for column in (lithium_column, spatial_group_column) if column in frame.columns])

    min_class = int(y.value_counts().min())
    folds = max(2, min(requested_folds, min_class))
    print(f"Muestras: {len(frame)} | Umbral Li alto: {threshold:.4f} ppm | Folds: {folds}")

    rows: list[dict] = []
    results: dict[str, object] = {}
    failed: list[str] = []

    for index, algorithm in enumerate(algorithms, start=1):
        print(f"\n[{index}/{len(algorithms)}] Entrenando {get_label(algorithm)}")
        try:
            result = run_classification_cv(x, y, algorithm, device, seed, folds, groups)
            results[algorithm] = result
            rows.append(_ranking_row(result))

            result.fold_table.to_csv(context.tables / f"fold_metrics_{algorithm}.csv", index=False)
            pd.DataFrame(
                {
                    "y_true": y,
                    "prospectivity_score": result.probabilities,
                    "predicted_class_0_5": (result.probabilities >= 0.5).astype(int),
                }
            ).to_csv(context.tables / f"oof_predictions_{algorithm}.csv", index=False)

            save_probability_histogram(
                y,
                result.probabilities,
                context.figures / algorithm / "score_distribution.png",
                f"{result.label} — Distribución de scores",
            )
            save_roc_pr(
                y,
                result.probabilities,
                context.figures / algorithm / "roc_pr.png",
                result.label,
            )
        except Exception as exc:
            logger.exception("Model 2 algorithm failed: %s", algorithm)
            failed.append(algorithm)
            rows.append(
                {
                    "algorithm": algorithm,
                    "label": get_label(algorithm),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"  ✗ {get_label(algorithm)} falló: {exc}")

    ranking = pd.DataFrame(rows)
    ok = ranking[ranking["status"] == "ok"].copy()

    if ok.empty:
        ranking.to_csv(context.tables / "competition_ranking.csv", index=False)
        export_workbook(context.exports / "model_2_results.xlsx", _excel_sheets(ranking, results, y))
        build_dashboard(context.root, "LithiumScope — Modelo 2")
        return CompetitionOutcome(context.root, ranking, None, [], failed)

    ok = ok.sort_values(
        ["roc_auc_mean", "average_precision_mean", "balanced_accuracy_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ok.insert(0, "rank", range(1, len(ok) + 1))
    ranking = pd.concat([ok, ranking[ranking["status"] != "ok"]], ignore_index=True, sort=False)

    ranking.to_csv(context.tables / "competition_ranking.csv", index=False)
    save_competition_chart(ok, context.figures / "competition_roc_auc.png")

    winner = str(ok.iloc[0]["algorithm"])
    print(f"\n🏆 Ganador provisional Modelo 2: {get_label(winner)}")

    final_estimator = create_model(winner, device, seed)
    final_estimator.fit(x, y)
    winner_result = results[winner]

    metadata = {
        "model_group": "model_2",
        "algorithm": winner,
        "label": get_label(winner),
        "samples": len(frame),
        "threshold_ppm": threshold,
        "target_quantile": q,
        "metrics": winner_result.overall_metrics,
        "competition_primary_metric": "roc_auc_mean",
        "winner_rule": "highest mean ROC-AUC; tie-break Average Precision then Balanced Accuracy",
        "ranking": ok.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "scope_note": "Score de priorización exploratoria; no es probabilidad de yacimiento económicamente viable.",
    }

    bundle = {
        "estimator": final_estimator,
        "features": list(x.columns),
        "threshold_ppm": threshold,
        "target_quantile": q,
        "algorithm": winner,
    }
    model_path, metadata_path = save_model_bundle(
        "model_2",
        f"winner_{winner}",
        bundle,
        metadata,
    )

    export_workbook(
        context.exports / "model_2_results.xlsx",
        _excel_sheets(ranking, results, y),
    )
    (context.exports / "winner.txt").write_text(
        f"algorithm={winner}\nmodel={model_path}\nmetadata={metadata_path}\n",
        encoding="utf-8",
    )

    build_dashboard(context.root, "LithiumScope — Modelo 2 · Competencia")
    return CompetitionOutcome(context.root, ranking, winner, list(results), failed)
