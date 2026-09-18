from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.run_context import RunContext
from lithiumscope.model_1.evaluation.artifacts import save_algorithm_artifacts, save_dataset_artifacts
from lithiumscope.model_1.evaluation.importance import save_feature_importance
from lithiumscope.model_1.evaluation.plots import save_competition_chart
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.training.cv_runner import run_nested_cv
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.persistence.save_model import save_model_bundle
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook

logger = get_logger("model_1.competition")


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
        "rmse_mean": float(folds["rmse"].mean()),
        "rmse_std": float(folds["rmse"].std(ddof=0)),
        "mae_mean": float(folds["mae"].mean()),
        "mae_std": float(folds["mae"].std(ddof=0)),
        "r2_mean": float(folds["r2"].mean()),
        "r2_std": float(folds["r2"].std(ddof=0)),
        "median_ae_mean": float(folds["median_ae"].mean()),
        "explained_variance_mean": float(folds["explained_variance"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
    }


def _rank(rows: list[dict]) -> pd.DataFrame:
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ok = ranking[ranking["status"] == "ok"].copy()
    failed = ranking[ranking["status"] != "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values(
            ["rmse_mean", "mae_mean", "r2_mean"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        ok.insert(0, "rank", range(1, len(ok) + 1))
    return pd.concat([ok, failed], ignore_index=True, sort=False)


def _excel_sheets(prepared, ranking: pd.DataFrame, results: dict[str, tuple]) -> dict[str, pd.DataFrame]:
    sheets = {
        "ranking": ranking,
        "pipeline": prepared.audit.to_frame(),
        "dataset_summary": prepared.frame.describe(include="all").transpose().reset_index(names="variable"),
        "correlations": prepared.frame.select_dtypes(include=[np.number]).corr(numeric_only=True).reset_index(),
    }
    for algorithm, (result, algorithm_prepared) in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "row_index": algorithm_prepared.y.index,
                "Li_real": algorithm_prepared.y.to_numpy(),
                "Li_pred": result.predictions,
                "residual": algorithm_prepared.y.to_numpy() - result.predictions,
            }
        )
    return sheets


def run_model_1_competition(dataset_path: Path, device: DeviceInfo) -> CompetitionOutcome:
    config = load_config("model_1")
    algorithms = list(config["model"]["algorithms"])
    context = RunContext.create("model_1")

    print("\n=== MODELO 1 · COMPETENCIA DE ALGORITMOS ===")
    print("Preparando dataset común...")
    prepared = prepare_training_data(dataset_path, model_family="random_forest")
    save_dataset_artifacts(prepared, context)

    rows: list[dict] = []
    results: dict[str, tuple] = {}
    failed: list[str] = []

    for index, algorithm in enumerate(algorithms, start=1):
        label = get_algorithm(algorithm).label
        print(f"\n[{index}/{len(algorithms)}] Entrenando {label}")
        try:
            algorithm_prepared = prepared if algorithm != "svm" else prepare_training_data(dataset_path, model_family="svm")
            result = run_nested_cv(algorithm_prepared, algorithm, device, config)
            results[algorithm] = (result, algorithm_prepared)
            save_algorithm_artifacts(algorithm_prepared, result, context)
            rows.append(_ranking_row(result))
        except Exception as exc:
            logger.exception("Algorithm failed: %s", algorithm)
            failed.append(algorithm)
            rows.append({"algorithm": algorithm, "label": label, "status": "failed", "error": str(exc)})
            print(f"  ✗ {label} falló: {exc}")

    ranking = _rank(rows)
    ranking.to_csv(context.tables / "competition_ranking.csv", index=False)
    successful = ranking[ranking["status"] == "ok"].copy() if not ranking.empty else pd.DataFrame()

    if successful.empty:
        export_workbook(context.exports / "model_1_results.xlsx", _excel_sheets(prepared, ranking, results))
        build_dashboard(context.root, "LithiumScope — Modelo 1")
        return CompetitionOutcome(context.root, ranking, None, [], failed)

    save_competition_chart(successful, context.figures / "competition_rmse.png")
    winner = str(successful.iloc[0]["algorithm"])
    winner_result, winner_prepared = results[winner]
    seed = int(config["validation"]["random_seed"])
    final_estimator = build_pipeline(winner, device, seed, winner_prepared.schema, winner_result.final_params)

    print(f"\n🏆 Ganador provisional: {get_algorithm(winner).label}")
    print("Entrenando ganador con todas las muestras disponibles...")
    final_estimator.fit(winner_prepared.x, winner_prepared.y)
    save_feature_importance(
        final_estimator,
        context.tables / "winner_feature_importance.csv",
        context.figures / "winner_feature_importance.png",
    )

    metadata = {
        "model_group": "model_1",
        "algorithm": winner,
        "label": get_algorithm(winner).label,
        "samples": len(winner_prepared.y),
        "metrics": winner_result.overall_metrics,
        "competition_primary_metric": "rmse_mean",
        "winner_rule": "lowest mean outer-CV RMSE; tie-break MAE then R2",
        "ranking": successful.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "device_name": device.name,
    }
    bundle = {
        "estimator": final_estimator,
        "algorithm": winner,
        "target": winner_prepared.target,
        "schema": winner_prepared.schema,
    }
    model_path, metadata_path = save_model_bundle("model_1", f"winner_{winner}", bundle, metadata)

    export_workbook(context.exports / "model_1_results.xlsx", _excel_sheets(prepared, ranking, results))
    (context.exports / "winner.txt").write_text(
        f"algorithm={winner}\nmodel={model_path}\nmetadata={metadata_path}\n",
        encoding="utf-8",
    )
    build_dashboard(context.root, "LithiumScope — Modelo 1 · Competencia")
    return CompetitionOutcome(context.root, ranking, winner, list(results), failed)
