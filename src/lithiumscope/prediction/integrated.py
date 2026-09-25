from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logger import get_logger, run_log
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.model_1.prediction.input_parser import load_prediction_input
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.model_2.prediction.image_parser import image_to_feature_frame
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.prediction.analysis import (
    concordance_table,
    correlation_table,
    ensure_case_ids,
    model_1_external_metrics,
    model_2_external_metrics,
    pair_model_outputs,
    training_vs_external_table,
)
from lithiumscope.prediction.demo import (
    audit_demo_overlap,
    demonstration_imagery_cache,
    load_demonstration_cases,
)
from lithiumscope.prediction.imagery import prepare_case_images
from lithiumscope.prediction.interpretation import (
    build_overall_interpretation,
    integrated_case_text,
    model_1_case_text,
    model_2_case_text,
)
from lithiumscope.prediction.report import (
    save_integrated_figures,
    save_spatial_maps,
    write_json,
    write_pdf_report,
    write_workbook,
)
from lithiumscope.prediction.session import (
    PredictionSession,
    load_model_with_identity,
)

logger = get_logger("prediction.integrated")


@dataclass(frozen=True)
class IntegratedPredictionResult:
    session: PredictionSession
    model_1_predictions: pd.DataFrame
    model_2_predictions: pd.DataFrame
    paired: pd.DataFrame
    correlations: pd.DataFrame
    concordance: pd.DataFrame
    model_1_metrics: dict
    model_2_metrics: dict
    interpretation: str
    report_path: Path
    workbook_path: Path
    manifest_path: Path


_IMAGE_EXTENSIONS = {".tif", ".tiff", ".npy"}
_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _missing_expected_columns(frame: pd.DataFrame, schema) -> list[str]:
    expected = list(schema.numeric) + list(schema.categorical)
    return [column for column in expected if column not in frame.columns]


def _score_model_2_cases(
    cases: pd.DataFrame,
    bundle: dict,
) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    missing_feature_counts: dict[str, int] = {}
    expected = list(bundle["features"])

    for _, row in cases.iterrows():
        if str(row.get("sentinel_status", "ready")) != "ready":
            continue
        image_path = Path(str(row["image_path"]))
        frame = image_to_feature_frame(
            image_path,
            band_names=bundle.get("band_names", ()),
            normalize_per_band=bool(bundle.get("normalize_per_band", False)),
        )
        missing = [column for column in expected if column not in frame.columns]
        for column in missing:
            missing_feature_counts[column] = missing_feature_counts.get(column, 0) + 1
            frame[column] = 0.0

        score = float(bundle["estimator"].predict_proba(frame[expected])[0, 1])
        operating_threshold = float(
            bundle.get("operating_threshold", 0.5)
        )
        priority = "alta" if score >= 0.70 else "media" if score >= 0.40 else "baja"
        applicability = out_of_range_fraction(
            frame[expected],
            bundle.get("applicability_profile", {}),
        )
        record = {
            "case_id": str(row["case_id"]),
            "Li_icpms": row.get("Li_icpms"),
            "image": str(image_path),
            "prospectivity_score": score,
            "priority": priority,
            "operating_positive": bool(
                score >= operating_threshold
            ),
            "operating_threshold": operating_threshold,
            "algorithm": bundle.get("algorithm", "unknown"),
            "out_of_training_range_fraction": float(applicability),
            "applicability_warning": "OUT_OF_DOMAIN" if applicability > 0.25 else "OK",
            "sentinel_scene_id": row.get("sentinel_scene_id"),
            "sentinel_cloud_cover": row.get("sentinel_cloud_cover"),
            "sentinel_datetime": row.get("sentinel_datetime"),
            "warning": "Prioridad exploratoria relativa; no es probabilidad de yacimiento.",
        }
        record["scientific_interpretation"] = model_2_case_text(pd.Series(record))
        records.append(record)

    diagnostics = {
        "expected_features": expected,
        "missing_feature_counts": missing_feature_counts,
        "ready_cases": len(records),
        "requested_cases": len(cases),
        "failed_cases": int(len(cases) - len(records)),
        "operating_threshold": float(
            bundle.get("operating_threshold", 0.5)
        ),
        "operating_threshold_objective": bundle.get(
            "operating_threshold_objective"
        ),
    }
    return pd.DataFrame(records), diagnostics


def _model_2_cases_from_input(
    path: Path,
    session: PredictionSession,
    model_1_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        case_id = (
            str(model_1_predictions.iloc[0]["case_id"])
            if len(model_1_predictions) == 1
            else "model_2_case_0001"
        )
        return (
            pd.DataFrame(
                [
                    {
                        "case_id": case_id,
                        "image_path": str(path),
                        "sentinel_status": "ready",
                        "Li_icpms": (
                            model_1_predictions.iloc[0].get("Li_icpms")
                            if len(model_1_predictions) == 1
                            else None
                        ),
                    }
                ]
            ),
            {
                "input_type": "single_image",
                "input_path": str(path),
                "input_sha256": file_sha256(path),
            },
        )

    if suffix not in _TABLE_EXTENSIONS:
        raise ValueError(
            "La predicción completa requiere una imagen multibanda o un manifest CSV/Excel para Modelo 2."
        )

    frame = ensure_case_ids(load_prediction_input(path))
    image_column = next(
        (name for name in ("image_path", "image", "path") if name in frame.columns),
        None,
    )
    if image_column is not None:
        rows: list[dict] = []
        for _, row in frame.iterrows():
            raw = Path(str(row[image_column]))
            resolved = raw if raw.is_absolute() else path.parent / raw
            rows.append(
                {
                    "case_id": str(row["case_id"]),
                    "Li_icpms": row.get("Li_icpms"),
                    "image_path": str(resolved),
                    "sentinel_status": "ready" if resolved.is_file() else "missing_image",
                    "sentinel_error": None if resolved.is_file() else f"No existe {resolved}",
                }
            )
        cases = pd.DataFrame(rows)
    else:
        cases = prepare_case_images(
            frame,
            session.model_2 / "imagery",
        )

    return (
        cases,
        {
            "input_type": "manifest",
            "input_path": str(path),
            "input_sha256": file_sha256(path),
        },
    )


def _write_outputs(
    *,
    session: PredictionSession,
    model_1_predictions: pd.DataFrame,
    model_2_predictions: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
    model_1_metrics: dict,
    model_2_metrics: dict,
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_diagnostics: dict,
    model_2_diagnostics: dict,
    input_info: dict,
    overlap_audit: dict | None,
    interpretation: str,
    run_log_path: Path,
    lithium_threshold: float,
    probability_threshold: float,
    model_2_predicted_positive_count: int,
) -> tuple[Path, Path, Path]:
    model_1_predictions.to_csv(session.model_1 / "predictions.csv", index=False)
    model_2_predictions.to_csv(session.model_2 / "predictions.csv", index=False)
    paired.to_csv(session.cross_model / "joined_cases.csv", index=False)
    correlations.to_csv(session.cross_model / "correlations.csv", index=False)
    concordance.to_csv(session.cross_model / "concordance.csv", index=False)
    training_vs_external.to_csv(session.root / "training_vs_external.csv", index=False)
    write_json(session.model_1 / "diagnostics.json", model_1_diagnostics)
    write_json(session.model_2 / "diagnostics.json", model_2_diagnostics)

    interpretation_path = session.root / "interpretation.txt"
    interpretation_path.write_text(interpretation + "\n", encoding="utf-8")
    maps = save_spatial_maps(
        paired,
        session.figures / "maps",
    )
    figures = save_integrated_figures(paired, session.figures)

    workbook_path = write_workbook(
        session.root / "evaluation.xlsx",
        model_1=model_1_predictions,
        model_2=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=training_vs_external,
    )
    runtime = runtime_fingerprint()
    report_path = write_pdf_report(
        session.root / "report.pdf",
        title=(
            "LithiumScope - Demostración integrada automática"
            if session.mode == "demonstration"
            else "LithiumScope - Predicción completa"
        ),
        run_id=session.run_id,
        mode=session.mode,
        interpretation=interpretation,
        model_1_identity=model_1_identity,
        model_2_identity=model_2_identity,
        model_1_metrics=model_1_metrics,
        model_2_metrics=model_2_metrics,
        model_1_predictions=model_1_predictions,
        model_2_predictions=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=training_vs_external,
        overlap_audit=overlap_audit,
        input_info=input_info,
        model_1_diagnostics=model_1_diagnostics,
        model_2_diagnostics=model_2_diagnostics,
        maps=maps,
        figures=figures,
        lithium_threshold_ppm=lithium_threshold,
        model_2_classification_threshold=probability_threshold,
        runtime=runtime,
    )

    manifest = {
        "schema_version": 1,
        "run_id": session.run_id,
        "mode": session.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "model_1": model_1_identity,
            "model_2": model_2_identity,
        },
        "inputs": input_info,
        "metrics": {
            "model_1_external": model_1_metrics,
            "model_2_external": model_2_metrics,
        },
        "overlap_audit": overlap_audit,
        "counts": {
            "model_1_cases": len(model_1_predictions),
            "model_2_cases": len(model_2_predictions),
            "paired_cases": len(paired),
            "model_2_predicted_positive": model_2_predicted_positive_count,
        },
        "thresholds": {
            "lithium_reference_ppm": lithium_threshold,
            "model_2_classification_score": probability_threshold,
        },
        "runtime": runtime,
        "artifacts": {
            "model_1_predictions": str(session.model_1 / "predictions.csv"),
            "model_2_predictions": str(session.model_2 / "predictions.csv"),
            "joined_cases": str(session.cross_model / "joined_cases.csv"),
            "correlations": str(session.cross_model / "correlations.csv"),
            "concordance": str(session.cross_model / "concordance.csv"),
            "training_vs_external": str(session.root / "training_vs_external.csv"),
            "workbook": str(workbook_path),
            "report_pdf": str(report_path),
            "interpretation": str(interpretation_path),
            "run_log": str(run_log_path),
            "maps": [str(path) for path in maps],
            "figures": [str(path) for path in figures],
        },
    }
    manifest_path = write_json(session.root / "prediction_manifest.json", manifest)
    return report_path, workbook_path, manifest_path


def _run_integrated(
    *,
    session: PredictionSession,
    model_1_input: pd.DataFrame,
    model_2_cases: pd.DataFrame,
    input_info: dict,
    overlap_audit: dict | None = None,
) -> IntegratedPredictionResult:
    prediction_cfg = load_config("prediction")["interpretation"]
    fallback_probability_threshold = float(
        prediction_cfg["model_2_classification_threshold"]
    )
    local_log_path = session.root / "prediction.log"

    with run_log("prediction", session.run_id) as log_path:
        logger.info(
            "Integrated prediction started run=%s mode=%s",
            session.run_id,
            session.mode,
        )

        model_1_bundle, _, model_1_identity = load_model_with_identity(
            "model_1"
        )
        model_2_bundle, _, model_2_identity = load_model_with_identity(
            "model_2"
        )
        logger.info(
            "Models loaded model_1=%s model_2=%s",
            model_1_identity["run_id"],
            model_2_identity["run_id"],
        )
        lithium_threshold = float(model_2_bundle["threshold_ppm"])
        probability_threshold = float(
            model_2_bundle.get(
                "operating_threshold",
                fallback_probability_threshold,
            )
        )

        raw_m1 = ensure_case_ids(model_1_input)
        logger.info(
            "Model 1 prediction stage started cases=%d",
            len(raw_m1),
        )
        model_1_predictions, prediction_diagnostics = (
            predict_model_1_frame(
                raw_m1,
                bundle=model_1_bundle,
            )
        )
        model_1_predictions["scientific_interpretation"] = (
            model_1_predictions.apply(
                lambda row: model_1_case_text(
                    row,
                    lithium_threshold_ppm=lithium_threshold,
                ),
                axis=1,
            )
        )
        model_1_diagnostics = {
            **prediction_diagnostics,
            "input_rows": len(raw_m1),
        }
        logger.info(
            "Model 1 prediction stage completed cases=%d ood_rows=%s",
            len(model_1_predictions),
            model_1_diagnostics.get("out_of_domain_rows"),
        )

        logger.info(
            "Model 2 prediction stage started requested_cases=%d",
            len(model_2_cases),
        )
        model_2_predictions, model_2_diagnostics = _score_model_2_cases(
            model_2_cases,
            model_2_bundle,
        )
        logger.info(
            "Model 2 prediction stage completed ready=%d failed=%d",
            len(model_2_predictions),
            model_2_diagnostics["failed_cases"],
        )

        paired = pair_model_outputs(
            model_1_predictions,
            model_2_predictions,
        )
        correlations = correlation_table(paired)
        concordance = concordance_table(
            paired,
            lithium_threshold_ppm=lithium_threshold,
            model_2_probability_threshold=probability_threshold,
        )
        if not concordance.empty:
            concordance["scientific_interpretation"] = concordance.apply(
                integrated_case_text,
                axis=1,
            )

        model_1_metrics = model_1_external_metrics(
            model_1_predictions
        )
        model_2_metrics = model_2_external_metrics(
            model_2_predictions,
            threshold_ppm=lithium_threshold,
            probability_threshold=probability_threshold,
        )
        training_vs_external = training_vs_external_table(
            model_1_identity,
            model_2_identity,
            model_1_metrics,
            model_2_metrics,
        )
        model_2_scores = pd.to_numeric(
            model_2_predictions.get(
                "prospectivity_score",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        model_2_predicted_positive_count = int(
            (model_2_scores >= probability_threshold).sum()
        )
        interpretation = build_overall_interpretation(
            model_1_metrics=model_1_metrics,
            model_2_metrics=model_2_metrics,
            correlations=correlations,
            concordance=concordance,
            overlap_audit=overlap_audit,
            model_1_case_count=len(model_1_predictions),
            model_2_case_count=len(model_2_predictions),
            paired_case_count=len(paired),
            model_2_predicted_positive_count=model_2_predicted_positive_count,
        )
        logger.info(
            "Cross-model analysis completed paired=%d correlations=%d",
            len(paired),
            len(correlations),
        )

        report_path, workbook_path, manifest_path = _write_outputs(
            session=session,
            model_1_predictions=model_1_predictions,
            model_2_predictions=model_2_predictions,
            paired=paired,
            correlations=correlations,
            concordance=concordance,
            training_vs_external=training_vs_external,
            model_1_metrics=model_1_metrics,
            model_2_metrics=model_2_metrics,
            model_1_identity=model_1_identity,
            model_2_identity=model_2_identity,
            model_1_diagnostics=model_1_diagnostics,
            model_2_diagnostics=model_2_diagnostics,
            input_info=input_info,
            overlap_audit=overlap_audit,
            interpretation=interpretation,
            run_log_path=local_log_path,
            lithium_threshold=lithium_threshold,
            probability_threshold=probability_threshold,
            model_2_predicted_positive_count=model_2_predicted_positive_count,
        )
        logger.info(
            "Integrated prediction finished run=%s report=%s workbook=%s",
            session.run_id,
            report_path,
            workbook_path,
        )

    shutil.copy2(log_path, local_log_path)

    return IntegratedPredictionResult(
        session=session,
        model_1_predictions=model_1_predictions,
        model_2_predictions=model_2_predictions,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        model_1_metrics=model_1_metrics,
        model_2_metrics=model_2_metrics,
        interpretation=interpretation,
        report_path=report_path,
        workbook_path=workbook_path,
        manifest_path=manifest_path,
    )

def run_complete_prediction(
    model_1_path: Path,
    model_2_path: Path,
) -> IntegratedPredictionResult:
    session = PredictionSession.create("complete")

    model_1_snapshot = (
        session.inputs / f"model_1_input{model_1_path.suffix.lower()}"
    )
    model_2_snapshot = (
        session.inputs / f"model_2_input{model_2_path.suffix.lower()}"
    )
    shutil.copy2(model_1_path, model_1_snapshot)
    shutil.copy2(model_2_path, model_2_snapshot)

    model_1_input = ensure_case_ids(
        load_prediction_input(model_1_snapshot)
    )
    model_2_cases, model_2_input_info = _model_2_cases_from_input(
        model_2_path,
        session,
        model_1_input,
    )
    input_info = {
        "model_1": {
            "original_path": str(model_1_path),
            "snapshot_path": str(model_1_snapshot),
            "sha256": file_sha256(model_1_snapshot),
        },
        "model_2": {
            **model_2_input_info,
            "original_path": str(model_2_path),
            "snapshot_path": str(model_2_snapshot),
            "snapshot_sha256": file_sha256(model_2_snapshot),
        },
        "pairing_rule": (
            "case_id; una imagen única se empareja automáticamente "
            "solo cuando Modelo 1 contiene un único caso"
        ),
    }
    return _run_integrated(
        session=session,
        model_1_input=model_1_input,
        model_2_cases=model_2_cases,
        input_info=input_info,
    )

def run_automatic_demonstration() -> IntegratedPredictionResult:
    session = PredictionSession.create("demonstration")
    cases = ensure_case_ids(load_demonstration_cases())
    demo_snapshot = session.inputs / "demonstration_cases.csv"
    cases.to_csv(demo_snapshot, index=False)
    overlap = audit_demo_overlap(cases)
    demo_cfg = load_config("prediction")["demonstration"]
    imagery = prepare_case_images(
        cases,
        demonstration_imagery_cache(),
        datetime_override=str(demo_cfg["sentinel_datetime"]),
    )
    input_info = {
        "demonstration": {
            "cases_path": str(demo_cfg["cases_path"]),
            "snapshot_path": str(demo_snapshot),
            "cases_sha256": file_sha256(
                PROJECT_ROOT / str(demo_cfg["cases_path"])
            ),
            "source_name": demo_cfg["source_name"],
            "source_repository": demo_cfg["source_repository"],
            "source_commit": demo_cfg["source_commit"],
            "source_path": demo_cfg["source_path"],
            "source_license": demo_cfg["source_license"],
            "sentinel_datetime": demo_cfg["sentinel_datetime"],
        }
    }
    return _run_integrated(
        session=session,
        model_1_input=cases,
        model_2_cases=imagery,
        input_info=input_info,
        overlap_audit=overlap,
    )
