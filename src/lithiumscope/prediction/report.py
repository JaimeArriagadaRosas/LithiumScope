from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def write_json(
    path: Path,
    payload: dict,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _load_reporting_callable(
    module_name: str,
    attribute: str,
):
    try:
        module = importlib.import_module(
            module_name
        )
    except ModuleNotFoundError:
        from lithiumscope.runtime.dependencies import (
            ensure_runtime_dependencies,
        )

        result = ensure_runtime_dependencies(
            verbose=True,
        )
        if not result.success:
            raise
        importlib.invalidate_caches()
        module = importlib.import_module(
            module_name
        )
    return getattr(
        module,
        attribute,
    )


def save_integrated_figures(
    paired: Any,
    figure_dir: Path,
) -> list[Path]:
    writer = _load_reporting_callable(
        "lithiumscope.prediction.reporting.charts",
        "save_integrated_figures",
    )
    return writer(
        paired,
        figure_dir,
    )


def save_spatial_maps(
    paired: Any,
    destination_dir: Path,
) -> list[Path]:
    writer = _load_reporting_callable(
        "lithiumscope.prediction.reporting.maps",
        "save_spatial_maps",
    )
    return writer(
        paired,
        destination_dir,
    )


def write_workbook(
    path: Path,
    *,
    model_1: Any,
    model_2: Any,
    paired: Any,
    correlations: Any,
    concordance: Any,
    training_vs_external: Any,
) -> Path:
    writer = _load_reporting_callable(
        "lithiumscope.prediction.reporting.workbook",
        "write_workbook",
    )
    return writer(
        path,
        model_1=model_1,
        model_2=model_2,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=training_vs_external,
    )


def write_pdf_report(
    path: Path,
    *,
    title: str,
    run_id: str,
    mode: str,
    interpretation: str,
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_metrics: dict,
    model_2_metrics: dict,
    model_1_predictions: Any,
    model_2_predictions: Any,
    paired: Any,
    correlations: Any,
    concordance: Any,
    training_vs_external: Any,
    overlap_audit: dict | None,
    input_info: dict | None,
    model_1_diagnostics: dict | None,
    model_2_diagnostics: dict | None,
    maps: list[Path],
    figures: list[Path],
    lithium_threshold_ppm: float | None = None,
    model_2_classification_threshold: float | None = None,
    runtime: dict | None = None,
) -> Path:
    writer = _load_reporting_callable(
        "lithiumscope.prediction.reporting.pdf",
        "write_pdf_report",
    )
    return writer(
        path,
        title=title,
        run_id=run_id,
        mode=mode,
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
        lithium_threshold_ppm=lithium_threshold_ppm,
        model_2_classification_threshold=(
            model_2_classification_threshold
        ),
        runtime=runtime,
    )
