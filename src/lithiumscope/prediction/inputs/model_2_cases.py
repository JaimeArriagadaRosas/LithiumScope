"""Resolve single imagery and manifest inputs without depending on orchestration."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.prediction.inputs.case_identity import ensure_case_ids
from lithiumscope.prediction.inputs.imagery import prepare_case_images
from lithiumscope.prediction.execution.session import PredictionSession

_IMAGE_EXTENSIONS = {".tif", ".tiff", ".npy"}
_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _model_2_cases_from_input(
    path: Path,
    session: PredictionSession,
    model_1_predictions: pd.DataFrame,
    *, reference_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    path = Path(path).resolve()
    reference_dir = Path(reference_dir).resolve() if reference_dir is not None else path.parent
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

    frame = ensure_case_ids(load_data(path))
    image_column = next(
        (name for name in ("image_path", "image", "path") if name in frame.columns),
        None,
    )
    if image_column is not None:
        rows: list[dict] = []
        for _, row in frame.iterrows():
            raw = Path(str(row[image_column]))
            resolved = raw if raw.is_absolute() else reference_dir / raw
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
