"""Focused integrated prediction phase; stable original public entrypoint."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lithiumscope.prediction.integrated import Path, PredictionSession, pd

def _model_2_cases_from_input(
    path: Path,
    session: PredictionSession,
    model_1_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    from lithiumscope.prediction.integrated import Path, PredictionSession, _IMAGE_EXTENSIONS, _TABLE_EXTENSIONS, ensure_case_ids, file_sha256, load_prediction_input, pd, prepare_case_images
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
