from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.prediction.report import write_pdf_report


def _resolve_run(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_dir():
        return candidate.resolve()
    candidate = RESULTS_DIR / "predictions" / raw
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"No existe la ejecucion: {raw}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def rebuild_pdf(run: str | Path) -> Path:
    run_dir = _resolve_run(str(run))
    manifest_path = run_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("La ejecucion no contiene prediction_manifest.json.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interpretation_path = run_dir / "interpretation.txt"
    interpretation = (
        interpretation_path.read_text(encoding="utf-8")
        if interpretation_path.is_file()
        else "Sin interpretacion general disponible."
    )

    model_1 = _read_csv(run_dir / "model_1" / "predictions.csv")
    model_2 = _read_csv(run_dir / "model_2" / "predictions.csv")
    paired = _read_csv(run_dir / "cross_model" / "joined_cases.csv")
    correlations = _read_csv(run_dir / "cross_model" / "correlations.csv")
    concordance = _read_csv(run_dir / "cross_model" / "concordance.csv")
    comparison = _read_csv(run_dir / "training_vs_external.csv")

    def read_json(path: Path) -> dict:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    figures = sorted(
        (run_dir / "cross_model" / "figures").glob("*.png")
    )
    title = (
        "LithiumScope - Demostracion integrada automatica"
        if manifest.get("mode") == "demonstration"
        else "LithiumScope - Prediccion completa"
    )
    destination = write_pdf_report(
        run_dir / "report.pdf",
        title=title,
        run_id=str(manifest.get("run_id", run_dir.name)),
        mode=str(manifest.get("mode", "unknown")),
        interpretation=interpretation,
        model_1_identity=manifest.get("models", {}).get("model_1", {}),
        model_2_identity=manifest.get("models", {}).get("model_2", {}),
        model_1_metrics=manifest.get("metrics", {}).get("model_1_external", {}),
        model_2_metrics=manifest.get("metrics", {}).get("model_2_external", {}),
        model_1_predictions=model_1,
        model_2_predictions=model_2,
        paired=paired,
        correlations=correlations,
        concordance=concordance,
        training_vs_external=comparison,
        overlap_audit=manifest.get("overlap_audit"),
        input_info=manifest.get("inputs"),
        model_1_diagnostics=read_json(
            run_dir / "model_1" / "diagnostics.json"
        ),
        model_2_diagnostics=read_json(
            run_dir / "model_2" / "diagnostics.json"
        ),
        figures=figures,
    )

    artifacts = manifest.setdefault("artifacts", {})
    artifacts["report_pdf"] = str(destination)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenera report.pdf para una ejecucion existente."
    )
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    destination = rebuild_pdf(args.run)
    print(f"PDF: {destination}")


if __name__ == "__main__":
    main()
