from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR


_TEXT_SUFFIXES = {".json", ".csv", ".txt", ".log", ".md"}


def _resolve_run(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_dir():
        return candidate.resolve()
    under_results = RESULTS_DIR / "predictions" / raw
    if under_results.is_dir():
        return under_results.resolve()
    raise FileNotFoundError(f"No existe la ejecucion de demostracion: {raw}")


def _sanitize_text(text: str, run_dir: Path) -> str:
    replacements = {
        str(PROJECT_ROOT.resolve()): "<PROJECT_ROOT>",
        str(run_dir.resolve()): "<RUN_DIR>",
        str(PROJECT_ROOT.resolve()).replace("\\", "\\\\"): "<PROJECT_ROOT>",
        str(run_dir.resolve()).replace("\\", "\\\\"): "<RUN_DIR>",
    }
    sanitized = text
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    return sanitized


def _publication_manifest(manifest: dict, archive_name: str) -> dict:
    runtime = dict(manifest.get("runtime", {}))
    runtime.pop("hostname", None)
    runtime.pop("pid", None)

    models = {}
    for model_group, identity in manifest.get("models", {}).items():
        models[model_group] = {
            "run_id": identity.get("run_id"),
            "algorithm": identity.get("algorithm"),
            "model_sha256": identity.get("model_sha256"),
            "dataset_sha256": identity.get("dataset_sha256"),
            "training_metrics": identity.get("training_metrics", {}),
        }

    return {
        "schema_version": 1,
        "artifact_type": "lithiumscope_demonstration",
        "run_id": manifest.get("run_id"),
        "mode": manifest.get("mode"),
        "created_at_utc": manifest.get("created_at_utc"),
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_name": archive_name,
        "models": models,
        "inputs": manifest.get("inputs", {}),
        "metrics": manifest.get("metrics", {}),
        "overlap_audit": manifest.get("overlap_audit"),
        "counts": manifest.get("counts", {}),
        "thresholds": manifest.get("thresholds", {}),
        "runtime": runtime,
        "artifacts": manifest.get("artifacts", {}),
    }


def build_demonstration_release_bundle(
    run: str | Path,
) -> tuple[Path, Path]:
    run_dir = _resolve_run(str(run))
    manifest_path = run_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("La ejecucion no contiene prediction_manifest.json.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "demonstration":
        raise RuntimeError("Solo se pueden publicar ejecuciones de demostracion.")
    report_pdf = run_dir / "report.pdf"
    if not report_pdf.is_file():
        raise RuntimeError(
            "La demostracion no contiene report.pdf. Ejecute nuevamente la opcion 4."
        )

    run_id = str(manifest["run_id"])
    stamp = run_id.removeprefix("demonstration_")
    output_dir = RESULTS_DIR / "release_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"lithiumscope-demo_{stamp}-artifacts.zip"
    temporary = archive.with_suffix(".zip.tmp")

    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            arcname = path.relative_to(run_dir).as_posix()
            if path.suffix.lower() in _TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="replace")
                bundle.writestr(arcname, _sanitize_text(text, run_dir))
            else:
                bundle.write(path, arcname=arcname)

        publication = _publication_manifest(manifest, archive.name)
        publication_text = json.dumps(
            publication,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        bundle.writestr(
            "release_manifest.json",
            _sanitize_text(publication_text, run_dir),
        )

    temporary.replace(archive)
    digest = file_sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(
        f"{digest}  {archive.name}\n",
        encoding="utf-8",
    )
    return archive, checksum


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empaqueta una demostracion LithiumScope para GitHub Release."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="run_id o ruta a results/predictions/<run_id>",
    )
    args = parser.parse_args()
    archive, checksum = build_demonstration_release_bundle(args.run)
    print(f"Bundle:   {archive}")
    print(f"SHA-256:  {checksum.read_text(encoding='utf-8').strip()}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
