"""Read-only environment health checks; no installation or resource repair."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import yaml

from lithiumscope.tools.inspection.code_size import ALLOWED, SKIP, scan
from lithiumscope.core.configuration.schema import validate_config
from lithiumscope.core.paths import CONFIG_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR
from lithiumscope.resources.manager import resource_snapshot
from lithiumscope.resources.allocation.policy import ResourcePolicy
from lithiumscope.core.hashing import file_sha256
from lithiumscope.runtime.dependencies import CORE_DEPENDENCIES, ML_DEPENDENCIES, IMAGERY_DEPENDENCIES, dependency_status
from lithiumscope.results.evidence import read_record


def architecture_audit(root: Path) -> dict:
    folders = Counter()
    for name in ("src", "tests", "scripts"):
        for folder, dirs, files in os.walk(root / name):
            dirs[:] = [d for d in dirs if d not in SKIP]
            folders[Path(folder).relative_to(root).as_posix()] = sum(Path(f).suffix.lower() in ALLOWED for f in files)
    return {
        "files_over_300": scan(root),
        "directories_at_20": [{"path": p, "files": n} for p, n in sorted(folders.items()) if n == 20],
        "directories_over_20": [{"path": p, "files": n} for p, n in sorted(folders.items()) if n > 20],
    }


def _dependencies(mapping: dict) -> dict:
    missing = [item.name for item in dependency_status(mapping) if not item.available]
    return {"state": "ADVERTENCIA" if missing else "OK", "missing": missing}


def _configs() -> tuple[dict, dict]:
    loaded, failures = {}, {}
    for name in ("app", "resources", "model_1", "model_2", "distribution", "logging", "prediction"):
        path = CONFIG_DIR / f"{name}.yaml"
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            loaded[name] = validate_config(name, payload)
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as exc:
            failures[name] = str(exc)
    return loaded, {"state": "ADVERTENCIA" if failures else "OK", "errors": failures}


def _datasets(config: dict) -> dict:
    training = config.get("model_2", {}).get("training", {})
    raw = training.get("manifest_path")
    path = PROJECT_ROOT / raw if raw else None
    available = path is not None and path.is_file()
    expected = training.get("frozen_manifest_sha256")
    integrity = None
    if available and expected:
        integrity = file_sha256(path) == expected
    m1_files = list((DATA_DIR / "raw" / "model_1").glob("*"))
    return {"state": "OK" if available and integrity is not False and any(p.is_file() and p.name != ".gitkeep" for p in m1_files) else "ADVERTENCIA",
            "m1_local_files": [str(p) for p in m1_files if p.is_file() and p.name != ".gitkeep"],
            "m2_manifest": str(path) if path else None, "m2_manifest_available": available,
            "m2_manifest_hash_matches": integrity, "scope": "Local availability and manifest hash only"}


def _models() -> dict:
    registry = read_record(MODELS_DIR / "active_models.json") or {}
    entries = registry.get("models", {})
    models = {}
    for group in ("model_1", "model_2"):
        entry = entries.get(group, {}) if isinstance(entries, dict) else {}
        raw = entry.get("model_path") if isinstance(entry, dict) else None
        path = PROJECT_ROOT / raw if raw else None
        if path and path.is_file():
            metadata = read_record(path.parent / "metadata.json") or {}
            expected = metadata.get("model_sha256")
            models[group] = {"path": str(path), "hash_matches": file_sha256(path) == expected if expected else None}
        else:
            models[group] = {"path": str(path) if path else None, "hash_matches": None}
    return {"state": "OK" if all(v["hash_matches"] is True for v in models.values()) else "ADVERTENCIA",
            "models": models, "scope": "No model deserialization performed"}


def inspect_health() -> dict:
    configs, config_status = _configs()
    architecture = architecture_audit(PROJECT_ROOT)
    paths = [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]
    permissions = {str(p): p.is_dir() and os.access(p, os.R_OK | os.W_OK) for p in paths}
    areas = {
        "Python": {"state": "OK" if sys.version_info >= (3, 11) else "ERROR", "version": sys.version.split()[0]},
        "Dependencias de inferencia": _dependencies(CORE_DEPENDENCIES),
        "Dependencias de entrenamiento": _dependencies(ML_DEPENDENCIES),
        "Configuración": config_status,
        "Recursos computacionales": {"state": "OK", **resource_snapshot(ResourcePolicy.from_config(configs.get("resources", {})))},
        "Datasets y manifiestos": _datasets(configs),
        "Modelos y persistencia": _models(),
        "Permisos y directorios": {"state": "OK" if all(permissions.values()) else "ADVERTENCIA", "paths": permissions},
        "Arquitectura del repositorio": {"state": "ADVERTENCIA" if any(architecture.values()) else "OK"},
    }
    recommendations = []
    if architecture["files_over_300"]:
        recommendations.append("Separar responsabilidades de los archivos sobre 300 líneas.")
    if architecture["directories_at_20"] or architecture["directories_over_20"]:
        recommendations.append("Evaluar/reorganizar los directorios identificados según responsabilidades.")
    if config_status["errors"]:
        recommendations.append("Revisar las configuraciones faltantes o inválidas.")
    for area, status in areas.items():
        if status.get("missing"):
            recommendations.append(f"Revisar dependencias de {area}: {', '.join(status['missing'])}.")
    if areas["Datasets y manifiestos"]["state"] != "OK":
        recommendations.append("Verificar fuentes y manifiestos mediante su procedimiento de preparación.")
    if areas["Modelos y persistencia"]["state"] != "OK":
        recommendations.append("Revisar disponibilidad e integridad de modelos registrados.")
    states = {value["state"] for value in areas.values()}
    return {"mode": "read_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": "ERROR" if "ERROR" in states else "COMPLETADO CON ADVERTENCIAS" if "ADVERTENCIA" in states else "COMPLETADO",
            "areas": areas, "architecture": architecture, "recommendations": recommendations,
            "imagery_dependencies": _dependencies(IMAGERY_DEPENDENCIES)}


def write_health_report(output_root: Path | None = None) -> Path:
    root = output_root if output_root is not None else RESULTS_DIR / "diagnostics"
    report = inspect_health()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"health_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "health_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
