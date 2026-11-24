"""Temporary option 5.4: shareable development report, without uploading anything."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import sys
from uuid import uuid4

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.tools.inspection.development.inventory import (
    active_models, directory_record, file_record, git_identity, latest_test_report, local_path,
)

LOGGER = logging.getLogger("lithiumscope.development_snapshot")
PACKAGES = ("lithiumscope", "numpy", "pandas", "scikit-learn", "scipy", "joblib",
            "catboost", "xgboost", "optuna", "torch", "pytorch-tabnet", "rasterio",
            "pystac-client", "psutil", "pytest", "pytest-cov", "ruff", "reportlab",
            "pypdf", "PyMuPDF", "openpyxl", "matplotlib")
ARTIFACTS = (
    "data/processed/model_1/training_combined.csv",
    "data/processed/model_1/source_merge_audit.json",
    "data/interim/model_1/georoc_harmonized.csv",
    "data/processed/model_2/training_manifest.csv",
    "data/processed/model_2/relocation.json",
    "data/processed/model_2/sentinel_features.csv",
    "data/processed/model_2/sentinel_features.csv.meta.json",
    "config/model_2_frozen_spatial_folds_706.json",
    "results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/training_v4_170.csv",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/build_status.json",
    "results/scientific_simulations/m2_rededge_v4_20261113/cohort_706/rededge_quality.csv",
    "examples/integrated_demo/cases.csv",
)
DIAGNOSTICS = (
    "m1_source_reconciliation", "m1_preprocessing_contract", "m1_high_lithium_audit",
    "m1_statistical_baselines", "m1_historical_extremes", "m1_extreme_provenance",
    "m1_spatial_training_contract", "m1_coordinate_lineage_eda",
    "m2_physical_v5_706_precision", "m2_v4_full_inference_parity",
    "historical_m2_competition_audit", "historical_m2_checkpoints_verified",
    "historical_optuna_audit", "historical_learning_curve_audit",
    "grouped_learning_curve_preflight", "resource_policy_benchmark",
    "historical_inference_parity_reorganization",
)


def _section(name, action):
    print(f"  Verificando {name}...", flush=True)
    try:
        value = action()
    except Exception as exc:
        # Keep detailed exception in local session log, never in shared JSON.
        LOGGER.exception("Development evidence failed: %s", name)
        value = {"state": "ERROR", "error_type": type(exc).__name__,
                 "detail": "See local session log; no repair attempted"}
    print(f"  {name}: {value.get('state', 'OK')}", flush=True)
    return value


def _environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"state": "OK", "python": platform.python_version(), "system": platform.system(),
            "system_release": platform.release(), "machine": platform.machine(),
            "virtualenv_active": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
            "packages": packages}


def _resources():
    from lithiumscope.resources.manager import resource_snapshot
    return {"state": "OK", **resource_snapshot()}


def _architecture(root):
    from lithiumscope.tools.inspection.health import architecture_audit
    data = architecture_audit(root)
    return {"state": "WARNING" if data["files_over_300"] or data["directories_over_20"] else "OK", **data}


def _inventory(root):
    files = [file_record(root, path) for path in ARTIFACTS]
    configs = [file_record(root, f"config/{name}.yaml") for name in
               ("app", "resources", "logging", "model_1", "model_2", "distribution", "prediction")]
    dirs = [directory_record(root, f"results/diagnostics/{name}") for name in DIAGNOSTICS]
    dirs += [directory_record(root, name) for name in (
        "models/model_1/trained", "models/model_2/trained", "results/model_1/runs",
        "results/model_2/runs", "results/predictions", "data/raw/model_2/sentinel2/patches",
        "results/scientific_simulations/m2_rededge_v4_20261113/patches",
        "results/scientific_simulations/sentinel_methodology_20261112/inventory/stac_item_cache",
    )]
    return {"state": "WARNING" if any(r["state"] != "OK" for r in files+configs+dirs) else "OK",
            "files": files, "configs": configs, "directories": dirs}


def collect_snapshot(root: Path = PROJECT_ROOT, *, validation=None) -> dict:
    from lithiumscope.tools.inspection.development.scientific import inspect_m1, inspect_m2
    root = root.resolve()
    # These scientific helpers use the application's configured workspace.
    if root != PROJECT_ROOT.resolve():
        raise ValueError("Development snapshot must use the current application workspace")
    git = _section("Git", lambda: git_identity(root))
    sections = {
        "git": git,
        "environment": _section("Entorno", _environment),
        "resources": _section("Recursos", _resources),
        "architecture": _section("Arquitectura", lambda: _architecture(root)),
        "inventory": _section("Inventario de artefactos", lambda: _inventory(root)),
        "models": _section("Modelos activos", lambda: active_models(root)),
        "tests": _section("Ultima regresion", lambda: latest_test_report(root, git.get("commit"))),
        "model_1": _section("Preflight M1", lambda: inspect_m1(root)),
        "model_2": _section("Preflight M2", lambda: inspect_m2(root)),
    }
    if validation is not None:
        sections["development_validation"] = validation
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "temporary development evidence; not external validation",
            "state": "COMPLETE" if all(x["state"] == "OK" for x in sections.values()) else "PARTIAL",
            "no_training_no_acquisition_no_activation": True, "sections": sections,
            "sharing": "Review this JSON before sharing; no raw samples, credentials, env dump, models or logs attached"}


def write_snapshot(root: Path = PROJECT_ROOT, *, validation=None) -> Path:
    root = root.resolve()
    report = collect_snapshot(root) if validation is None else collect_snapshot(root, validation=validation)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = local_path(root, f"results/diagnostics/development_{stamp}_{uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "development_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def run_interactive() -> None:
    print("\nEVIDENCIAS DE DESARROLLO (TEMPORAL)")
    print("Suite local y comprobaciones; no entrenamiento definitivo, descarga, cambio de modelos ni envio de informacion.")
    from lithiumscope.tools.inspection.development.validation import run_development_validation
    validation = run_development_validation(PROJECT_ROOT)
    path = write_snapshot(validation=validation)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"\nEstado: {report['state']}")
    for name, section in report["sections"].items():
        print(f"  {name:<16} {section['state']}")
    for record in report["sections"].get("inventory", {}).get("files", []):
        if record["state"] != "OK":
            print(f"  [PENDIENTE] {record.get('path', 'archivo no disponible')}: {record['state']}")
    print(f"\nInforme para compartir (revise primero su contenido): {path.relative_to(PROJECT_ROOT.resolve())}")
    print("Los logs y datos originales no se adjuntan. La suite ejecutada y sus rutas constan en el JSON.")
