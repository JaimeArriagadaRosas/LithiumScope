"""Conservative reuse of complete local robustness evidence; never rerun predictions here."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR


def image_catalogs(root=None):
    root = root or RESULTS_DIR / "scientific_simulations"
    catalogs = {"pilot": root / "sentinel2_real" / "manifest.csv"}
    for path in sorted(root.glob("**/extended_manifest.csv")):
        catalogs["extended_" + str(len(catalogs))] = path
    return catalogs


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _frame_digest(frame):
    # Include column order and dtypes as well as values and row order.
    return _digest({"columns": list(frame.columns), "dtypes": list(map(str, frame.dtypes)),
                    "values": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()})


def _revision():
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True,
                              capture_output=True, text=True, timeout=20).stdout.strip()
    revision = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=normal", "--",
           "src", "config", "scripts", "examples", "pyproject.toml"):
        raise ValueError("Hay cambios locales de código o configuración; reutilización no certificada")
    return revision


def _environment():
    versions = {"python": sys.version}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "catboost", "xgboost",
                 "torch", "pytorch-tabnet", "rasterio", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _m2_inputs(available):
    from lithiumscope.experiments.model_2.evaluation.descriptor_stress import load_common_matrix
    from lithiumscope.model_2.prediction.spectral_contract import prepare_spectral_input
    frame, ids = load_common_matrix()
    evidence = {"matrix": _frame_digest(frame), "ids": _frame_digest(ids.to_frame()), "catalogs": {}}
    for name, manifest in image_catalogs().items():
        table = pd.read_csv(manifest)
        images = []
        for row in table.to_dict("records"):
            if row.get("status") != "ACCEPT":
                continue
            path = Path(row["path"])
            image = {"sha256": file_sha256(path), "metadata": {}, "features": {}}
            for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
                if sidecar.is_file():
                    image["metadata"][sidecar.name] = file_sha256(sidecar)
            for role, (bundle, _) in available.items():
                features, diagnostics = prepare_spectral_input(path, bundle, allow_acquisition=False)
                image["features"][role] = {"frame": _frame_digest(features), "evidence": diagnostics}
            images.append(image)
        if not images:
            raise ValueError(f"Sin imágenes aceptadas: {name}")
        evidence["catalogs"][name] = {"manifest": file_sha256(manifest), "images": images}
    return evidence


def input_identity(group, available):
    """Return a signature only for verified, complete input identities."""
    try:
        if set(available) != {"release", "candidate"}:
            raise ValueError("Faltan release y candidato verificados")
        models = {}
        for role, (_, identity) in available.items():
            model = {key: identity[key] for key in ("run_id", "algorithm", "sha256", "dataset_sha256")}
            if not all(model.values()):
                raise ValueError(f"Identidad incompleta: {role}")
            models[role] = model
        if group == "model_1":
            inputs = {"cases": file_sha256(PROJECT_ROOT / "examples/integrated_demo/cases.csv")}
        elif group == "model_2":
            inputs = _m2_inputs(available)
        else:
            raise ValueError("Modelo no reconocido")
        return _digest({"schema": 1, "group": group, "models": models,
                        "revision": _revision(), "environment": _environment(), "inputs": inputs}), None
    except Exception as exc:
        # Failure to certify reuse is not a reason to suppress the original comparison.
        return None, f"{type(exc).__name__}: {exc}"


def output_hashes(folder):
    return {path.relative_to(folder).as_posix(): file_sha256(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path != folder / "manifest.json" and not path.is_symlink()}


def find_reusable_robustness(group, *, root=None):
    """Trust only complete new manifests with matching inputs and all output hashes."""
    root = root or RESULTS_DIR / "scientific_simulations"
    candidates = []
    for path in sorted(root.glob("sim_*/manifest.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("state") == "COMPLETADO" and record.get("groups") == [group]
                    and record.get("include_saved_oof") is False and record.get("input_identity")
                    and record.get("output_hashes")):
                candidates.append((path, record))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    if not candidates:
        return None, "No existe una ejecución de robustez completa con identidad verificable"
    from lithiumscope.experiments.shared.comparison_models import discover_models
    available, _ = discover_models(group)
    identity, reason = input_identity(group, available)
    if identity is None:
        return None, reason
    for path, record in candidates:
        try:
            if (record["input_identity"] == identity
                    and record["output_hashes"] == output_hashes(path.parent)):
                return path, None
        except (OSError, ValueError):
            continue
    return None, "Las identidades de entradas o archivos de robustez no coinciden"
