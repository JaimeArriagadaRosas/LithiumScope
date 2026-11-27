"""Resolve diagnostic inputs without acquisition or dataset reconstruction."""
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT, DATA_DIR
from lithiumscope.datasets.management.registry import get_dataset_spec


def _existing_local(raw: str | Path) -> Path:
    path = Path(raw)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Diagnostic source must belong to this workspace")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local diagnostic source unavailable: {path}; acquisition is not automatic")
    return path


def local_training_source(model_group: str) -> Path:
    config = load_config(model_group)
    if model_group == "model_1":
        sources = config["data_sources"]
        georoc = sources.get("georoc", {})
        if georoc.get("enabled", False):
            path = _existing_local(georoc["combined_path"])
            expected = georoc.get("frozen_combined_sha256")
            if expected:
                from lithiumscope.core.hashing import file_sha256
                if file_sha256(path) != expected:
                    raise ValueError("Frozen combined M1 dataset hash mismatch")
            return path
        spec = get_dataset_spec(sources["base"]["key"])
        return _existing_local(DATA_DIR / "raw" / spec.model / spec.destination_name)
    if model_group == "model_2":
        training = config["training"]
        expansion = training.get("expansion", {})
        if expansion.get("enabled", False):
            from lithiumscope.datasets.management.sources.model_2 import require_expanded_manifest
            return require_expanded_manifest(training, PROJECT_ROOT)
        from lithiumscope.datasets.management.sources.model_2 import require_frozen_manifest
        return _existing_local(require_frozen_manifest(training, PROJECT_ROOT))
    raise ValueError(f"Unknown model group: {model_group}")
