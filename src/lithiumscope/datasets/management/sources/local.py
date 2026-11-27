"""Resolve configured training inputs without acquisition or reconstruction."""
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import PROJECT_ROOT, DATA_DIR
from lithiumscope.datasets.management.registry import get_dataset_spec
from .model_2 import require_expanded_manifest, require_frozen_manifest


def _existing_local(raw: str | Path, *, root: Path | None = None) -> Path:
    root = Path(root or PROJECT_ROOT).resolve()
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Diagnostic source must belong to this workspace")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local diagnostic source unavailable: {path}; acquisition is not automatic")
    return path


def local_training_source(model_group: str, *, config: dict | None = None,
                          root: Path | None = None, data_dir: Path | None = None,
                          verify_images: bool = False) -> Path:
    if model_group not in {"model_1", "model_2"}:
        raise ValueError(f"Unknown model group: {model_group}")
    cfg = load_config(model_group) if config is None else config
    root = Path(root or PROJECT_ROOT).resolve()
    data_dir = Path(data_dir if data_dir is not None else (root / "data"))
    logger = get_logger("datasets.management.sources.local")
    try:
        if model_group == "model_1":
            sources = cfg["data_sources"]
            georoc = sources.get("georoc", {})
            if georoc.get("enabled", False):
                path = _existing_local(georoc["combined_path"], root=root)
                expected = georoc.get("frozen_combined_sha256")
                if expected and file_sha256(path) != expected:
                    raise ValueError("Frozen combined M1 dataset hash mismatch")
                mode = "frozen_combined" if expected else "combined"
            else:
                spec = get_dataset_spec(sources["base"]["key"])
                path = _existing_local(data_dir / "raw" / spec.model / spec.destination_name, root=root)
                mode = "base"
        else:
            training = cfg["training"]
            if training.get("expansion", {}).get("enabled", False):
                path = require_expanded_manifest(training, root, verify_images=verify_images)
                mode = "expanded"
            else:
                path = require_frozen_manifest(training, root, verify_images=verify_images)
                mode = "frozen"
        logger.info("Source selected model=%s mode=%s path=%s", model_group, mode, path)
        return path
    except (OSError, ValueError, RuntimeError, KeyError):
        logger.warning("Configured source unavailable model=%s", model_group, exc_info=True)
        raise
