from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import yaml

from lithiumscope.core.configuration.schema import validate_config
from lithiumscope.core.paths import CONFIG_DIR


def _ensure_packaged_config_defaults() -> None:
    """Copy bundled defaults only when missing, without overwriting user config.

    Editable/source checkouts continue to read the repository config/ directory;
    installed wheels use the embedded defaults in a writable user workspace.
    """
    packaged = Path(__file__).resolve().parents[2] / "_defaults"
    if not packaged.is_dir():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(packaged.iterdir()):
        if source.is_file() and source.suffix.lower() in {".yaml", ".json"}:
            target = CONFIG_DIR / source.name
            if not target.exists():
                shutil.copyfile(source, target)


def load_config(name: str) -> dict[str, Any]:
    path = Path(name)
    config_name = path.stem
    if not path.suffix:
        config_name = name
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        _ensure_packaged_config_defaults()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return validate_config(config_name, payload)
