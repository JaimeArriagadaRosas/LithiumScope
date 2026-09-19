from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lithiumscope.core.config_schema import validate_config
from lithiumscope.core.paths import CONFIG_DIR


def load_config(name: str) -> dict[str, Any]:
    path = Path(name)
    config_name = path.stem
    if not path.suffix:
        config_name = name
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return validate_config(config_name, payload)
