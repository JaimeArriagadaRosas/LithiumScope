from __future__ import annotations

from typing import Any

CURRENT_SCHEMA_VERSION = 1

_REQUIRED_SECTIONS = {
    "app": {"app", "paths"},
    "model_1": {"model", "data", "validation", "optimization", "competition"},
    "model_2": {"model", "imagery", "training", "validation", "competition"},
}


class ConfigurationSchemaError(ValueError):
    pass


def validate_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigurationSchemaError(
            f"Config {name!r} uses schema_version={version!r}; "
            f"expected {CURRENT_SCHEMA_VERSION}."
        )

    required = _REQUIRED_SECTIONS.get(name, set())
    missing = sorted(section for section in required if section not in payload)
    if missing:
        raise ConfigurationSchemaError(
            f"Config {name!r} is missing required sections: {', '.join(missing)}"
        )
    return payload
