from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib

from lithiumscope.core.paths import MODELS_DIR


def save_model_bundle(
    model_group: str,
    name: str,
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    trained_dir = MODELS_DIR / model_group / "trained"
    metadata_dir = MODELS_DIR / model_group / "metadata"
    trained_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = trained_dir / f"{name}_{stamp}.joblib"
    metadata_path = metadata_dir / f"{name}_{stamp}.json"

    joblib.dump(bundle, model_path)
    payload = {
        **metadata,
        "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
        "saved_at_utc": stamp,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_path, metadata_path
