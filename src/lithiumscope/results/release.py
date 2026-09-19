from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.results.catalog import latest_release_candidate


def build_release_candidate_manifest() -> Path:
    model_1 = latest_release_candidate("model_1")
    model_2 = latest_release_candidate("model_2")

    if model_1 is None or model_2 is None:
        missing = []
        if model_1 is None:
            missing.append("model_1")
        if model_2 is None:
            missing.append("model_2")
        raise RuntimeError(
            "No existe una ejecución completa apta para versionar en: "
            + ", ".join(missing)
        )

    if model_1["git_commit"] != model_2["git_commit"]:
        raise RuntimeError(
            "Los candidatos de Modelo 1 y Modelo 2 fueron generados con commits "
            "distintos. Ejecute ambos modelos sobre el mismo código antes de crear un tag."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "schema_version": 1,
        "status": "candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": model_1["git_commit"],
        "suggested_tag": "lithiumscope-" + str(model_1["run_id"]),
        "model_1": model_1,
        "model_2": model_2,
        "note": (
            "This file is only a local release candidate manifest. "
            "It does not create a GitHub tag."
        ),
    }

    destination = RESULTS_DIR / "release_candidates" / f"{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return destination
