"""Read-only engineering smoke: expanded November M1 + September M2 release.

Ten pre-existing demo cases, NOT an external holdout. Never activates models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_1.prediction.predictor import predict_model_1_frame
from lithiumscope.model_2.prediction.predictor import score_model_2_image
from lithiumscope.prediction.demo import (
    audit_demo_overlap,
    demonstration_imagery_cache,
    load_demonstration_cases,
)

OUT = PROJECT_ROOT / "results/scientific_simulations/m2_rededge_v4_20261113/integration_m1nov_m2sep"
M1 = PROJECT_ROOT / "models/model_1/trained/training_20261106_111653_m0300/model.joblib"
M2 = PROJECT_ROOT / "models/model_2/trained/training_20260920_195933_m0300/model.joblib"


def integrate(*, output: Path = OUT) -> dict:
    registry = PROJECT_ROOT / "models/active_models.json"
    active_hash = file_sha256(registry)
    cases = load_demonstration_cases()
    if len(cases) != 10 or cases.case_id.duplicated().any():
        raise ValueError("Expected ten distinct pre-existing demonstration cases")
    m1, m2 = joblib.load(M1), joblib.load(M2)
    if m1["algorithm"] != "svm" or m2["algorithm"] != "catboost":
        raise ValueError("Unexpected M1/M2 model identities")
    if len(m2["features"]) != 126:
        raise ValueError("Original M2 schema differs")
    frame, _ = predict_model_1_frame(cases, bundle=m1)
    rows = []
    for item in frame.itertuples(index=False):
        image = demonstration_imagery_cache() / f"{item.case_id}.npy"
        if not image.is_file():
            raise FileNotFoundError(image)
        pred, diag = score_model_2_image(image, bundle=m2)
        score = float(pred["prospectivity_score"])
        if not np.isfinite(score):
            raise ValueError("M2 release produced no finite score")
        rows.append(
            {
                "case_id": str(item.case_id),
                "Li_observed": float(item.Li_icpms),
                "M1_estimated_Li_ppm": float(item.Li_icpms_predicted),
                "M1_out_of_domain": float(item.out_of_training_range_fraction),
                "M2_spectral_score": score,
                "M2_out_of_domain": float(pred["out_of_training_range_fraction"]),
                "M2_missing_input_features": json.dumps(diag["missing_features"]),
                "image_sha256": file_sha256(image),
            }
        )
    scored = pd.DataFrame(rows)
    top = scored.sort_values("M1_estimated_Li_ppm", ascending=False).head(3)
    if len(top) != 3 or top.case_id.duplicated().any():
        raise ValueError("Not three distinct M1-selected candidates")
    m1_choice = str(top.iloc[0].case_id)
    m2_choice = str(top.sort_values("M2_spectral_score", ascending=False).iloc[0].case_id)
    best = str(top.sort_values("Li_observed", ascending=False).iloc[0].case_id)
    meta = {
        "status": "integration_demo_only_not_external_validation",
        "m1_three_candidates": top.case_id.tolist(),
        "m1_top1": m1_choice,
        "m2_top1_of_m1_three": m2_choice,
        "observed_best_of_m1_three": best,
        "m1_selected_observed_best": m1_choice == best,
        "m2_selected_observed_best": m2_choice == best,
        "cases": len(scored),
        "m1_ood_count": int((scored.M1_out_of_domain > 0.25).sum()),
        "m2_ood_count": int((scored.M2_out_of_domain > 0.25).sum()),
        "m2_missing_input_features": int(scored.M2_missing_input_features.ne("[]").sum()),
        "m1_model_sha256": file_sha256(M1),
        "m2_model_sha256": file_sha256(M2),
        "demo_overlap": audit_demo_overlap(cases),
        "not_holdout": True,
        "no_activation": True,
    }
    if file_sha256(registry) != active_hash:
        raise ValueError("Active registry changed during read-only smoke test")
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "ten_cases_predictions.csv", index=False)
    top.to_csv(output / "m1_top_three.csv", index=False)
    (output / "integration_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Integración de candidato M1 ampliado y M2 release original",
        "",
        "**Prueba de ingeniería con diez casos reutilizados, no validación externa.**",
        f"Los 3 candidatos de M1: {', '.join(meta['m1_three_candidates'])}.",
        f"Top 1 según M1: {m1_choice}.",
        f"Top 1 según M2 release entre esos tres: {m2_choice}.",
        f"Top 1 según Li real entre esos tres: {best}.",
        f"Casos M1 fuera de dominio: {meta['m1_ood_count']}/10.",
        f"Casos M2 fuera de dominio: {meta['m2_ood_count']}/10.",
        "No utilizar esta demostración para declarar un release superior.",
    ]
    (output / "INTEGRACION_DEMO.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("M1_NOV_M2_SEP_INTEGRATION", json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


if __name__ == "__main__":
    integrate()
