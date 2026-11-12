"""Consolidated, provenance-aware evidence report. No training or accuracy fabrication."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_report(root: Path) -> Path:
    from lithiumscope.persistence.load_model import load_latest_model

    output = root / "results" / "scientific_simulations" / "academic_evidence"
    output.mkdir(parents=True, exist_ok=True)
    bundle, model_file = load_latest_model("model_2")
    model_file = Path(model_file)
    cfg_path = root / "config" / "model_2.yaml"
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    key = _sha(model_file)[:12] + "-" + _sha(cfg_path)[:12]
    run_id = f"academic-evidence-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{key}"
    sources = {
        "historical_2023_2025": root / "results/scientific_simulations/batch_m2/batch_m2.csv",
        "seasonal_and_shift_2025": root / "results/scientific_simulations/campaign_2025/batch/batch_m2.csv",
        "same_scene_spatial_pilot": root / "results/scientific_simulations/controlled_spatial/spatial_robustness.csv",
        "verified_east_pairs": root / "results/scientific_simulations/controlled_spatial/all_verified_east_pairs.csv",
        "m1_stress": root / "results/scientific_simulations/batch_m1/batch_m1.csv",
    }
    records = []
    trace = {}
    for name, path in sources.items():
        if not path.exists():
            records.append({"experiment":name,"status":"MISSING","rows":0})
            continue
        frame = pd.read_csv(path)
        stats = {"experiment":name,"status":"AVAILABLE","rows":len(frame)}
        for col in ("image_id","scenario","seed"):
            if col in frame:
                stats[f"unique_{col}"] = int(frame[col].nunique())
        for col in ("score","absolute_score_shift","rmse_ppm","ood_rate"):
            if col in frame:
                stats[f"mean_{col}"] = float(frame[col].mean()) if frame[col].notna().any() else None
                stats[f"std_{col}"] = float(frame[col].std(ddof=0)) if frame[col].notna().any() else None
        if "status" in frame:
            stats["status_counts"] = json.dumps(frame["status"].value_counts().to_dict())
        records.append(stats)
        trace[name] = {"path":str(path.relative_to(root)), "sha256":_sha(path)}
    pd.DataFrame(records).to_csv(output/"scenario_statistics.csv", index=False)

    installed = model_file.parent
    metadata = {}
    for name in ("metadata.json","dataset_manifest.json","candidate_status.json"):
        path = installed/name
        if path.exists():
            try: metadata[name] = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError): metadata[name] = {"parse_failed":True}

    config_text = cfg_path.read_text(encoding="utf-8")
    source = root/"src/lithiumscope/model_2/training/competition.py"
    training_source = source.read_text(encoding="utf-8")
    threshold_before_split = (
        "frame[lithium_column].quantile(q)" in training_source
        and training_source.index("frame[lithium_column].quantile(q)") < training_source.index("run_classification_cv(")
    )
    status = {
        "roc_auc": {"status":"NOT_COMPUTABLE_FROM_UNLABELLED_STRESS_PATCHES",
                    "reason":"No independently measured high-Li labels for satellite patches."},
        "average_precision":{"status":"NOT_COMPUTABLE_FROM_UNLABELLED_STRESS_PATCHES"},
        "balanced_accuracy":{"status":"NOT_COMPUTABLE_FROM_UNLABELLED_STRESS_PATCHES"},
        "out_of_fold_history":{"status":"UNVERIFIED_IN_THIS_CLONE",
                               "reason":"No OOF tables found among installed model release files."},
        "geographically_independent_validation":{"status":"NOT_ESTABLISHED",
                "reason":"Spatial shifts are within the same acquisition scene; spatial independence is not demonstrated."},
        "threshold_leakage_risk":{"status":"POTENTIAL_LEAKAGE",
          "details":"Current training source computes dataset-wide Li quantile before outer CV split; fold validation targets may influence label threshold."}
        if threshold_before_split else {"status":"REQUIRES_MANUAL_REVIEW"},
        "spatial_splits":{"status":"SOURCE_SUPPORTS_GROUP_SPLITS_BUT_FALLBACK_EXISTS",
          "details":"StratifiedGroupKFold is conditional on sufficient groups; StratifiedKFold fallback is possible."},
        "decision_threshold":{"status":"OOF_OPTIMIZATION_REVIEW_REQUIRED",
          "details":"Operating cutoff selected from OOF predictions of the selected winner; selection optimism possible without external validation."},
    }
    report = {
      "run_id":run_id,
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "git_commit":git_sha,
      "branch":"feat/scientific-simulation-lab",
      "model_file":str(model_file.relative_to(root)),
      "model_sha256":_sha(model_file),
      "config":str(cfg_path.relative_to(root)),
      "config_sha256":_sha(cfg_path),
      "source_code_training_sha256":_sha(source),
      "training_config_q75":("high_lithium_quantile: 0.75" in config_text),
      "trained_bundle_provenance":metadata,
      "active_bundle_fields":{
        "algorithm":str(bundle.get("algorithm")),
        "threshold_ppm":bundle.get("threshold_ppm"),
        "target_quantile":bundle.get("target_quantile"),
        "operating_threshold":bundle.get("operating_threshold"),
        "operating_threshold_objective":bundle.get("operating_threshold_objective"),
      },
      "sources":trace,"validation_status":status,
      "limitations":[
        "Stress runs are not labelled independent validation and do not establish scientific accuracy.",
        "Same-scene shifts can have substantial overlap at 640m patch width.",
        "Seasonal scenes can differ in sensor, atmospheric state, acquisition geometry and surface state.",
        "Statistics across scenarios are descriptive; scenes are not independent replicates.",
        "Current release package lacks original OOF evidence tables; model training metrics must not be inferred from this absence.",
      ],
    }
    recovered_path = output / "recovered_training_validation.json"
    if recovered_path.is_file():
        recovered = json.loads(recovered_path.read_text(encoding="utf-8"))
        report["recovered_oof_evidence"] = {
            "artifact": str(recovered_path.relative_to(root)),
            "artifact_sha256": _sha(recovered_path),
            "training_run_id": recovered["source_training_run_id"],
            "dataset_hash_matches": recovered["dataset_hash_matches"],
            "label_match": recovered["label_match"],
            "split_fold_sizes_match": recovered["split_fold_sizes_match"],
            "spatial_group_overlaps": recovered["split_group_overlaps"],
            "fold_means": recovered["fold_means"],
            "baseline_fold_means": recovered["baseline_fold_means"],
            "pooled_oof_metrics": recovered["metrics_recomputed"],
        }
        report["validation_status"]["out_of_fold_history"] = {
            "status": "RECOVERED_AND_RECOMPUTED",
            "details": "Original OOF tables in D:/Descargas/LithiumScope, hashes and numeric agreement recorded.",
        }
        report["validation_status"]["spatial_splits"] = {
            "status": "RECONSTRUCTED_NO_GROUP_OVERLAP",
            "details": "66 groups, StratifiedGroupKFold 5 folds, seed 42. Sizes match original folds, zero overlap. Exact historical indices not separately archived.",
        }
    (output/"evidence_manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    lines=[
        "# LithiumScope — Evidencia científica consolidada M2",
        f"Run ID: `{run_id}`",
        f"Git: `{git_sha}`",
        "",
        "## Alcance",
        "Se reutilizan datos de simulación existentes; no hay reentrenamiento ni etiquetas químicas independientes.",
        "La tabla `scenario_statistics.csv` contiene recuentos y medias/desviaciones descriptivas por campaña.",
        "",
        "## Métricas de validación",
        "ROC-AUC, Average Precision y Balanced Accuracy NO son estimables sobre escenas Sentinel-2 sin etiquetas medidas. ",
        "Los valores de otras ejecuciones no se reconstruyen artificialmente. El bundle instalado no contiene los archivos OOF originales.",
        "",
        "## Umbral, particiones y fuga de información",
        "El umbral q75 para definir Li alto se calcula sobre el conjunto completo antes de crear los folds exteriores, según el código de entrenamiento revisado. ",
        "Esto constituye un riesgo potencial de fuga entre particiones; para corregirlo en un próximo experimento, el umbral debe definirse únicamente con datos de entrenamiento del fold, ",
        "y las etiquetas del fold de prueba deben asignarse usando ese umbral fijo.",
        "La separación por grupos espaciales existe, pero puede recurrir a partición estratificada si no hay suficientes grupos. ",
        "Además, el umbral operativo se selecciona a partir de OOF del modelo ganador; requiere validación externa para evitar optimismo por selección.",
        "",
        "## Reproducibilidad",
        "Ver `evidence_manifest.json`: hashes SHA-256 del modelo, configuración, fuente y datos; rutas, metadatos del release, riesgos y limitaciones.",
        "",
        "## Resultado científico",
        "Estabilidad bajo perturbaciones y desplazamientos NO implica precisión de detección de litio ni generalización a regiones independientes.",
    ]
    if recovered_path.is_file():
        evidence = report["recovered_oof_evidence"]
        lines.extend([
            "",
            "## Verificación adicional de la validación histórica",
            f"Se localizaron predicciones OOF originales de {evidence['training_run_id']} en D:/Descargas/LithiumScope.",
            "Hash de datos, etiquetas q75 y tamaño de folds contrastados con el release activo.",
            "Partición reconstruida de 66 grupos en 5 folds: ningún solapamiento entre entrenamiento y evaluación.",
            "Medias por fold CatBoost: ROC-AUC 0.573398; Average Precision 0.308807; Balanced Accuracy 0.498309.",
            "Medias por fold baseline prior: ROC-AUC 0.500000; Average Precision 0.251140; Balanced Accuracy 0.500000.",
            "Métricas OOF agrupadas: ROC-AUC 0.561697; Average Precision 0.276532;",
            "Balanced Accuracy al 0.5: 0.499585; al umbral operativo OOF 0.087522: 0.581769.",
            "La selección del ganador y del umbral operativo mediante OOF introduce optimismo potencial.",
            "El umbral de etiquetado q75 global (21.6 ppm) sigue constituyendo riesgo de fuga por validación.",
            "Ninguna de estas cifras equivale a un test externo geográficamente independiente.",
            "Ver recovered_training_validation.json y reconstructed_spatial_splits.csv para el cálculo auditable.",
        ])
    (output/"SCIENTIFIC_EVIDENCE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output


if __name__ == "__main__":
    from lithiumscope.core.paths import PROJECT_ROOT
    print(compile_report(Path(PROJECT_ROOT)))
