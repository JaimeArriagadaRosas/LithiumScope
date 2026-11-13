"""Recoverable six-algorithm M2 OOF diagnostic, including VP/VN/FP/FN.

Runs from saved competition artefacts; does not fit, alter a trained model,
select a new threshold, edit the interactive menu or generate any PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training.diagnostic_inputs import (
    loaded_competitions, release_contract,
)
from lithiumscope.model_2.training.diagnostic_metrics import (
    binary_diagnostic, reliability_rows, sample_outcomes,
    spatial_bootstrap_difference,
)
from lithiumscope.model_2.training.diagnostic_release_comparison import (
    compare_to_release_protocol,
)

COMMON_THRESHOLD = 0.5


def analyse_saved_competition(
    manifest_path: Path,
    tables: Path,
    destination: Path,
    *,
    root: Path = PROJECT_ROOT,
    older_tables: Path | None = None,
    verify_release: bool = True,
    bootstrap: int = 200,
) -> dict:
    """Fixed-threshold diagnosis and error attribution across all candidates."""
    manifest, new, old, info = loaded_competitions(
        manifest_path, tables, root, older_tables=older_tables,
    )
    target = np.asarray(info["y"], int)
    folds = np.asarray(info["folds"], int)
    group = manifest.spatial_group.astype(str).to_numpy()
    cohort = np.where(np.arange(len(manifest)) < 706,
                      "historical", "GEOROC_provisional")
    audit_path = tables / "scientific_sample_audit_783.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else None
    if audit is not None and (
        len(audit) != len(manifest)
        or list(audit.sample_id.astype(str)) != list(manifest.sample_id.astype(str))
    ):
        raise ValueError("Scientific QA audit does not match M2 source samples")
    extra_masks: dict[str, np.ndarray] = {
        "all": np.ones(len(target), dtype=bool),
        "historical": cohort == "historical",
        "GEOROC_provisional": cohort == "GEOROC_provisional",
    }
    if audit is not None:
        extra_masks.update({
            "QA95_failed_but_test_kept":
                ~audit.train_quality_pass_95.astype(bool).to_numpy(),
            "SCL_abstention":
                ~audit.SCL_proxy_accepted.astype(bool).to_numpy(),
            "OOD_above_25pct":
                audit.OOD_above_25pct.astype(bool).to_numpy(),
            "mixed_label_site":
                audit.mixed_label_site_at_Li21p6.astype(bool).to_numpy(),
            "latitude_20S_30S":
                audit.latitude.between(-30, -20, inclusive="neither").to_numpy(),
            "latitude_10S_20S":
                audit.latitude.between(-20, -10, inclusive="neither").to_numpy(),
        })
    for fold in sorted(np.unique(folds)):
        extra_masks[f"outer_fold_{fold}"] = folds == fold

    current_rows = []
    comparison = []
    reliability = []
    samples = []
    strata = []
    for name in sorted(new):
        probability = np.asarray(new[name], float)
        overall = binary_diagnostic(target, probability)
        current_rows.append({"algorithm": name, "version": "current", **overall})
        reliability.extend(reliability_rows(name, target, probability))
        outcome = sample_outcomes(target, probability)
        for idx in range(len(target)):
            rows = {
                "algorithm": name,
                "sample_id": str(manifest.sample_id.iloc[idx]),
                "index": idx, "fold": int(folds[idx]),
                "spatial_group": str(group[idx]), "cohort": str(cohort[idx]),
                "y_true": int(target[idx]), "score": float(probability[idx]),
                "outcome_at_0p5": str(outcome[idx]),
                "squared_error": float((probability[idx] - target[idx]) ** 2),
            }
            if old:
                before = float(old[name][idx])
                rows["previous_score"] = before
                rows["brier_delta_new_minus_previous"] = (
                    rows["squared_error"] - (before - target[idx]) ** 2
                )
            samples.append(rows)
        for subset_name, mask in extra_masks.items():
            if mask.any():
                strata.append({
                    "algorithm": name, "subset": subset_name,
                    **binary_diagnostic(target[mask], probability[mask])
                })
        if old:
            prior = np.asarray(old[name], float)
            before_metrics = binary_diagnostic(target, prior)
            ci = spatial_bootstrap_difference(
                target, prior, probability, group, n=bootstrap, seed=20261112
            )
            comparison.append({
                "algorithm": name,
                "roc_auc_old": before_metrics["roc_auc"],
                "roc_auc_new": overall["roc_auc"],
                "average_precision_old": before_metrics["average_precision"],
                "average_precision_new": overall["average_precision"],
                "brier_old": before_metrics["brier"],
                "brier_new": overall["brier"],
                "brier_pos_old": before_metrics["brier_positive"],
                "brier_pos_new": overall["brier_positive"],
                "brier_neg_old": before_metrics["brier_negative"],
                "brier_neg_new": overall["brier_negative"],
                "FP_old": before_metrics["FP"], "FP_new": overall["FP"],
                "FN_old": before_metrics["FN"], "FN_new": overall["FN"],
                "delta_roc_auc": ci["roc_auc"]["delta"],
                "delta_ap": ci["average_precision"]["delta"],
                "delta_brier": ci["brier"]["delta"],
                "roc_auc_ci95_low": (
                    ci["roc_auc"]["ci95"][0]
                    if ci["roc_auc"]["ci95"] else None
                ),
                "roc_auc_ci95_high": (
                    ci["roc_auc"]["ci95"][1]
                    if ci["roc_auc"]["ci95"] else None
                ),
                "ap_ci95_low": (
                    ci["average_precision"]["ci95"][0]
                    if ci["average_precision"]["ci95"] else None
                ),
                "ap_ci95_high": (
                    ci["average_precision"]["ci95"][1]
                    if ci["average_precision"]["ci95"] else None
                ),
                "brier_ci95_low": (
                    ci["brier"]["ci95"][0]
                    if ci["brier"]["ci95"] else None
                ),
                "brier_ci95_high": (
                    ci["brier"]["ci95"][1]
                    if ci["brier"]["ci95"] else None
                ),
            })
    destination.mkdir(parents=True, exist_ok=True)
    tables_out = {
        "confusion_matrices_all_algorithms.csv": current_rows,
        "confusion_by_region_quality_ood_fold.csv": strata,
        "calibration_reliability_bins.csv": reliability,
        "sample_error_diagnostics.csv": samples,
        "paired_competition_deltas.csv": comparison,
    }
    for filename, records in tables_out.items():
        pd.DataFrame(records).to_csv(destination / filename, index=False)

    historical_comparison = compare_to_release_protocol(
        tables, destination, repeats=bootstrap,
    )
    published = release_contract(root) if verify_release else {
        "state": "verification_disabled_for_synthetic_test"
    }
    old_reference_path = tables / "old_release_protocol_paired_706.csv"
    reference_status = (
        "historical_retrained_protocol_not_fitted_release"
        if old_reference_path.is_file() else
        "historical_retrained_protocol_not_present"
    )
    verdict = {
        "state": "completed_exploratory_diagnostic",
        "output_dir": str(destination),
        "rows": len(manifest), "algorithms": sorted(new),
        "source_run": str(tables),
        "source_provenance": info["metadata"],
        "reference_release": published,
        "legacy_protocol_reference": reference_status,
        "historical_706_all_algorithm_comparison": historical_comparison,
        "common_confusion_threshold": COMMON_THRESHOLD,
        "threshold_is_fixed_descriptive_not_optimized_on_test": True,
        "VP_VN_FP_FN_meaning": {
            "VP": "true positive", "VN": "true negative",
            "FP": "false positive", "FN": "false negative",
        },
        "model_scores_are_not_geological_deposit_probabilities": True,
        "operating_threshold_requires_strict_inner_train_only_fit": True,
        "no_release_authorization": True,
        "no_menu_change": True,
        "did_not_fit_models": True,
        "candidate_choosing_remains_six_algorithm_competition": True,
    }
    (destination / "diagnostic_manifest.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summaries = [
        "# Diagnóstico multidimensional Modelo 2",
        "",
        "**Estado:** exploratorio, no constituye nuevo holdout ni aprobación de release.",
        "",
        "La matriz VP/VN/FP/FN usa **umbral fijo 0,5** para los seis algoritmos.",
        "No se ha optimizado un umbral operativo con etiquetas del test.",
        "En la siguiente competencia, la selección de umbrales debe ocurrir",
        "únicamente dentro de los folds de entrenamiento.",
        "",
        "| Algoritmo | VP | VN | FP | FN | AUC | AP | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in current_rows:
        summaries.append(
            f"| {item['algorithm']} | {item['VP']} | {item['VN']} | "
            f"{item['FP']} | {item['FN']} | {item['roc_auc']:.4f} | "
            f"{item['average_precision']:.4f} | {item['brier']:.4f} |"
        )
    summaries.extend([
        "",
        "La clasificación al umbral 0,5 no equivale al ranking de sitios.",
        "AUC/AP miden ordenación; Brier y tablas de calibración miden el ajuste",
        "de scores frente a las etiquetas disponibles, no la probabilidad física de depósito.",
        "",
        "Archivos: confusion_by_region_quality_ood_fold.csv,",
        "sample_error_diagnostics.csv, calibration_reliability_bins.csv",
        "y paired_competition_deltas.csv.",
        "",
        "**Limitación:** 783 muestras de desarrollo con 77 GEOROC provisionales;",
        "no hay conjunto externo independiente ni tasa Top-1@3 condicionada a M1.",
    ])
    (destination / "DIAGNOSTICO_MODELO2.md").write_text(
        "\n".join(summaries) + "\n", encoding="utf-8"
    )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    root = PROJECT_ROOT

    def resolve(path):
        return path if path.is_absolute() else root / path

    result = analyse_saved_competition(
        resolve(args.manifest), resolve(args.tables), resolve(args.output),
        root=root, bootstrap=args.bootstrap,
    )
    print("M2_DIAGNOSTIC_COMPLETED", json.dumps({
        "rows": result["rows"], "algorithms": result["algorithms"],
        "output_dir": result["output_dir"],
    }), flush=True)


if __name__ == "__main__":
    main()
