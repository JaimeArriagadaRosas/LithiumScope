"""Read-only presentation of existing EDA evidence, independent of scientific code."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.results.eda_artifacts import verify_eda


def _table(title, frame, columns=None):
    print(f"\n--- {title} ---")
    if columns is not None:
        frame = frame[[name for name in columns if name in frame]]
    if frame.empty:
        print("Sin registros en esta tabla.")
        return
    print(frame.to_string(index=False, na_rep="N/D", float_format=lambda v: f"{v:.4g}"))


def _read(directory, manifest, name, **kwargs):
    relative = f"tables/{name}.csv"
    if relative not in manifest["artifacts"]:
        return None
    return pd.read_csv(directory / relative, **kwargs)


def _features(directory, manifest):
    profiles = _read(directory, manifest, "feature_profiles")
    if profiles is None:
        print("Perfiles de variables: no disponibles.")
        return
    prepared = profiles.loc[profiles["stage"].eq("prepared")].copy()
    prepared["missing_pct"] = 100 * prepared["missing"] / prepared["rows"].where(prepared["rows"].gt(0))
    raw = profiles.loc[profiles["stage"].eq("raw"), ["source", "feature", "missing", "rows"]].copy()
    if not raw.empty:
        raw["raw_missing_pct"] = 100 * raw["missing"] / raw["rows"].where(raw["rows"].gt(0))
        prepared = prepared.merge(raw[["source", "feature", "raw_missing_pct"]],
                                  on=["source", "feature"], how="left", validate="one_to_one")
    for source, table in prepared.groupby("source", sort=True):
        _table(f"Variables preparadas por fuente: {source}", table, [
            "feature", "raw_missing_pct", "missing_pct", "unique_present", "numeric_nonfinite",
            "min", "median", "max", "std"])
        high = table.loc[table["missing_pct"].ge(50), "feature"].tolist()
        constant = table.loc[table["unique_present"].le(1), "feature"].tolist()
        print(f"Alerta descriptiva >=50% faltantes: {len(high)} | " + (", ".join(high) or "ninguna"))
        print(f"Constantes o vacías en esta fuente: {len(constant)} | " + (", ".join(constant) or "ninguna"))
    train = _read(directory, manifest, "outer_train_features")
    if train is not None and not train.empty:
        train = train.copy()
        train["missing_pct"] = 100 * train["train_missing"] / train["train_eligible_rows"].where(train["train_eligible_rows"].gt(0))
        summary = train.groupby("feature", sort=True).agg(
            train_missing_pct_max=("missing_pct", "max"),
            train_unique_min=("train_unique_present", "min"),
            folds_all_missing=("train_all_missing", "sum")).reset_index()
        _table("Variables: peor cobertura observada SOLO TRAIN por fold", summary)
    correlations = _read(directory, manifest, "spearman_correlations", index_col=0)
    if correlations is not None and "Li_icpms" in correlations:
        values = pd.to_numeric(correlations["Li_icpms"], errors="coerce").drop(labels=["Li_icpms"], errors="ignore")
        order = values.abs().sort_values(ascending=False).index
        _table("Correlación Spearman descriptiva con Li (no selección de variables)",
               values.reindex(order).rename("rho").rename_axis("feature").reset_index())
    redundancy = _read(directory, manifest, "spectral_redundancy")
    if redundancy is not None:
        _table("Redundancia espectral exacta y constantes", redundancy)


def show_eda_summary(directory: Path) -> bool:
    """Render verified existing tables without rebuilding or changing any artifact."""
    try:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not verify_eda(directory, manifest["signature"]):
            raise ValueError("Integridad del EDA no verificada")
        print(f"\n=== EDA DETALLADO · {manifest['model_group'].upper()} ===")
        print(f"Estado: {manifest['state']} | Filas originales: {manifest['rows_raw']} | "
              f"Filas preparadas: {manifest['rows_prepared']} | Variables: {manifest['features']}")
        print("Datos descriptivos; no hay ajuste de modelos ni selección automática de variables.")
        print("Reutilización: evidencia descriptiva compatible; el entrenamiento verifica datos y folds.")
        print("Imputación, escalado y transformaciones aprendidas se ajustan exclusivamente en TRAIN.")
        config = manifest.get("effective_config", {})
        validation = config.get("validation", {})
        print(f"Validación: {manifest.get('fold_profile_state', 'N/D')} | "
              f"Folds internos configurados: {validation.get('inner_folds', 'N/D')}")
        for name, title, columns in (
            ("preparation_stages", "Preparación y pérdidas por etapa", ["stage", "rows", "columns", "missing_cells"]),
            ("cohort_summary", "Concentración de Li por fuente (ppm)", None),
            ("target_tails", "Colas de Li por fuente (ppm)", None),
            ("qc_recovery_summary", "Exclusiones QC y recuperación potencial (no aprobada)", None),
            ("target_analytical_summary", "Estado analítico del Li", None),
            ("target_bands_by_source", "Representatividad por fuente y concentración", None),
            ("compositional_readiness", "Disponibilidad de óxidos para análisis composicional", None),
            ("categorical_source_association", "Asociación categórica con fuente", None),
            ("outer_identity_support_summary", "Identidad y soporte químico TRAIN por fold", None),
            ("spatial_buffer_sensitivity", "Sensibilidad a separación nominal (sin cambiar entrenamiento)", None),
            ("outer_train_summary", "Distribución y balance de TRAIN por fold", None),
            ("outer_spatial_summary", "Separación espacial TRAIN/TEST (metros)", [
                "fold", "minimum_fit_m", "median_fit_m", "within_100m", "within_1km", "within_10km",
                "sentinel_scene_id_shared_test_rows", "possible_patch_overlap_test_rows", "minimum_required_m", "distance_policy_pass"]),
        ):
            frame = _read(directory, manifest, name)
            if frame is not None:
                _table(title, frame, columns)
        coordinates = manifest.get("coordinate_report")
        if coordinates:
            print("\nCoordenadas: " + " | ".join(
                f"{k}={coordinates[k]}" for k in ("valid_rows", "unknown_crs_rows",
                    "conflicting_alias_rows", "coordinate_accuracy_certified") if k in coordinates))
        quality = manifest.get("quality_audit")
        if quality:
            print("\nQA espectral: " + " | ".join(
                f"{k}={quality[k]}" for k in ("training_eligible", "training_excluded",
                    "all_outer_test_retained", "policy") if k in quality))
            threshold = config.get("model", {}).get("high_lithium_quantile")
            print(f"Cuantil objetivo: {threshold if threshold is not None else 'N/D'}; etiquetas determinadas por fold.")
        _features(directory, manifest)
        print("\n--- Interpretación y límites ---")
        print("Las alertas de faltantes, constantes, extremos, correlación y cercanía espacial son riesgos, no pruebas de fuga.")
        print("Sobreajuste e infraajuste: NO EVALUADOS. Requieren métricas TRAIN/validación y curvas de aprendizaje.")
        print("No se eliminaron variables ni se modificaron datasets. Tablas y figuras completas: " + str(directory))
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[AVISO] No se pudo presentar el resumen EDA: {exc}")
        return False
