"""Read-only synthesis of paired spectral perturbations, spatial and temporal variance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_experiments(base: Path) -> Path:
    out = base / "scenario_analysis"
    out.mkdir(parents=True, exist_ok=True)
    spatial_file = base / "controlled_spatial" / "all_684_scenario_metrics.csv"
    spatial_manifest = base / "controlled_spatial" / "spatial_manifest.csv"
    historical_file = base / "batch_m2" / "batch_m2.csv"
    seasonal_file = base / "campaign_2025" / "batch" / "batch_m2.csv"
    seasonal_manifest = base / "campaign_2025" / "campaign_manifest.csv"
    history_manifest = base / "sentinel2_real" / "manifest.csv"

    spatial = pd.read_csv(spatial_file).merge(
        pd.read_csv(spatial_manifest)[["image_id", "region", "direction", "distance_m", "scene_id"]],
        on="image_id", validate="many_to_one")
    spatial["run_type"] = "spatial_same_scene"
    historical = pd.read_csv(historical_file).merge(
        pd.read_csv(history_manifest)[["image_id", "latitude", "longitude", "scene_datetime"]],
        on="image_id", validate="many_to_one")
    historical["region"] = historical["image_id"].str.rsplit("_", n=1).str[0]
    historical["run_type"] = "interannual"
    seasonal = pd.read_csv(seasonal_file).merge(
        pd.read_csv(seasonal_manifest)[["image_id", "experiment_group", "scene_datetime"]],
        on="image_id", validate="many_to_one")
    seasonal["region"] = seasonal["image_id"].str.split("_").str[0]
    seasonal["run_type"] = "seasonal_shifted"
    datasets = [spatial, historical, seasonal]
    per_case = []
    summaries = []
    for df in datasets:
        group = df["run_type"].iloc[0]
        baselines = df.loc[df.scenario.eq("baseline"), ["image_id", "score"]].rename(
            columns={"score": "baseline_score"})
        scores = df.merge(baselines, on="image_id", validate="many_to_one")
        scores["paired_shift"] = scores["score"] - scores["baseline_score"]
        scores["abs_paired_shift"] = scores["paired_shift"].abs()
        scores["score_available"] = scores["status"].eq("SCORED")
        per_case.append(scores)
        for (scenario, kind), x in scores.groupby(["scenario", "kind"], sort=True):
            values = x["abs_paired_shift"].dropna()
            summaries.append({
                "run_type": group, "scenario": scenario, "kind": kind,
                "n_images": x.image_id.nunique(), "n_scored": int(x.score_available.sum()),
                "n_abstained": int((~x.score_available).sum()),
                "mean_abs_shift": float(values.mean()) if len(values) else np.nan,
                "median_abs_shift": float(values.median()) if len(values) else np.nan,
                "std_abs_shift": float(values.std(ddof=0)) if len(values) else np.nan,
                "p90_abs_shift": float(values.quantile(.9)) if len(values) else np.nan,
                "max_abs_shift": float(values.max()) if len(values) else np.nan,
                "mean_signed_shift": float(x.paired_shift.mean()) if len(values) else np.nan,
                "status": "DESCRIPTIVE_ONLY" if len(values) else "ALL_ABSTAINED",
            })
    combined = pd.concat(per_case, ignore_index=True)
    combined.to_csv(out / "paired_per_image.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "scenario_sensitivity.csv", index=False)

    baseline_spatial = spatial.loc[spatial.scenario.eq("baseline")].copy()
    centres = baseline_spatial.loc[baseline_spatial.distance_m.eq(0), ["region", "score"]].rename(
        columns={"score": "centre_score"})
    comparisons = baseline_spatial.loc[baseline_spatial.distance_m.gt(0)].merge(
        centres, on="region", validate="many_to_one")
    comparisons["abs_shift_from_centre"] = (comparisons.score - comparisons.centre_score).abs()
    comparisons.to_csv(out / "spatial_directions.csv", index=False)
    by_area_distance = comparisons.groupby(["region", "distance_m"]).agg(
        n=("abs_shift_from_centre", "size"),
        mean_abs_shift=("abs_shift_from_centre", "mean"),
        std_abs_shift=("abs_shift_from_centre", lambda x: x.std(ddof=0)),
        max_abs_shift=("abs_shift_from_centre", "max")
    ).reset_index()
    by_area_distance.to_csv(out / "spatial_by_region_distance.csv", index=False)

    hist_baseline = historical.loc[historical.scenario.eq("baseline")].copy()
    temporal = hist_baseline.groupby("region").agg(
        n_years=("year", "nunique"),
        score_mean=("score", "mean"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        score_std=("score", lambda x: x.std(ddof=0))
    ).reset_index()
    temporal["range"] = temporal.score_max - temporal.score_min
    temporal.to_csv(out / "interannual_variability.csv", index=False)

    seasonal_baseline = seasonal.loc[seasonal.scenario.eq("baseline")].copy()
    seasonal_baseline["season"] = seasonal_baseline.image_id.str.extract(
        r"_(summer|winter|east)_2025$")[0]
    wide = seasonal_baseline.pivot(index="region", columns="season", values="score")
    wide["winter_minus_summer"] = wide["winter"] - wide["summer"]
    wide["abs_winter_summer_change"] = wide["winter_minus_summer"].abs()
    wide.to_csv(out / "seasonal_differences.csv")

    def make_finite(value):
        return round(float(value), 9) if pd.notna(value) else None
    meta = {
        "interpretation": "Unlabelled stress tests, not independent predictive validation.",
        "files": {p.name: _sha(p) for p in (spatial_file, spatial_manifest,
                                           historical_file, seasonal_file,
                                           seasonal_manifest, history_manifest)},
        "spatial_positions": int(baseline_spatial.image_id.nunique()),
        "spatial_perturbations": int(len(spatial)),
        "seasonal_positions": int(seasonal_baseline.image_id.nunique()),
        "historical_positions": int(hist_baseline.image_id.nunique()),
        "n_perturbation_scored": int(combined.score_available.sum()),
        "n_abstentions": int((~combined.score_available).sum()),
        "largest_spatial_changes": comparisons.nlargest(5, "abs_shift_from_centre")[
            ["image_id", "region", "distance_m", "direction", "abs_shift_from_centre"]
        ].to_dict("records"),
        "max_interannual_range": make_finite(temporal["range"].max()),
        "caveats": [
            "The 640m spatial patches may overlap and therefore are dependent.",
            "Different years and seasons use different acquisition scenes and possibly satellite platforms.",
            "The east-shifted seasonal campaign uses December scenes and cannot identify pure spatial effects.",
            "All-band-missing scenarios yield abstentions, not accuracy measurements.",
            "No independent lithium reference labels exist for these images.",
        ]
    }
    (out / "analysis_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# LithiumScope — análisis cuantitativo de sensibilidad espectral",
        "",
        "El análisis es **descriptivo y pareado por imagen**; no recalcula ni entrena modelos.",
        f"Se incluyeron {len(spatial)} escenarios espaciales, {len(historical)} históricos y {len(seasonal)} estacionales/traslaciones.",
        "",
        "## Sensibilidad a perturbaciones",
        "Ver `scenario_sensitivity.csv`: media, mediana, desviación, percentil 90, máximos y abstenciones por escenario y campaña.",
        "Ver `paired_per_image.csv` para auditar cada puntuación frente al baseline de su propia imagen.",
        "",
        "## Espacio y tiempo",
        "Ver `spatial_by_region_distance.csv` y `spatial_directions.csv` para distancia y dirección dentro de la misma escena.",
        "Ver `interannual_variability.csv` y `seasonal_differences.csv` para variación entre adquisiciones.",
        "",
        "## Limitaciones",
        "La variación de score no es un error predictivo ni demuestra generalización. La comparación con Li real queda pendiente de datos externos.",
        "No presentar estas simulaciones como ROC-AUC, AP o Balanced Accuracy externos.",
    ]
    (out / "ANALISIS_SENSIBILIDAD.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(summarize_experiments(Path("results/scientific_simulations")))
