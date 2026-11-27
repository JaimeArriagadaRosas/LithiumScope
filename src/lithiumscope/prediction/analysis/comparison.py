"""Pairing, correlations, concordance and training/external comparison."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def pair_model_outputs(model_1: pd.DataFrame, model_2: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in model_1.columns or "case_id" not in model_2.columns:
        return pd.DataFrame()
    left_columns = [
        column
        for column in (
            "case_id",
            "Li_icpms",
            "Li_icpms_predicted",
            "Li_icpms_interval_low_q90",
            "Li_icpms_interval_high_q90",
            "out_of_training_range_fraction",
            "applicability_warning",
            "Longitude",
            "Latitude",
            "Longitude (X)",
            "Latitude (Y)",
            "Logintude (X)",
            "longitude",
            "latitude",
        )
        if column in model_1.columns
    ]
    right_columns = [
        column
        for column in (
            "case_id",
            "prospectivity_score",
            "priority",
            "out_of_training_range_fraction",
            "applicability_warning",
            "image",
            "sentinel_scene_id",
            "sentinel_cloud_cover",
        )
        if column in model_2.columns
    ]
    right = model_2[right_columns].copy().rename(
        columns={
            "out_of_training_range_fraction": "model_2_out_of_training_range_fraction",
            "applicability_warning": "model_2_applicability_warning",
        }
    )
    left = model_1[left_columns].copy().rename(
        columns={
            "out_of_training_range_fraction": "model_1_out_of_training_range_fraction",
            "applicability_warning": "model_1_applicability_warning",
        }
    )
    return left.merge(right, on="case_id", how="inner", validate="one_to_one")


def _correlation_pair(x: pd.Series, y: pd.Series) -> tuple[int, float | None, float | None]:
    values = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return len(values), None, None
    if values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return len(values), None, None

    pearson = float(np.corrcoef(values["x"], values["y"])[0, 1])
    ranked_x = values["x"].rank(method="average")
    ranked_y = values["y"].rank(method="average")
    spearman = float(np.corrcoef(ranked_x, ranked_y)[0, 1])
    return len(values), pearson, spearman


def correlation_table(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(columns=["relationship", "n", "pearson", "spearman"])

    relationships = [
        ("Li real ↔ Li predicho M1", "Li_icpms", "Li_icpms_predicted"),
        ("Li real ↔ score M2", "Li_icpms", "prospectivity_score"),
        ("Li predicho M1 ↔ score M2", "Li_icpms_predicted", "prospectivity_score"),
    ]
    if "Li_icpms" in paired.columns and "Li_icpms_predicted" in paired.columns:
        work = paired.copy()
        work["model_1_absolute_error"] = (
            pd.to_numeric(work["Li_icpms"], errors="coerce")
            - pd.to_numeric(work["Li_icpms_predicted"], errors="coerce")
        ).abs()
        relationships.append(
            ("Error absoluto M1 ↔ score M2", "model_1_absolute_error", "prospectivity_score")
        )
    else:
        work = paired

    if (
        "model_1_out_of_training_range_fraction" in work.columns
        and "model_2_out_of_training_range_fraction" in work.columns
    ):
        relationships.append(
            (
                "OOD M1 ↔ OOD M2",
                "model_1_out_of_training_range_fraction",
                "model_2_out_of_training_range_fraction",
            )
        )

    rows: list[dict] = []
    for label, x_name, y_name in relationships:
        if x_name not in work.columns or y_name not in work.columns:
            continue
        n, pearson, spearman = _correlation_pair(work[x_name], work[y_name])
        rows.append(
            {
                "relationship": label,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows)


def concordance_table(
    paired: pd.DataFrame,
    *,
    lithium_threshold_ppm: float,
    model_2_probability_threshold: float = 0.5,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "model_1_relative_high",
                "model_2_relative_high",
                "concordance",
            ]
        )

    output = paired.copy()
    li_cut, score_cut = float(lithium_threshold_ppm), float(model_2_probability_threshold)
    if not math.isfinite(li_cut) or li_cut < 0 or not 0 <= score_cut <= 1:
        raise ValueError("Invalid concordance thresholds")
    li = pd.to_numeric(output["Li_icpms_predicted"], errors="coerce")
    scores = pd.to_numeric(output["prospectivity_score"], errors="coerce")
    valid_li = np.isfinite(li) & li.ge(0)
    valid_score = np.isfinite(scores) & scores.between(0, 1)
    output["model_1_relative_high"] = li.ge(li_cut).astype("boolean").where(valid_li)
    output["model_2_relative_high"] = scores.ge(score_cut).astype("boolean").where(valid_score)

    def label(row) -> str:
        if pd.isna(row["model_1_relative_high"]) or pd.isna(row["model_2_relative_high"]):
            return "no_comparable"
        m1 = bool(row["model_1_relative_high"])
        m2 = bool(row["model_2_relative_high"])
        if m1 and m2:
            return "concordante_alta"
        if not m1 and not m2:
            return "concordante_baja"
        if m1:
            return "divergente_m1_alto"
        return "divergente_m2_alto"

    output["concordance"] = output.apply(label, axis=1)
    return output


def training_vs_external_table(
    model_1_identity: dict,
    model_2_identity: dict,
    model_1_external: dict[str, float],
    model_2_external: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict] = []
    mappings = [
        ("model_1", "rmse", model_1_identity, model_1_external),
        ("model_1", "mae", model_1_identity, model_1_external),
        ("model_1", "r2", model_1_identity, model_1_external),
        ("model_2", "roc_auc", model_2_identity, model_2_external),
        ("model_2", "average_precision", model_2_identity, model_2_external),
        ("model_2", "balanced_accuracy", model_2_identity, model_2_external),
    ]
    for model_group, metric, identity, external in mappings:
        training = identity.get("training_metrics", {}) or {}
        if metric not in training and metric not in external:
            continue
        train_value = training.get(metric)
        external_value = external.get(metric)
        delta = None
        if train_value is not None and external_value is not None:
            delta = float(external_value) - float(train_value)
        rows.append(
            {
                "model_group": model_group,
                "metric": metric,
                "training_oof": train_value,
                "external_demo": external_value,
                "delta_external_minus_training": delta,
            }
        )
    return pd.DataFrame(rows)
