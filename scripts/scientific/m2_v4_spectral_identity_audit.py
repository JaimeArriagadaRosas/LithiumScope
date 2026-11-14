"""Exact spectral signature duplicates, mixed labels and physical-ratio flags.

Read-only: no aggregation, editing labels or filtering TEST in this audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT


def analyze(cohort: int, *, root: Path = PROJECT_ROOT) -> dict:
    if cohort not in (706, 783):
        raise ValueError("Only frozen M2 cohorts 706/783")
    folder = root / "results/scientific_simulations/m2_rededge_v4_20261113" / f"cohort_{cohort}"
    training = folder / "training_v4_170.csv"
    if not training.is_file():
        raise FileNotFoundError(training)
    frame = pd.read_csv(training)
    manifest = pd.read_csv(
        root
        / (
            "data/processed/model_2/training_manifest.csv"
            if cohort == 706
            else "data/processed/model_2/vnext_expanded/training_manifest.csv"
        )
    )
    quality = pd.read_csv(folder / "rededge_quality.csv")
    if (
        len(frame) != cohort
        or len(manifest) != cohort
        or not np.array_equal(frame.Li_icpms, manifest.Li_icpms)
        or not np.array_equal(frame.spatial_group, manifest.spatial_group)
        or not np.array_equal(manifest.sample_id.astype(str), quality.sample_id.astype(str))
    ):
        raise ValueError("Spectral identity audit source alignment failed")
    features = frame.drop(columns=["Li_icpms", "spatial_group"])
    if len(features.columns) != 170:
        raise ValueError("M2 v4 must have 170 spectral attributes")
    sig126 = pd.util.hash_pandas_object(features.iloc[:, :126], index=False)
    sig170 = pd.util.hash_pandas_object(features, index=False)
    table = pd.DataFrame(
        {
            "sample_id": manifest.sample_id.astype(str),
            "row_index": np.arange(cohort, dtype=int),
            "spatial_group": manifest.spatial_group.astype(str),
            "Li_icpms": manifest.Li_icpms.to_numpy(float),
            "label_fixed_21p6": (manifest.Li_icpms >= 21.6).astype(int),
            "signature_126": sig126.astype(str),
            "signature_170": sig170.astype(str),
            "rededge_QA95": (quality.rededge_joint_valid_fraction >= 0.95),
            "ndre1_p90_gt1": (features.ndre1_refl_p90 > 1),
            "ndre1_p10_ltminus1": (features.ndre1_refl_p10 < -1),
        }
    )
    groups = table.groupby("signature_170", sort=False).agg(
        n=("sample_id", "size"),
        n_spatial_groups=("spatial_group", "nunique"),
        n_Li_unique=("Li_icpms", "nunique"),
        min_Li=("Li_icpms", "min"),
        max_Li=("Li_icpms", "max"),
        n_positive=("label_fixed_21p6", "sum"),
        group=("spatial_group", "first"),
    )
    groups["n_negative"] = groups.n - groups.n_positive
    groups["mixed_binary"] = (groups.n_positive > 0) & (groups.n_negative > 0)
    groups["minimum_unavoidable_binary_errors"] = groups[["n_positive", "n_negative"]].min(axis=1)
    groups["sample_ids"] = table.groupby("signature_170").sample_id.apply(lambda ids: ";".join(ids))
    table["repeated_spectral_signature"] = table.signature_170.map(groups.n).gt(1)
    table["mixed_binary_label_same_spectrum"] = table.signature_170.map(groups.mixed_binary)
    identical_groups = int((groups.n_spatial_groups > 1).sum())
    if identical_groups:
        raise ValueError("Spectrally identical rows span spatial TRAIN/TEST groups")
    mixed = groups[groups.mixed_binary]
    repeated = groups[groups.n > 1]
    summary = {
        "cohort": cohort,
        "full_feature_count": 170,
        "distinct_exact_signatures_126": int(sig126.nunique()),
        "distinct_exact_signatures_170": int(sig170.nunique()),
        "repeated_signature_count": len(repeated),
        "rows_with_repeated_spectrum": int(repeated.n.sum()),
        "excess_redundant_rows": int((repeated.n - 1).sum()),
        "mixed_fixed_class_signatures": len(mixed),
        "mixed_fixed_class_rows": int(mixed.n.sum()),
        "minimum_classification_errors_due_only_to_identical_X": int(mixed.minimum_unavoidable_binary_errors.sum()),
        "identical_signatures_across_spatial_groups": identical_groups,
        "rededge_ndre1_p90_above_one": int(table.ndre1_p90_gt1.sum()),
        "rededge_ndre1_p10_below_minus_one": int(table.ndre1_p10_ltminus1.sum()),
        "rededge_band1_negative_sample_means": int((features.rededge1_refl_mean < 0).sum()),
        "no_label_changes_or_data_filtering": True,
        "not_evidence_of_incorrect_lab_measurements": True,
        "physical_ratio_flags_are_diagnostics_not_automatic_rejections": True,
    }
    groups.reset_index().to_csv(folder / "spectral_signature_groups.csv", index=False)
    table.to_csv(folder / "spectral_signature_per_sample.csv", index=False)
    (folder / "spectral_identity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("M2_V4_SPECTRAL_IDENTITY", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    analyze(706)
    analyze(783)
