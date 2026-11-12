"""Audit FeOT sample validity and M2 training sample sufficiency; no source downloads.

All reports are research-only. No changes to canonical data, training menus,
active models, or release bundles. Repeated spatially held-out learning curves
use an outer-train-only threshold that is constant across training fractions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.experiments.analysis.run_vnext_m1 import _frame
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target
from lithiumscope.core.config import load_config
from lithiumscope.model_1.steps.step_09_validation import build_spatial_groups
from lithiumscope.model_2.training.scientific_validation import make_scientific_folds

OUT = RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage4_evidence"


def _json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def sample_keys(frame: pd.DataFrame) -> pd.Series:
    """Explicit-provenance sample identifiers with rounded coordinates."""
    ids = frame.get(
        "source_sample", pd.Series(np.nan, index=frame.index)
    ).astype("string").fillna("").str.strip()
    if "Sample" in frame:
        ids = ids.mask(ids.eq(""), frame["Sample"].astype("string").fillna("").str.strip())
    ids = ids.str.casefold()
    lon = pd.to_numeric(frame.Longitude, errors="coerce").round(5)
    lat = pd.to_numeric(frame.Latitude, errors="coerce").round(5)
    sources = frame.source_dataset.astype("string").fillna("")
    good = ids.ne("") & lat.notna() & lon.notna()
    keys = (
        sources + "::" + ids + "::" + lon.astype(str) + "::" + lat.astype(str)
    ).where(good)
    return keys


def audit_m1(root: Path = PROJECT_ROOT, out: Path = OUT) -> dict:
    path_a = root / "data/processed/model_1/training_combined.csv"
    path_b = (root / "results/scientific_simulations/vnext_20261106"
              / "stage2_m1/feot_combined_experiment_only.csv")
    path_raw = root / "data/raw/model_1/georoc/GEOROC_Andean_Arc_LithiumScope.csv"
    a_all = _frame(pd.read_csv(path_a, low_memory=False))
    a_all, _ = filter_target(
        a_all, list(load_config("model_1")["data"]["target_candidates"])
    )
    b_all = _frame(pd.read_csv(path_b, low_memory=False))
    b_all = b_all.dropna(subset=["Li_icpms"])
    a = a_all.loc[protocol_masks(a_all)["A_original"]].copy()
    flags = protocol_masks(b_all)
    b = b_all.loc[flags["B_reconstructed"]].copy()
    c = b_all.loc[flags["C_experimental"]].copy()
    assert len(a) == 1120, "Reference A differs from previously validated dataset"
    assert len(b) == 3420, "Reconstructed B differs from validated experiment"
    id_a, id_b = sample_keys(a), sample_keys(b)
    a_keys, b_keys = set(id_a.dropna()), set(id_b.dropna())
    missing_from_b = a_keys - b_keys
    added_to_b = b_keys - a_keys

    georoc = b.loc[b.source_dataset.astype(str).eq("GEOROC")].copy()
    provenance = georoc.get(
        "Fe_total_provenance", pd.Series("", index=georoc.index)
    ).astype(str)
    geo_groups = build_spatial_groups(georoc)
    a_groups = build_spatial_groups(a)
    b_groups = build_spatial_groups(b)

    raw = pd.read_csv(path_raw, low_memory=False)
    iron_feot = pd.to_numeric(raw["FEOT(WT%)"], errors="coerce")
    iron_ferric = pd.to_numeric(raw["FE2O3(WT%)"], errors="coerce")
    li = pd.to_numeric(raw["LI(PPM)"], errors="coerce")
    double_iron = iron_feot.notna() & iron_ferric.notna() & li.notna()
    ratio = (iron_ferric / (iron_feot * 1.111348))[double_iron]
    valid_raw_iron = iron_feot.loc[
        iron_feot.notna() & li.notna()
    ]
    citations = raw["CITATIONS"].astype("string").fillna("").str.strip()
    sources = (
        b.groupby("source_dataset", dropna=False).size().to_dict()
    )
    total_b = pd.to_numeric(georoc["SUM (no water)"], errors="coerce")
    georoc_qc_pass = total_b.between(94, 102)
    if not georoc_qc_pass.all():
        raise AssertionError("Invalid GEOROC oxide closure in protocol B")
    if georoc[["Longitude", "Latitude", "Li_icpms"]].isna().any().any():
        raise AssertionError("Missing essential GEOROC entries in protocol B")
    if "Fe_total_provenance" not in georoc.columns:
        raise AssertionError("Missing total-iron provenance")

    # Detect same explicit sample id occurring in multiple spatial blocks.
    ids = georoc.source_sample.astype("string").fillna("").str.strip().str.casefold()
    byid = pd.DataFrame({"sample": ids, "group": geo_groups})
    leak = byid.loc[byid["sample"].ne("")].groupby("sample").group.nunique()
    leakage_candidates = leak[leak > 1]
    summary = {
        "experiment_only": True,
        "original_A_count": len(a),
        "reconstructed_B_count": len(b),
        "provisional_C_count": len(c),
        "B_minus_A": len(b) - len(a),
        "B_gain_percent": 100.0 * (len(b) / len(a) - 1),
        "source_breakdown_B": {str(k): int(v) for k, v in sources.items()},
        "explicit_identifiers_A": int(id_a.notna().sum()),
        "explicit_identifiers_B": int(id_b.notna().sum()),
        "A_unique_identifiers": len(a_keys),
        "B_unique_identifiers": len(b_keys),
        "A_explicit_keys_absent_from_B": len(missing_from_b),
        "new_explicit_keys_in_B": len(added_to_b),
        "georoc_iron_provenance": provenance.value_counts().to_dict(),
        "georoc_oxide_closure_quantiles": {
            str(k): float(v) for k, v in total_b.quantile([0,.01,.1,.5,.9,.99,1]).items()
        },
        "georoc_oxide_sum_range_verified": True,
        "raw_FeOT_valid_lithium": int(valid_raw_iron.notna().sum()),
        "raw_FeOT_nonphysical_below_zero": int((valid_raw_iron < 0).sum()),
        "raw_FeOT_above_100": int((valid_raw_iron > 100).sum()),
        "raw_both_FeOT_and_Fe2O3_with_Li": int(double_iron.sum()),
        "raw_Fe2O3_vs_FeOT_equivalent_ratio_quantiles": {
            str(k): float(v) for k, v in ratio.quantile([0,.1,.25,.5,.75,.9,1]).items()
        },
        "raw_Li_rows_with_citation": int((li.notna() & citations.ne("")).sum()),
        "raw_Li_total": int(li.notna().sum()),
        "georoc_number_of_groups_B": int(geo_groups.nunique()),
        "all_groups_A": int(a_groups.nunique()),
        "all_groups_B": int(b_groups.nunique()),
        "B_group_top5": {str(k): int(v) for k,v in b_groups.value_counts().head().items()},
        "distinct_sample_ids_in_multiple_0p5deg_groups": int(len(leakage_candidates)),
        "ambiguous_sample_ids_examples": leakage_candidates.head(8).index.tolist(),
        "c_limit": "C contains records without complete analytical closure; do not promote",
        "scientific_limit": "Total iron as FeOT is converted to Fe2O3-equivalent; this does not certify original lab method, oxidation state, or independent provenance",
        "raw_sha256": hashlib.sha256(path_raw.read_bytes()).hexdigest(),
    }
    if missing_from_b:
        (out/"m1_original_ids_absent_in_B.txt").write_text(
            "\n".join(sorted(missing_from_b)[:100])+"\n",encoding="utf-8"
        )
    _json(out/"m1_chemical_provenance_validation.json",summary)
    print("M1_AUDIT",json.dumps(summary,ensure_ascii=False)[:4800],flush=True)
    return summary


def audit_m2(root: Path = PROJECT_ROOT, out: Path = OUT) -> dict:
    manifest_path = root / "data/processed/model_2/training_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    groups = manifest.spatial_group.astype("string")
    counts = groups.value_counts()
    samples = len(manifest)
    portions = counts / samples
    global_q75 = float(manifest.Li_icpms.quantile(.75))
    labels = manifest.Li_icpms >= global_q75
    bygroup = manifest.assign(high_lithium_descriptive=labels).groupby(
        "spatial_group"
    ).agg(
        samples=("Li_icpms","size"),
        positive_samples=("high_lithium_descriptive","sum"),
        li_median=("Li_icpms","median"),
        distinct_scenes=("sentinel_scene_id","nunique"),
        cloud_cover_mean=("sentinel_cloud_cover","mean"),
        latitude_min=("latitude","min"),
        latitude_max=("latitude","max"),
        longitude_min=("longitude","min"),
        longitude_max=("longitude","max"),
    ).reset_index()
    bygroup["high_li_fraction_descriptive"] = (
        bygroup.positive_samples / bygroup.samples
    )
    bygroup.sort_values("samples",ascending=False).to_csv(
        out/"m2_group_coverage.csv",index=False
    )
    folds = make_scientific_folds(
        manifest.Li_icpms,groups,n_splits=5,quantile=.75
    )
    foldrows=[]
    for index,f in enumerate(folds,start=1):
        test_groups=groups.iloc[f.test_idx]
        foldrows.append({
            "fold":index, "n_train":len(f.train_idx), "n_test":len(f.test_idx),
            "train_groups":f.n_groups_train,"test_groups":f.n_groups_test,
            "threshold_train_q75":f.label_threshold_ppm,
            "train_positive_fraction":float(f.y_train.mean()),
            "test_positive_fraction":float(f.y_test.mean()),
            "test_largest_group_share":float(
                test_groups.value_counts().max()/len(f.test_idx)
            ),
        })
    pd.DataFrame(foldrows).to_csv(out/"m2_spatial_fold_imbalance.csv",index=False)
    summary = {
        "total_samples":samples,
        "groups":len(counts),
        "groups_singleton":int((counts==1).sum()),
        "groups_below_five":int((counts<5).sum()),
        "group_top1":int(counts.iloc[0]),
        "group_top1_fraction":float(portions.iloc[0]),
        "group_top5_fraction":float(portions.iloc[:5].sum()),
        "effective_group_count_inverse_hhi":float(1.0/np.sum(portions**2)),
        "distinct_sample_ids":int(manifest.sample_id.nunique()),
        "distinct_image_paths":int(manifest.image_path.nunique()),
        "distinct_sentinel_scenes":int(manifest.sentinel_scene_id.nunique()),
        "cloud_cover_quantiles": {
            str(k): float(v) for k,v in manifest.sentinel_cloud_cover.quantile([.1,.5,.9]).items()
        },
        "global_q75_li_for_descriptive_group_audit_only":global_q75,
        "groups_with_zero_descriptive_positive":int((bygroup.positive_samples==0).sum()),
        "groups_with_all_descriptive_positive":int((bygroup.positive_samples==bygroup.samples).sum()),
        "folds":foldrows,
        "sample_geography_bbox": {
            "latitude":[float(manifest.latitude.min()),float(manifest.latitude.max())],
            "longitude":[float(manifest.longitude.min()),float(manifest.longitude.max())],
        },
        "data_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "warning":"Global q75 used only in descriptive audit. Train-fold q75 used in any predictive evaluation. Group effective count is concentration index, not formal statistical effective sample size.",
    }
    _json(out/"m2_sample_coverage_validation.json",summary)
    print("M2_AUDIT",json.dumps(summary,ensure_ascii=False)[:6500],flush=True)
    return summary


def group_train_indices(
    train_indices, groups, fraction: float, seed: int,
) -> np.ndarray:
    """Subsample whole outer-train spatial groups; no evaluation group leakage."""
    train_idx = np.asarray(train_indices,dtype=int)
    if not 0 < fraction <= 1 or len(train_idx) == 0:
        raise ValueError("Invalid train selection")
    allgroups = pd.Series(groups).iloc[train_idx].astype("string")
    unique = np.asarray(sorted(allgroups.unique()),dtype=object)
    count = max(3, min(len(unique),int(round(fraction*len(unique)))))
    if fraction == 1.0:
        return train_idx
    rng=np.random.default_rng(seed)
    chosen = set(rng.permutation(unique)[:count])
    mask = allgroups.isin(chosen).to_numpy()
    selected=train_idx[mask]
    assert len(selected)>0
    return selected


from lithiumscope.experiments.analysis.verify_vnext_data_adequacy_run_m2_learning_curve import run_m2_learning_curve


if __name__=="__main__":
    import sys
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage in ("audit","all"):
        audit_m1()
        audit_m2()
    if stage in ("curve","all"):
        curve=run_m2_learning_curve()
        print("M2_LEARNING_CURVE_DONE",json.dumps(curve["mean_by_fraction"]),flush=True)
