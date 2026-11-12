"""Phase E — bounded opt-in site/parch representation pilot for LithiumScope M2.

All outer test labels and the 706 historical test cases are frozen. No use
of the ten public demonstration cases, GEOROC field assumptions, model
promotion, training-menu changes, downloads or calibration. CatBoost fixed
parameters and phase-C archived baseline, no re-optimization.

All site-policy decisions are *TRAIN only*; outer TEST keeps sample labels,
site membership, original coordinates and all low-quality samples. Alternative
site label policies are hypothetical, NOT a relabeling of original data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.neighbors import BallTree

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.steps.step_05_spectral_features import (
    DEFAULT_BAND_NAMES, extract_spectral_features,
)
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.methodology.ablations import (
    HISTORICAL_N, PARAMS, build_inputs, define_outer_fold, fit_classifier,
    score_metrics, bootstrap_group_delta, sha
)

# Pre-registered before any run is examined:
ARMS = (
    "historical_706_reference",  # frozen Phase-C OOF; same CatBoost hyperparams
    "site_median_li",            # one site => median Li class (train only)
    "site_majority_consensus",   # one site => strict majority, abstain ties (train only)
    "quality_95_train",          # exclude finite < 95% (train only)
    "site_median_quality_95",    # combined stress test, not single-factor
    "central_16x16_patch",       # 126 features over fixed central 16x16 pixels
)
D_QC_REL = (
    "results/scientific_simulations/m2_phase_d_20261112/"
    "physical_label_audit/patch_image_integrity_783.csv"
)
D_SUM_REL = (
    "results/scientific_simulations/m2_phase_d_20261112/"
    "physical_label_audit/scientific_quality_summary.json"
)
REF_REL = (
    "results/scientific_simulations/m2_phase_c_20261112/"
    "pilot_8arms/oof_legacy_all.csv"
)
REF_DESIGN_REL = (
    "results/scientific_simulations/m2_phase_c_20261112/"
    "pilot_8arms/experiment_design.json"
)
SITE_THRESHOLD = 0.95
HALF_WINDOW = 16
LABEL_POLICY = "outer_train_historical_Li_q75"
PRIMARY_TEST = "all_706_unchanged"
PRIMARY_METRICS = ("auc", "ap", "brier")


def historical_train_groups(contract: pd.DataFrame, training_indices: np.ndarray):
    """Map each site to its ordered original TRAIN indices, without consulting test."""
    rows = pd.DataFrame({
        "index": list(np.asarray(training_indices, dtype=int)),
        "site": contract.site_screening_id.iloc[training_indices].astype(str).to_numpy(),
    })
    return {
        str(site): group["index"].to_numpy(dtype=int)
        for site, group in rows.groupby("site", sort=True)
    }


def aggregate_training_sites(
    contract: pd.DataFrame, features: pd.DataFrame, training_indices: np.ndarray,
    *, lithium_threshold: float, policy: str,
    sample_sha256: pd.Series,
) -> tuple[pd.DataFrame,np.ndarray,pd.DataFrame]:
    """Produce exactly one spectral representation and TRAIN label per site.

    Never includes held-out indices. Identical-site images must have identical
    file content; if this invariant breaks, force reinspection, no silent
    feature averaging between distinct patches. Median Li or strict label
    consensus are hypothetical target definitions, not scientific truth.
    """
    if policy not in {"median", "majority"}:
        raise ValueError("Unknown site aggregation policy")
    if not 0 <= lithium_threshold <= 1_000_000:
        raise ValueError("Invalid Li cut")
    groups = historical_train_groups(contract,training_indices)
    xrows=[]
    yrows=[]
    inventory=[]
    for site, indices in groups.items():
        images=sample_sha256.iloc[indices].astype(str)
        if len(set(images)) != 1:
            raise ValueError(f"Site contains different spectral patches: {site}")
        li=contract.Li_icpms.iloc[indices].to_numpy(dtype=float)
        pos=(li >= lithium_threshold).astype(int)
        if policy=="majority" and pos.sum()*2 == len(pos):
            inventory.append({
                "site":site,"original_train_samples":len(indices),
                "n_pos":int(pos.sum()),"n_neg":int(len(pos)-pos.sum()),
                "chosen_class":None,"included":False,
                "decision":"EXCLUDE_TIED_SITE_TRAIN_ONLY",
            })
            continue
        y=int(np.median(li)>=lithium_threshold) if policy=="median" else int(pos.sum()*2>len(pos))
        representative=int(indices[0])
        # Same spectral patch for all original entries, so choose one.
        xrows.append(features.iloc[representative].to_numpy(float))
        yrows.append(y)
        inventory.append({
            "site":site,"original_train_samples":len(indices),
            "n_pos":int(pos.sum()),"n_neg":int(len(pos)-pos.sum()),
            "chosen_class":int(y),"included":True,
            "decision":"MEDIAN_LI_SITE" if policy=="median" else "STRICT_MAJORITY_SITE",
        })
    if not xrows:
        raise ValueError("No site observations available after aggregation")
    return (
        pd.DataFrame(xrows,columns=features.columns),
        np.asarray(yrows,dtype=int),
        pd.DataFrame(inventory),
    )


def finite_quality_filter(
    training_indices: np.ndarray, qc: pd.DataFrame,
    *, minimum_fraction: float=SITE_THRESHOLD
) -> np.ndarray:
    """Predefined image input integrity filter: TRAIN only, no target labels."""
    if not (0 < minimum_fraction <= 1):
        raise ValueError("Invalid image quality threshold")
    q=qc.finite_pixel_fraction.iloc[training_indices].to_numpy(float)
    if not np.isfinite(q).all():
        raise ValueError("Missing quality fractions cannot be silently accepted")
    return np.asarray(training_indices,dtype=int)[q>=minimum_fraction]


def half_patch_predictors(
    contract:pd.DataFrame, expected_columns:list[str], quality:pd.DataFrame,
    *, progress=None
) -> pd.DataFrame:
    """Extract frozen 126 spectral descriptors from central 16x16 of all 783.

    The model is trained and evaluated on the same new representation. All
    pixels are from existing six-band NPYs. No label/scene/date selection.
    """
    if len(contract)!=len(quality) or len(expected_columns)!=126:
        raise ValueError("Data or schema mismatch for center patch")
    cache={}
    rows=[]
    for i,(src, digest) in enumerate(zip(
            contract.image_path.astype(str),quality.image_sha256.astype(str))):
        if digest not in cache:
            image=np.load(Path(src),allow_pickle=False)
            if image.shape != (6,32,32):
                raise ValueError(f"Unexpected shape at sample {i}: {image.shape}")
            center=image[:,8:24,8:24]
            features=extract_spectral_features(
                preprocess_image(center,normalize_per_band=False),
                band_names=DEFAULT_BAND_NAMES)
            if list(features)!=expected_columns:
                raise ValueError("Center spectral feature schema differs from frozen full patch")
            arr=np.asarray(list(features.values()),dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError("Nonfinite center features")
            cache[digest]=arr
        rows.append(cache[digest])
        if progress is not None and (i%130==0 or i==len(contract)-1):
            progress(f"CENTER_FEATURES {i+1}/{len(contract)} unique={len(cache)}")
    return pd.DataFrame(rows,columns=expected_columns)


from lithiumscope.model_2.methodology.site_patch_lab_validate_inputs import validate_inputs


def site_level_evaluation(arms:pd.DataFrame) -> pd.DataFrame:
    """Same frozen site labels for every model, alternative metric not tuned."""
    rows=[]
    for arm,sub in arms.groupby("arm",sort=False):
        sites=sub.groupby("site_screening_id",sort=True).agg(
            n_samples=("sample_id","size"),label_classes=("y_true","nunique"),
            fold_count=("fold","nunique"),score_median=("score","median"),
            li_median=("li_ppm","median"),test_fold_threshold=("fold_li_threshold_ppm","first"),
        )
        if sites.fold_count.max()!=1:
            raise ValueError("One site appears in multiple outer test folds")
        y=(sites.li_median>=sites.test_fold_threshold).to_numpy(dtype=int)
        sc=sites.score_median.to_numpy(float)
        for subset_name,idx in (
            ("all_sites",np.ones(len(sites),bool)),
            ("consistent_sites",sites.label_classes.to_numpy(int)==1),
            ("ambiguous_sites",sites.label_classes.to_numpy(int)>1),
        ):
            yy=y[idx]
            score=sc[idx]
            metrics=score_metrics(yy,score) if len(yy) and len(set(yy))==2 else {"auc":None,"ap":None,"brier":float(brier_score_loss(yy,score)) if len(yy) else None}
            rows.append({"arm":arm,"subset":subset_name,"n_sites":int(sum(idx)),
                         "positive_sites":int(sum(yy)),"ambiguous_sites_in_subset":int(sum(
                             sites.label_classes.to_numpy(int)[idx]>1)),
                         "roc_auc":metrics["auc"],"ap":metrics["ap"],"brier":metrics["brier"]})
    return pd.DataFrame(rows)


def describe_sample_subsets(arms:pd.DataFrame, qc:pd.DataFrame) -> pd.DataFrame:
    """Stratify fixed test samples without ever redefining the training."""
    reference=arms.loc[arms.arm==ARMS[0]].copy()
    by_index=reference.test_index.to_numpy(int)
    nsite=reference.site_screening_id.map(reference.site_screening_id.value_counts()).to_numpy(int)
    bad=qc.finite_pixel_fraction.iloc[by_index].to_numpy(float)<SITE_THRESHOLD
    counts=pd.DataFrame({"test_index":by_index,
                         "low_quality":bad,
                         "repeated_site":nsite>1})
    augmented=arms.merge(counts,on="test_index",validate="many_to_one")
    rows=[]
    for name,sub in augmented.groupby("arm",sort=False):
        for subset,mask in (
            ("all_706",np.ones(len(sub),bool)),
            ("test_quality_below_95pct",sub.low_quality.to_numpy(bool)),
            ("test_quality_at_least_95pct",~sub.low_quality.to_numpy(bool)),
            ("test_repeated_site",sub.repeated_site.to_numpy(bool)),
            ("test_singleton_site",~sub.repeated_site.to_numpy(bool)),
        ):
            subset_df=sub.loc[mask]
            mm=score_metrics(subset_df.y_true,subset_df.score) if len(subset_df) else {}
            rows.append({"arm":name,"subset":subset,"n":int(len(subset_df)),
                         "positive":int(subset_df.y_true.sum()),**mm})
    return pd.DataFrame(rows)


def irreducible_identical_input_minority(
    contract:pd.DataFrame, folds:list[dict], qc:pd.DataFrame
)->pd.DataFrame:
    """Training diagnostics per fold. Not a population error bound."""
    rows=[]
    for n,fold in enumerate(folds,1):
        _,tr,te,threshold=define_outer_fold(fold,contract)
        for partition,indices in (("train",tr),("outer_test",te)):
            groups=historical_train_groups(contract,indices)
            min_errors=0
            mixed_sites=0
            mixed_samples=0
            for site,idx in groups.items():
                labels=(contract.Li_icpms.iloc[idx].to_numpy(float)>=threshold)
                hashes=qc.image_sha256.iloc[idx].astype(str)
                if len(set(hashes))!=1:
                    raise ValueError("Same site mapped to multiple satellite inputs")
                if labels.any() and (~labels).any():
                    mixed_sites+=1
                    mixed_samples+=len(idx)
                    min_errors+=int(min(sum(labels),sum(~labels)))
            rows.append({"fold":n,"partition":partition,
                         "threshold":float(threshold),
                         "n_samples":len(indices),"n_sites":len(groups),
                         "mixed_sites":mixed_sites,"mixed_site_samples":mixed_samples,
                         "minimum_unavoidable_binary_errors":min_errors,
                         "interpretation":"If model receives identical predictor vectors at same site; only same-fold labels."})
    return pd.DataFrame(rows)


from lithiumscope.model_2.methodology.site_patch_lab_run_pilot import run_pilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    opts=parser.parse_args()
    dest=Path(opts.output)
    if not dest.is_absolute():
        dest=PROJECT_ROOT/dest
    run_pilot(PROJECT_ROOT,dest)


if __name__=="__main__":
    main()
