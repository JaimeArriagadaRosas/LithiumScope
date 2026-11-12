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


def validate_inputs(root:Path):
    contract, predictors, folds, _, source_hashes = build_inputs(root)
    qcpath=root/D_QC_REL
    qsumpath=root/D_SUM_REL
    reference=root/REF_REL
    design_path=root/REF_DESIGN_REL
    paths=(qcpath,qsumpath,reference,design_path)
    if not all(x.is_file() for x in paths):
        raise FileNotFoundError("Required frozen Phase C/D artifacts missing")
    qc=pd.read_csv(qcpath)
    summary=json.loads(qsumpath.read_text(encoding="utf-8"))
    design=json.loads(design_path.read_text(encoding="utf-8"))
    if not summary["sources_hashes_sha256"]["manifest"] == source_hashes["training_manifest.csv"]:
        raise RuntimeError("Physical QA and training manifest no longer match")
    if source_hashes["training_manifest.csv"] != design["source_sha256"]["training_manifest.csv"]:
        raise RuntimeError("Baseline and current training manifest differ")
    if source_hashes["spectral_features.csv"] != design["source_sha256"]["spectral_features.csv"]:
        raise RuntimeError("Baseline and current features differ")
    if source_hashes["scientific_folds.json"] != design["source_sha256"]["scientific_folds.json"]:
        raise RuntimeError("Baseline and frozen folds differ")
    if len(qc)!=783 or qc.sample_id.astype(str).tolist()!=contract.sample_id.astype(str).tolist():
        raise ValueError("QA inventory out of alignment with training manifest")
    if qc.manifest_index.to_numpy(int).tolist()!=list(range(len(qc))):
        raise ValueError("QA input order changed")
    if qc.image_sha256.isna().any() or not qc.image_available.all():
        raise ValueError("Missing original image in the QA inventory")
    if not qc.meta_sample_id_matches.all() or not qc.meta_scene_matches.all() or not qc.meta_coords_match.all():
        raise ValueError("Original image metadata mismatch")
    if not qc.meta_bands_match.all() or not qc.image_shape_ok.all():
        raise ValueError("Original spectral schema mismatch")
    if not int((qc.finite_pixel_fraction<.95).sum())==16:
        raise RuntimeError("Unexpected number of incomplete patches")
    # Compare current image contents to the audit hashes; no silent mutation
    # to the underlying original NPYs is allowed.
    for i,(p,h) in enumerate(zip(contract.image_path,qc.image_sha256)):
        if sha(Path(p))!=str(h):
            raise ValueError(f"Patch content changed since physical-label audit: {i}")
    baseline=pd.read_csv(reference)
    if (
        len(baseline)!=HISTORICAL_N or
        baseline.test_index.sort_values().tolist()!=list(range(HISTORICAL_N)) or
        baseline.arm.ne("legacy_all").any()
    ):
        raise ValueError("Incorrect archived baseline")
    # Explicit label and outer-fold identity checks independent of predictions:
    for i,fold in enumerate(folds,1):
        _,_,te,th=define_outer_fold(fold,contract)
        ref=baseline.set_index("test_index").loc[te]
        if ref.fold.ne(i).any():
            raise ValueError("Archived baseline outer-fold assignment changed")
        if not np.allclose(ref.fold_li_threshold_ppm.to_numpy(float),th,atol=1e-8):
            raise ValueError("Archived baseline Li threshold changed")
        if not np.array_equal(ref.y_true.to_numpy(int),(
                contract.Li_icpms.iloc[te].to_numpy(float)>=th).astype(int)):
            raise ValueError("Archived baseline labels changed")
    return contract,predictors,folds,qc,baseline,source_hashes,{
        "qc":sha(qcpath),"reference":sha(reference),"phase_c_design":sha(design_path),
    }


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


def run_pilot(root:Path,out:Path, *, with_central=True) -> dict:
    contract,X,folds,qc,baseline,hashes,source_hashes=validate_inputs(root)
    out.mkdir(parents=True,exist_ok=True)
    design={
        "experiment":"Model2_phase_E_site_patch_quality_pilot_20261112",
        "arms":list(ARMS),"parameters":PARAMS,
        "reference":"Archived phase C legacy_all predictions, unchanged",
        "outer_test_indices":"706 frozen historical rows across 5 folds",
        "all_test_cases_same":True,"fold_li_label_rule":LABEL_POLICY,
        "new_label_policies":"TRAIN ONLY; hypothetical site definitions, not certified",
        "image_quality_threshold":SITE_THRESHOLD,
        "image_quality_applied_to_train_only":True,
        "central_patch_pixels":"[8:24,8:24] from original 32x32 6-band patches",
        "central_features":126,"central_transform_applied_to_all_test_rows":True,
        "no_parameter_tuning":True,
        "known_demo_cases_used":False,
        "training_source_sha256":hashes,
        "existing_evidence_sha256":source_hashes,
        "not_production_calibration":True,
    }
    (out/"experiment_design.json").write_text(
        json.dumps(design,indent=2,ensure_ascii=False),encoding="utf-8")
    center=None
    if with_central:
        t=time.monotonic()
        center=half_patch_predictors(
            contract,list(X.columns),qc,
            progress=lambda message:print(message,flush=True))
        center.to_csv(out/"central_16x16_features.csv",index=False)
        print("CENTER_COMPLETE seconds",round(time.monotonic()-t,2),flush=True)
    all_rows=[]
    per_fold=[]
    inventory=[]
    baseline_copy=baseline.copy()
    baseline_copy["arm"]=ARMS[0]
    baseline_copy["train_unit_count"]=baseline_copy["n_train"]
    baseline_copy["training_source_samples"]=baseline_copy["n_train"]
    baseline_copy["site_policy"]="UNCHANGED"
    all_rows.extend(baseline_copy.to_dict("records"))
    for arm in ARMS[1:]:
        t=time.monotonic()
        rows=[]
        for fidx,fold in enumerate(folds,1):
            _,tr,te,threshold=define_outer_fold(fold,contract)
            original_train=len(tr)
            features=center if arm=="central_16x16_patch" else X
            if features is None:
                raise ValueError("Central feature table was not generated")
            if arm in ("quality_95_train","site_median_quality_95"):
                tr=finite_quality_filter(tr,qc)
            if arm in ("site_median_li","site_majority_consensus","site_median_quality_95"):
                policy=("majority" if arm=="site_majority_consensus" else "median")
                trainX,y_train,site_inventory=aggregate_training_sites(
                    contract,features,tr,
                    lithium_threshold=threshold,policy=policy,
                    sample_sha256=qc.image_sha256)
                for rec in site_inventory.to_dict("records"):
                    rec.update({"arm":arm,"fold":fidx})
                    inventory.append(rec)
                n_training_units=len(trainX)
            else:
                trainX=features.iloc[tr]
                y_train=(contract.Li_icpms.iloc[tr].to_numpy(float)>=threshold).astype(int)
                n_training_units=len(tr)
            if np.unique(y_train).size!=2:
                raise ValueError(f"Single-class train for {arm} fold {fidx}")
            y_test=(contract.Li_icpms.iloc[te].to_numpy(float)>=threshold).astype(int)
            score=fit_classifier(trainX,y_train,features.iloc[te])
            samplemetrics=score_metrics(y_test,score)
            per_fold.append({
                "arm":arm,"fold":fidx,"n_train_samples_before":original_train,
                "n_train_samples_after_quality":len(tr),
                "n_train_site_units":n_training_units,
                "n_test":len(te),"n_test_high_Li":int(y_test.sum()),
                **samplemetrics
            })
            print("ARM",arm,"FOLD",fidx,"TRAIN",n_training_units,
                  "TEST",len(te),"AUC",round(samplemetrics["auc"],4),
                  "AP",round(samplemetrics["ap"],4),flush=True)
            for idx,yt,prob in zip(te,y_test,score):
                rows.append({
                    "arm":arm,
                    "fold":fidx,
                    "test_index":int(idx),
                    "sample_id":str(contract.sample_id.iloc[idx]),
                    "spatial_group":str(contract.spatial_group.iloc[idx]),
                    "site_screening_id":str(contract.site_screening_id.iloc[idx]),
                    "latitude":float(contract.latitude.iloc[idx]),
                    "li_ppm":float(contract.Li_icpms.iloc[idx]),
                    "fold_li_threshold_ppm":threshold,
                    "y_true":int(yt),
                    "score":float(prob),
                    "n_train":original_train,
                    "n_features":len(features.columns),
                    "train_unit_count":n_training_units,
                    "training_source_samples":len(tr),
                    "site_policy":arm,
                })
        pred=pd.DataFrame(rows).sort_values("test_index")
        if pred.test_index.tolist()!=list(range(HISTORICAL_N)):
            raise AssertionError("Each site-patch arm must retain the complete frozen 706 TEST rows")
        if not np.array_equal(pred.y_true.to_numpy(int),
                              baseline_copy.sort_values("test_index").y_true.to_numpy(int)):
            raise AssertionError("Held-out label changed in Phase E")
        all_rows.extend(rows)
        pred.to_csv(out/f"oof_{arm}.csv",index=False)
        (out/"progress.json").write_text(json.dumps({
            "completed":arm,"state":"running","elapsed_arm_s":time.monotonic()-t
        },indent=2),encoding="utf-8")
        print("ARM_COMPLETED",arm,"SECONDS",round(time.monotonic()-t,2),flush=True)
    pred=pd.DataFrame(all_rows)
    pred.to_csv(out/"all_arms_paired_oof.csv",index=False)
    pd.DataFrame(per_fold).to_csv(out/"fold_metrics.csv",index=False)
    if inventory:
        pd.DataFrame(inventory).to_csv(out/"train_site_aggregation_decisions.csv",index=False)
    (out/"identical_patch_label_error_bounds.csv").write_text(
        irreducible_identical_input_minority(contract,folds,qc).to_csv(index=False),
        encoding="utf-8")
    sample_metrics=describe_sample_subsets(pred,qc)
    sample_metrics.to_csv(out/"sample_quality_site_subgroups.csv",index=False)
    site_metrics=site_level_evaluation(pred)
    site_metrics.to_csv(out/"site_level_metrics.csv",index=False)
    summary=[]
    for arm,sub in pred.groupby("arm",sort=False):
        record={"arm":arm,**score_metrics(sub.y_true,sub.score)}
        record["n_sites_in_test"]=int(sub.site_screening_id.nunique())
        record["n_test_high"]=int(sub.y_true.sum())
        summary.append(record)
    pd.DataFrame(summary).to_csv(out/"paired_model_comparison.csv",index=False)
    # Group-bootstrapped exploratory paired effects, not a confirmation test.
    contrast=[]
    for arm in ARMS[1:]:
        result=bootstrap_group_delta(
            pred,ARMS[0],arm,repeats=320,seed=20261112)
        contrast.append(result)
    pd.DataFrame(contrast).to_csv(out/"group_bootstrap_deltas.csv",index=False)
    full={
        "state":"complete",
        "frozen_historical_test_rows":HISTORICAL_N,
        "frozen_outer_folds":5,"arms":list(ARMS),
        "baseline_reused_no_retrain":True,
        "new_fit_count":5*(len(ARMS)-1),
        "data_sources_sha256":hashes,
        "diagnostic_only":True,"model_activation":False,
        "known_demo_cases_read":False,"external_validation":False,
        "scientific_warning":"No candidate may be selected on these historical outer folds; Phase E is exploratory."
    }
    (out/"experiment_summary.json").write_text(
        json.dumps(full,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"progress.json").write_text(
        json.dumps({"state":"complete","arms":list(ARMS)},indent=2),encoding="utf-8")
    print("PHASE_E_COMPLETE",out,flush=True)
    return full


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
