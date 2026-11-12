"""M2 experimental grouped CV, regularization, image augmentation, calibration.

Runs only against existing cached Sentinel-2 images and training data. Does not
replace active or candidate release bundles. No random fold fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features
from lithiumscope.model_2.training.scientific_validation import make_scientific_folds


def augment_image_training_only(
    cube: np.ndarray, *, seed: int = 42, brightness: float = 1.04,
    noise_fraction: float = 0.02, missing_fraction: float = 0.03,
) -> np.ndarray:
    """Small, bounded pixel-domain perturbation, not a substitute for SCL masks."""
    if not 0.0 < brightness <= 1.25:
        raise ValueError("Unexpected brightness factor")
    if not 0 <= noise_fraction <= 0.1 or not 0 <= missing_fraction <= 0.1:
        raise ValueError("Invalid augmentation intensity")
    original = np.asarray(cube, dtype=np.float32)
    if original.ndim != 3 or original.shape[0] != 6:
        raise ValueError("Expected 6-band Sentinel-2 patch")
    rng = np.random.default_rng(seed)
    value = original.copy()
    std = np.nanstd(value, axis=(1, 2), keepdims=True)
    scale = np.maximum(std, 0.005) * noise_fraction
    value = (value * brightness + rng.normal(
        0, 1, size=value.shape
    ).astype(np.float32) * scale).astype(np.float32)
    value = np.clip(value, 0, 1.6)
    value[rng.random(value.shape) < missing_fraction] = np.nan
    return value


def expected_calibration_error(
    labels, probabilities, *, bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Invalid labels / probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    edges = np.linspace(0, 1, bins+1)
    result = 0.0
    for idx in range(bins):
        mask = ((probabilities >= edges[idx])
                & (probabilities < edges[idx+1]
                   if idx < bins - 1 else probabilities <= edges[idx+1]))
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(result)


def score_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("Heldout fold must contain both classes")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p>=0.5)),
        "f1": float(f1_score(y, p>=0.5, zero_division=0)),
        "precision": float(precision_score(y, p>=0.5, zero_division=0)),
        "recall": float(recall_score(y, p>=0.5, zero_division=0)),
        "brier": float(brier_score_loss(y,p)),
        "ece_10": expected_calibration_error(y,p),
    }


def _calibration_splits(x, y, groups, n_splits: int = 3):
    if groups.nunique() < n_splits:
        return None
    splits = list(GroupKFold(n_splits=n_splits).split(x, groups=groups))
    for tr, va in splits:
        if len(np.unique(np.asarray(y)[tr])) < 2 or len(np.unique(np.asarray(y)[va])) < 2:
            return None
        if set(groups.iloc[tr]).intersection(set(groups.iloc[va])):
            raise RuntimeError("Group leakage during probability calibration")
    return splits


def make_augmentation_table(
    manifest: pd.DataFrame,
    bundle: dict,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    features = list(bundle["features"])
    names = tuple(bundle["band_names"])
    rows = []
    for idx, item in enumerate(manifest.itertuples(index=False)):
        image_path = Path(item.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = load_imagery(image_path)
        if image.shape[0] != len(names):
            raise ValueError("Sentinel band mismatch: "+str(image_path))
        disturbed = augment_image_training_only(image, seed=seed+idx)
        clean = preprocess_image(
            disturbed, normalize_per_band=bool(bundle.get("normalize_per_band",False))
        )
        result = extract_spectral_features(clean, band_names=names)
        rows.append(result)
        if (idx+1)%100==0:
            print("M2_AUG_FEATURES",idx+1,"/",len(manifest),flush=True)
    table=pd.DataFrame(rows).reindex(columns=features)
    if table.isna().any().any() or not np.isfinite(table.to_numpy(dtype=float)).all():
        raise RuntimeError("Augmented training features are not all finite.")
    return table


def run_experiments(
    root: Path = PROJECT_ROOT,
    destination: Path | None = None,
) -> dict:
    out = destination or (
        RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage3_m2"
    )
    out.mkdir(parents=True, exist_ok=True)
    datafile=root/"data/processed/model_2/sentinel_features.csv"
    manifestfile=root/"data/processed/model_2/training_manifest.csv"
    frame=pd.read_csv(datafile)
    manifest=pd.read_csv(manifestfile)
    if len(frame)!=len(manifest) or not np.allclose(
        frame.Li_icpms.to_numpy(dtype=float), manifest.Li_icpms.to_numpy(dtype=float)
    ) or not frame.spatial_group.astype(str).equals(manifest.spatial_group.astype(str)):
        raise RuntimeError("Spectral feature cache misaligned with image manifest.")
    new=joblib.load(
        root/"models/model_2/trained/training_20261105_223618_m0300/model.joblib"
    )
    old=joblib.load(
        root/"models/model_2/trained/training_20260920_195933_m0300/model.joblib"
    )
    columns=list(new["features"])
    if columns!=list(old["features"]):
        raise RuntimeError("Release and new candidate require different features.")
    x=frame[columns].copy()
    groups=frame.spatial_group.reset_index(drop=True)
    lithium=frame.Li_icpms.reset_index(drop=True)
    folds=make_scientific_folds(lithium, groups, n_splits=5, quantile=0.75)
    augmented=make_augmentation_table(manifest,new)
    augmented.to_csv(out/"training_image_augmented_features_only.csv",index=False)
    print("M2_GROUP_AUGMENTATION_READY",augmented.shape,flush=True)
    regular=clone(new["estimator"])
    regular.set_params(n_estimators=220,min_samples_leaf=4,max_depth=9,n_jobs=4)
    calibrated=clone(regular)
    calibrated.set_params(n_estimators=160)
    legacy=clone(old["estimator"])
    old_params=legacy.get_params()
    replace={}
    if "iterations" in old_params:
        replace["iterations"]=min(250,int(old_params["iterations"]))
    if "verbose" in old_params:
        replace["verbose"]=False
    if "allow_writing_files" in old_params:
        replace["allow_writing_files"]=False
    if "thread_count" in old_params:
        replace["thread_count"]=4
    if replace:
        legacy.set_params(**replace)
    specs=[
        ("rf_existing",new["estimator"],False,False),
        ("rf_regularized",regular,False,False),
        ("rf_pixel_augmented",regular,True,False),
        ("rf_group_calibrated",calibrated,False,True),
        ("catboost_existing",legacy,False,False),
    ]
    records=[]
    predictions=[]
    for name,base,augment,calibrate in specs:
        for index,fold in enumerate(folds,start=1):
            tr=fold.train_idx
            te=fold.test_idx
            xtrain=x.iloc[tr].reset_index(drop=True)
            ytrain=np.asarray(fold.y_train,dtype=int)
            labels=np.asarray(fold.y_test,dtype=int)
            if augment:
                xtrain=pd.concat(
                    [xtrain,augmented.iloc[tr].reset_index(drop=True)],
                    ignore_index=True,
                )
                ytrain=np.concatenate([ytrain,ytrain])
            estimator=clone(base)
            if calibrate:
                split=_calibration_splits(
                    x.iloc[tr].reset_index(drop=True),
                    fold.y_train,
                    groups.iloc[tr].reset_index(drop=True),
                )
                if split is None:
                    records.append({
                        "variant":name,"fold":index,"status":"SKIPPED_NO_SPATIAL_CALIBRATION",
                        "n_test":len(te),
                    })
                    print("M2_FOLD_SKIPPED_CALIBRATION",index,flush=True)
                    continue
                estimator=CalibratedClassifierCV(
                    estimator=estimator,method="sigmoid",cv=split,
                )
            estimator.fit(xtrain,ytrain)
            proba=np.asarray(estimator.predict_proba(x.iloc[te])[:,1],dtype=float)
            perturbed=augmented.iloc[te]
            altered=np.asarray(estimator.predict_proba(perturbed)[:,1],dtype=float)
            score=score_metrics(labels,proba)
            drift=float(np.mean(abs(altered-proba)))
            records.append({
                "variant":name,"fold":index,"status":"ok",
                "n_train":len(tr),"n_train_augmented":len(ytrain),
                "n_test":len(te),"n_groups_test":fold.n_groups_test,
                "threshold_ppm":fold.label_threshold_ppm,
                "mean_pixel_perturbation_score_shift":drift,**score,
            })
            predictions.extend({
                "variant":name,"fold":index,"row_index":int(j),
                "y_true":int(yi),"score":float(pi),
                "augmented_score":float(ai),"spatial_group":str(groups.iloc[j])
            } for j,yi,pi,ai in zip(te,labels,proba,altered))
            print(
                "M2_CV_FOLD",name,index,"ROC",round(score["roc_auc"],4),
                "AP",round(score["average_precision"],4),
                "BRIER",round(score["brier"],4),"GROUPS",fold.n_groups_test,
                flush=True,
            )
    measures=pd.DataFrame(records)
    measures.to_csv(out/"m2_grouped_fold_metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"m2_grouped_oof_predictions.csv",index=False)
    valid=measures.loc[measures.status=="ok"]
    averages=valid.groupby("variant").agg(
        mean_auc=("roc_auc","mean"),mean_ap=("average_precision","mean"),
        mean_balanced_accuracy=("balanced_accuracy","mean"),
        mean_brier=("brier","mean"),
        mean_ece=("ece_10","mean"),
        mean_noise_score_shift=("mean_pixel_perturbation_score_shift","mean"),
        n_folds=("fold","nunique"),
    ).reset_index()
    averages.to_csv(out/"m2_experiment_rankings.csv",index=False)
    print("M2_RANKINGS",averages.to_string(index=False),flush=True)
    rng=GroupShuffleSplit(n_splits=3,test_size=.2,random_state=42)
    stress=[]
    for index,(train_idx,test_idx) in enumerate(rng.split(x,groups=groups),start=1):
        threshold=float(np.quantile(lithium.iloc[train_idx],.75))
        y_train=(lithium.iloc[train_idx]>=threshold).astype(int)
        y_test=(lithium.iloc[test_idx]>=threshold).astype(int)
        if y_train.nunique()<2 or y_test.nunique()<2:
            stress.append({"repeat":index,"status":"SKIPPED_SINGLE_CLASS"})
            continue
        est=clone(regular)
        est.fit(x.iloc[train_idx],y_train)
        y_score=est.predict_proba(x.iloc[test_idx])[:,1]
        stress.append({
            "repeat":index,"status":"ok",
            "threshold_ppm":threshold,"n_train":len(train_idx),
            "n_test":len(test_idx),"n_groups_test":int(groups.iloc[test_idx].nunique()),
            **score_metrics(y_test,y_score),
        })
    pd.DataFrame(stress).to_csv(out/"m2_repeated_group_holdouts.csv",index=False)

    # Save a development-only model, not a production or release bundle.
    eligible=averages.loc[
        averages.variant.isin(["rf_regularized","rf_pixel_augmented","rf_group_calibrated"])
        & averages.n_folds.eq(5)
    ].sort_values(["mean_auc","mean_ap"],ascending=False)
    selected=(
        str(eligible.iloc[0].variant)
        if not eligible.empty else "rf_regularized"
    )
    if selected=="rf_group_calibrated":
        splits=_calibration_splits(x,(lithium>=float(lithium.quantile(.75))).astype(int),groups)
        if splits is None:
            selected="rf_regularized"
    spec=next(z for z in specs if z[0]==selected)
    trained=clone(spec[1])
    labels_full=(lithium>=float(lithium.quantile(.75))).astype(int)
    xfull=x
    yfull=labels_full.to_numpy()
    if selected=="rf_pixel_augmented":
        xfull=pd.concat([xfull,augmented],ignore_index=True)
        yfull=np.concatenate([yfull,yfull])
    if selected=="rf_group_calibrated":
        trained=CalibratedClassifierCV(estimator=trained,method="sigmoid",cv=splits)
    trained.fit(xfull,yfull)
    operating_threshold = (
        float(np.quantile(
            trained.predict_proba(x)[:,1], 1.0-float(labels_full.mean())
        )) if selected == "rf_group_calibrated" else 0.50
    )
    folder=out/"experimental_models"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        {"estimator":trained,"features":columns,"band_names":new["band_names"],
         "version":"DEVELOPMENT_ONLY_NOT_RELEASE","algorithm":"random_forest",
         "operating_threshold":operating_threshold,
         "operating_threshold_source":"TRAIN_ONLY_PREVALENCE_RULE"},
        folder/"model_2_development_only.joblib",
    )
    meta={
        "phase":"DEVELOPMENT_ONLY_NOT_RELEASE",
        "selected":selected,
        "operating_threshold":operating_threshold,
        "operating_threshold_source":"TRAIN_ONLY_PREVALENCE_RULE",
        "reference_training_rows":len(x),
        "folds":5,"seed":42,
        "augmentation":"Pixel-level brightness/noise/missing; no independent SCL scene labels",
        "fold_labels":"Outer-training-only 75th percentile; no target-informed fold allocation",
        "images_hash":hashlib.sha256(manifestfile.read_bytes()).hexdigest(),
        "model_is_in_experimental_results_only":True,
        "rankings":averages.to_dict("records"),
        "spatial_group_distribution_warning":"156/706 in one group",
        "no_final_release_gate":True,
        "heldout_repeated":stress,
    }
    (out/"m2_experiment_manifest.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding='utf8'
    )
    return meta


if __name__=="__main__":
    result=run_experiments()
    print("M2_DEVELOPMENT_DONE",result["selected"],flush=True)
