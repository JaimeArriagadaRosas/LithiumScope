"""Separated run_experiments responsibility; preserve prior public API."""
from __future__ import annotations

from lithiumscope.experiments.analysis.run_vnext_m2 import PROJECT_ROOT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.experiments.analysis.run_vnext_m2 import Path

def run_experiments(
    root: Path = PROJECT_ROOT,
    destination: Path | None = None,
) -> dict:
    from lithiumscope.experiments.analysis.run_vnext_m2 import CalibratedClassifierCV, GroupShuffleSplit, PROJECT_ROOT, Path, RESULTS_DIR, _calibration_splits, clone, hashlib, joblib, json, make_augmentation_table, make_scientific_folds, np, pd, score_metrics
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
