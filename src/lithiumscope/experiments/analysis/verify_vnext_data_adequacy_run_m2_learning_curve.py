"""Separated run_m2_learning_curve responsibility; preserve prior public API."""
from __future__ import annotations

from lithiumscope.experiments.analysis.verify_vnext_data_adequacy import OUT, PROJECT_ROOT
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.experiments.analysis.verify_vnext_data_adequacy import Path

def run_m2_learning_curve(
    root: Path = PROJECT_ROOT, out: Path = OUT,
    *,
    fractions: tuple[float,...] = (.35,.6,.8,1.0),
    seeds: tuple[int,...] = (13,23,37),
    trees: int = 200,
) -> dict:
    """Run exploratory spatial heldout learning curve using fixed model capacity."""
    from lithiumscope.experiments.analysis.verify_vnext_data_adequacy import OUT, PROJECT_ROOT, Path, _json, average_precision_score, brier_score_loss, clone, group_train_indices, joblib, make_scientific_folds, np, pd, roc_auc_score
    feature_path=root/"data/processed/model_2/sentinel_features.csv"
    df=pd.read_csv(feature_path)
    bundle=joblib.load(root / (
        "models/model_2/trained/training_20261105_223618_m0300/model.joblib"
    ))
    x=df[list(bundle["features"])]
    y=df.Li_icpms
    groups=df.spatial_group.astype("string")
    folds=make_scientific_folds(y, groups,n_splits=5,quantile=.75)
    rows=[]
    predictions=[]
    for f_id,fold in enumerate(folds,start=1):
        test_labels=np.asarray(fold.y_test,dtype=int)
        test=fold.test_idx
        prior_ap=float(test_labels.mean())
        for fraction in fractions:
            for seed in seeds[:1] if fraction==1 else seeds:
                selected=group_train_indices(
                    fold.train_idx,groups,fraction,seed+f_id
                )
                ytrain=(y.iloc[selected].to_numpy(dtype=float)
                        >=fold.label_threshold_ppm).astype(int)
                if len(np.unique(ytrain))<2:
                    rows.append({
                        "fold":f_id,"fraction":fraction,"seed":seed,
                        "status":"skipped_one_train_class","n_train":len(selected)
                    })
                    continue
                if set(groups.iloc[selected]).intersection(set(groups.iloc[test])):
                    raise RuntimeError("Spatial leakage detected")
                estimator=clone(bundle["estimator"])
                estimator.set_params(n_estimators=trees,n_jobs=4)
                estimator.fit(x.iloc[selected],ytrain)
                score=estimator.predict_proba(x.iloc[test])[:,1]
                if np.unique(test_labels).size!=2:
                    raise RuntimeError("Test class distribution changed")
                row={
                    "fold":f_id,"fraction":fraction,"seed":seed,"status":"ok",
                    "n_train":len(selected),
                    "n_groups_train":int(groups.iloc[selected].nunique()),
                    "n_test":len(test),"n_groups_test":fold.n_groups_test,
                    "train_positive_fraction":float(ytrain.mean()),
                    "test_positive_fraction":prior_ap,
                    "outer_train_q75_threshold":fold.label_threshold_ppm,
                    "roc_auc":float(roc_auc_score(test_labels,score)),
                    "ap":float(average_precision_score(test_labels,score)),
                    "ap_gain_over_test_prevalence":float(average_precision_score(test_labels,score)-prior_ap),
                    "brier":float(brier_score_loss(test_labels,score)),
                }
                rows.append(row)
                if fraction==1:
                    predictions.extend({
                        "outer_fold":f_id,"sample_index":int(i),
                        "spatial_group":str(groups.iloc[i]),
                        "label":int(label),"score":float(prediction),
                    } for i,label,prediction in zip(test,test_labels,score))
                print(
                    "M2_CURVE",f_id,"frac",fraction,"seed",seed,
                    "n",len(selected),"groups",row["n_groups_train"],
                    "AUC",round(row["roc_auc"],4),"AP",round(row["ap"],4),
                    flush=True,
                )
    result=pd.DataFrame(rows)
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"m2_grouped_learning_curve_folds.csv",index=False)
    good=result[result.status=="ok"]
    per_fold=good.groupby(["fold","fraction"],as_index=False).agg(
        n_train=("n_train","mean"),n_groups=("n_groups_train","mean"),
        auc=("roc_auc","mean"),ap=("ap","mean"),
        ap_gain=("ap_gain_over_test_prevalence","mean"),
        brier=("brier","mean"),repeats=("seed","count"),
    )
    per_fold.to_csv(out/"m2_grouped_learning_curve_per_fold.csv",index=False)
    avg=per_fold.groupby("fraction").agg(
        train_samples=("n_train","mean"),train_groups=("n_groups","mean"),
        roc_auc_mean=("auc","mean"),roc_auc_std=("auc","std"),
        average_precision_mean=("ap","mean"),ap_gain_mean=("ap_gain","mean"),
        brier_mean=("brier","mean"),n_folds=("fold","nunique"),
    ).reset_index()
    avg.to_csv(out/"m2_grouped_learning_curve_summary.csv",index=False)
    pd.DataFrame(predictions).to_csv(
        out/"m2_grouped_learning_curve_full_oof.csv",index=False
    )
    summary={
        "purpose":"Assess marginal gains of more spatial groups/samples, not a final candidate.",
        "model":"RF fixed 200 trees, remaining saved hyperparameters unchanged",
        "n_outer_folds":5,"fractions":list(fractions),"seeds":list(seeds),
        "rows":len(result),"full_oof_rows":len(predictions),
        "mean_by_fraction":avg.to_dict("records"),
        "protocol":"Fixed outer train-derived q75 per fold across all fractions; whole-group subsampling; exactly same outer test indices",
        "not_external_validation":True,
    }
    _json(out/"m2_learning_curve_decision_evidence.json",summary)
    return summary
