"""Separated nested_decision_probe responsibility; preserve prior public API."""
from __future__ import annotations

def nested_decision_probe(contract,predictors,folds, *,arm):
    """Before outer test, determine threshold via 3 geographically grouped inner fits.

    Thresholds, including reference Li q75, use inner TRAIN-only information.
    Decision estimates still exploratory because these arms are reviewed on
    the same 706 outer observations. No probability calibrator is fitted.
    """
    from lithiumscope.model_2.methodology.ablations import HISTORICAL_N, StratifiedGroupKFold, define_outer_fold, fit_classifier, freeze_inner_oof_threshold, np, pd, select_train_indices, site_weights
    from lithiumscope.model_2.methodology.decision import freeze_inner_oof_threshold
    records=[]
    all_outer=[]
    for k,fold in enumerate(folds,1):
        full_train,hist_train,outer_test,outer_threshold=define_outer_fold(
            fold,contract)
        tr=select_train_indices(arm,fold,contract,threshold=outer_threshold)
        # Use only sites in current outer train for nested validation.
        train_group=contract.spatial_group.iloc[tr].astype(str).reset_index(drop=True)
        split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
        outer_labels=(contract.Li_icpms.iloc[tr].to_numpy(float)>=outer_threshold).astype(int)
        oof_rows=[]
        # No target from outer TEST used in inner training, labels, or threshold selection.
        for inner,(i_tr,i_val) in enumerate(split.split(
                predictors.iloc[tr],outer_labels,groups=train_group),1):
            inner_global_tr=tr[i_tr]
            inner_global_valid=tr[i_val]
            inner_hist=inner_global_tr[inner_global_tr<HISTORICAL_N]
            if len(inner_hist)<10:
                raise ValueError("Too few historical inner train rows")
            inner_threshold=float(np.quantile(
                contract.Li_icpms.iloc[inner_hist].to_numpy(float),.75))
            y_train=(contract.Li_icpms.iloc[inner_global_tr].to_numpy(float)>=inner_threshold).astype(int)
            y_valid=(contract.Li_icpms.iloc[inner_global_valid].to_numpy(float)>=inner_threshold).astype(int)
            if len(np.unique(y_valid))<2:
                raise ValueError("Inner fold has one validation class; no random fallback")
            w=site_weights(contract,inner_global_tr) if arm=="expanded_inverse_site_weights" else None
            score=fit_classifier(predictors.iloc[inner_global_tr],
                y_train,predictors.iloc[inner_global_valid],weights=w)
            for orig,yt,s in zip(inner_global_valid,y_valid,score):
                oof_rows.append({"row_index":int(orig),
                    "group":str(contract.spatial_group.iloc[orig]),
                    "inner_fold":inner,"y_true":int(yt),"score":float(s)})
        frozen=freeze_inner_oof_threshold(
            pd.DataFrame(oof_rows),permitted_outer_train_indices=tr,
            reserved_outer_test_indices=np.asarray(fold["test_indices"],int))
        # Use the outer TEST score of the already fitted outer model.
        records.append({"fold":k,"arm":arm,"decision_threshold":frozen.selection.threshold,
                        "train_rows":len(tr),"inner_groups":frozen.group_count,
                        "inner_oof_sha256":frozen.inner_rows_sha256})
    return pd.DataFrame(records)
