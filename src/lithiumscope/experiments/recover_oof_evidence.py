"""Audit immutable training evidence; no fitting or changes to original."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(r"D:\Documentos\MDL\Informatico\Apredizaje\LithiumScope-Scientific-Lab")
ORIGINAL=Path(r"D:\Descargas\LithiumScope")
run_id="training_20260920_195933_m0300"
src=ORIGINAL/"results"/"model_2"/"runs"/run_id/"tables"
release=ROOT/"models"/"model_2"/"trained"/run_id
out=ROOT/"results"/"scientific_simulations"/"academic_evidence"
out.mkdir(parents=True,exist_ok=True)
data_path=ORIGINAL/"data"/"processed"/"model_2"/"training_manifest.csv"
manifest=pd.read_csv(data_path)
oof=pd.read_csv(src/"oof_predictions_catboost.csv")
folds=pd.read_csv(src/"fold_metrics_catboost.csv")
ranking=pd.read_csv(src/"competition_ranking.csv")
baseline=pd.read_csv(src/"baseline_fold_metrics.csv")
threshold=pd.read_csv(src/"winner_operating_threshold.csv")
meta=json.loads((release/"metadata.json").read_text(encoding="utf8"))
meta_dataset=json.loads((release/"dataset_manifest.json").read_text(encoding="utf8"))
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
data_hash=h(data_path)
y=oof["y_true"].to_numpy()
scores=oof["prospectivity_score"].to_numpy()
q=manifest.Li_icpms.quantile(.75)
expected=(manifest.Li_icpms>=q).astype(int).to_numpy()
operating=float(meta["operating_threshold"])
metrics={
    "roc_auc_oof_pooled":float(roc_auc_score(y,scores)),
    "average_precision_oof_pooled":float(average_precision_score(y,scores)),
    "balanced_accuracy_oof_at_0_5":float(balanced_accuracy_score(y,scores>=.5)),
    "balanced_accuracy_oof_at_operating":float(balanced_accuracy_score(y,scores>=operating)),
}
folds_n=int(len(folds))
splitter=StratifiedGroupKFold(n_splits=folds_n,shuffle=True,random_state=42)
checks=[]
for ix,(train,test) in enumerate(splitter.split(manifest,expected,manifest.spatial_group),1):
    checks.append({"fold":ix,"train":len(train),"test":len(test),
       "overlapping_spatial_groups":len(set(manifest.spatial_group.iloc[train]) & set(manifest.spatial_group.iloc[test])),
       "recorded_train":int(folds.iloc[ix-1].n_train),
       "recorded_test":int(folds.iloc[ix-1].n_test)})
pd.DataFrame(checks).to_csv(out/"reconstructed_spatial_splits.csv",index=False)
record={
 "source_training_run_id":run_id,
 "source_path":str(src),
 "dataset_manifest_path":str(data_path),
 "dataset_sha256_actual":data_hash,
 "dataset_sha256_release":meta_dataset.get("source_sha256"),
 "dataset_hash_matches":data_hash==meta_dataset.get("source_sha256"),
 "row_count_match":len(oof)==len(manifest)==meta["samples"],
 "label_match":bool(np.array_equal(y,expected)),
 "threshold_ppm_reproduced":float(q),
 "threshold_ppm_metadata":float(meta["threshold_ppm"]),
 "groups_count":int(manifest.spatial_group.nunique()),
 "splitter_reconstructed_from_current_source":"StratifiedGroupKFold(seed=42,5)",
 "split_fold_sizes_match":all(z["train"]==z["recorded_train"] and z["test"]==z["recorded_test"] for z in checks),
 "split_group_overlaps":sum(z["overlapping_spatial_groups"] for z in checks),
 "metrics_recomputed":metrics,
 "metrics_metadata":meta.get("metrics",{}),
 "fold_means":{k:float(folds[k].mean()) for k in ("roc_auc","average_precision","balanced_accuracy")},
 "baseline_fold_means":{k:float(baseline[k].mean()) for k in ("roc_auc","average_precision","balanced_accuracy")},
 "operating_threshold_recorded":threshold.to_dict("records"),
 "source_hashes":{p.name:h(p) for p in [data_path,src/"oof_predictions_catboost.csv",src/"fold_metrics_catboost.csv",src/"competition_ranking.csv",src/"baseline_fold_metrics.csv",src/"winner_operating_threshold.csv"]},
 "limitations":["OOF data generated using global dataset q75 threshold before CV; risk of label leakage.",
 "Operating threshold selected on winning model OOF; do not consider operating-threshold OOF score as external estimate.",
 "CV spatial group isolation does not establish independent geographical holdout or real-world prospectivity accuracy.",
 "Reconstructed group folds reflect current code and seed; agreement of fold sizes is supporting, not conclusive proof of original split indices."],
}
(out/"recovered_training_validation.json").write_text(json.dumps(record,indent=2,ensure_ascii=False,default=str),encoding="utf8")
pd.DataFrame([{**{"algorithm":"catboost","type":"oof_pooled"},**metrics}]).to_csv(out/"recovered_oof_metrics.csv",index=False)
print(json.dumps({k:v for k,v in record.items() if k in ("dataset_hash_matches","label_match","split_fold_sizes_match","split_group_overlaps","metrics_recomputed","fold_means","baseline_fold_means")},indent=2),flush=True)
