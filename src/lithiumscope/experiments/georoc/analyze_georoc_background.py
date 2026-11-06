"""Consolidate exploratory external batch without making verified accuracy claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
B=Path("results/scientific_simulations/external_geochemistry_audit/background_georoc_exploratory")
def main():
 m=pd.read_csv(B/"patch_manifest.csv")
 samples=pd.read_csv(B/"sample_image_map.csv")
 records=[]
 for row in m.itertuples():
  f=B/"scores"/row.image_id/"image_scenario_metrics.csv"
  if not f.exists():continue
  x=pd.read_csv(f)
  base=x.loc[x.scenario.eq("baseline") & x.status.eq("SCORED")]
  if len(base)!=1: continue
  records.append({"image_id":row.image_id,"score":float(base.iloc[0].score),
      "ood_fraction":float(base.iloc[0].ood_fraction),
      "scenario_rows":len(x),"scenario_scored":int(x.status.eq("SCORED").sum()),
      "scenario_abstained":int(x.status.ne("SCORED").sum()),
      "scenario_mean_abs_shift":float(pd.to_numeric(x.absolute_score_shift,errors="coerce").mean())})
 scores=pd.DataFrame(records)
 sm=samples.merge(m[["image_id","status","scene_id","scene_datetime","local_cloud_fraction"]],on="image_id",validate="many_to_one").merge(scores,on="image_id",how="left",validate="many_to_one")
 sm["label_21_6"]=sm.li_ppm.ge(21.6).astype(int)
 sm["scored"]=sm.score.notna()
 sm.to_csv(B/"final_sample_scores.csv",index=False)
 scores.to_csv(B/"final_coordinate_scores.csv",index=False)
 valid=sm.loc[sm.scored].copy()
 # Sample- and location-weighted descriptive statistics are distinct due to duplicate sites.
 coord=valid.groupby("image_id").agg(score=("score","first"),
     contains_high=("label_21_6","max"),contains_low=("label_21_6",lambda s:int(s.eq(0).any())),
     n_samples=("sample_name","size")).reset_index()
 summary={"coordinates_total":len(m),"coordinates_scored":len(scores),
 "coordinates_unscored":int(m.status.ne("ACCEPT").sum()),
 "unscored_reason_counts":m.loc[m.status.ne("ACCEPT"),"status"].value_counts().to_dict(),
 "sample_rows_total":len(sm),"sample_rows_scored":len(valid),
 "sample_rows_not_scored":int((~sm.scored).sum()),
 "class_balance_all":sm.label_21_6.value_counts().sort_index().to_dict(),
 "class_balance_scored":valid.label_21_6.value_counts().sort_index().to_dict(),
 "coordinate_rows_with_high":int(coord.contains_high.sum()),
 "coordinate_rows_with_low":int(coord.contains_low.sum()),
 "coordinates_both_classes":int((coord.contains_high.eq(1)&coord.contains_low.eq(1)).sum()),
 "scenario_rows_total":int(scores.scenario_rows.sum()),
 "scenario_scored":int(scores.scenario_scored.sum()),
 "scenario_abstained":int(scores.scenario_abstained.sum()),
 "score_summary":valid.score.describe().to_dict(),
 "mean_ood_fraction":float(scores.ood_fraction.mean()),
 "external_validation_status":"EXPLORATORY_ONLY_NOT_VERIFIED"}
 if valid.label_21_6.nunique()==2:
  summary["naive_sample_roc_auc"]=float(roc_auc_score(valid.label_21_6,valid.score))
  summary["naive_sample_average_precision"]=float(average_precision_score(valid.label_21_6,valid.score))
  summary["naive_sample_balanced_accuracy_0_5"]=float(balanced_accuracy_score(valid.label_21_6,valid.score>=.5))
  summary["naive_sample_balanced_accuracy_historical_threshold"]=float(balanced_accuracy_score(valid.label_21_6,valid.score>=.0875222435003305))
 summary["caveats"]=["Repeated Li sample rows at a single satellite coordinate create pseudoreplication",
 "Sample labels were not independently approved","This is not an unbiased held-out external validation",
 "Some locations contain mixed Li labels and a single shared spectral score",
 "Threshold 0.08752 was tuned on historical OOF and was not independently calibrated"]
 (B/"final_exploratory_metrics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
 print("FINAL",json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
