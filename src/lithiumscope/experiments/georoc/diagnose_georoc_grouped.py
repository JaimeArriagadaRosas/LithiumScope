"""Diagnostic-only analysis: grouped uncertainty and source/site sensitivity for GEOROC."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
B=Path("results/scientific_simulations/external_geochemistry_audit/background_georoc_exploratory")
def metric(data):
 y=data.label_21_6.to_numpy(dtype=int);s=data.score.to_numpy(dtype=float)
 if len(np.unique(y))!=2:return None
 return {"roc_auc":float(roc_auc_score(y,s)),"ap":float(average_precision_score(y,s)),
 "ba_0_5":float(balanced_accuracy_score(y,s>=.5)),
 "ba_historical_0_0875":float(balanced_accuracy_score(y,s>=.0875222435003305)),
 "n":len(data),"high":int(y.sum())}
def main():
 frame=pd.read_csv(B/"final_sample_scores.csv")
 ok=frame.loc[frame.scored].copy()
 coord=ok.groupby("image_id").agg(score=("score","first"),li_ppm=("li_ppm","median"),
  n_samples=("li_ppm","size"),n_high=("label_21_6","sum"),
  citation_count=("citation_number","nunique"),citation_number=("citation_number","first"),
  latitude=("latitude","first"),longitude=("longitude","first")).reset_index()
 coord["label_21_6"]=coord.li_ppm.ge(21.6).astype(int)
 coord["mixed"]=coord.n_high.gt(0)&coord.n_high.lt(coord.n_samples)
 coord.to_csv(B/"diagnostic_site_level.csv",index=False)
 results={"sample_naive":metric(ok),"site_median_li":metric(coord),
  "site_median_li_exclude_mixed":metric(coord.loc[~coord.mixed]),
  "unique_single_sample_sites":metric(coord.loc[coord.n_samples.eq(1)]),
  "sites":len(coord),"mixed_sites":int(coord.mixed.sum()),
  "multi_sample_sites":int(coord.n_samples.gt(1).sum())}
 # Bootstrap entire coordinates, not samples, with fixed reproducible RNG.
 rng=np.random.default_rng(42)
 ids=coord.image_id.to_numpy()
 bins={k:[] for k in ("sample_cluster","site_median_li")}
 grouped={k:g for k,g in ok.groupby("image_id",sort=False)}
 for _ in range(1200):
  chosen=ids[rng.integers(0,len(ids),len(ids))]
  sites=coord.set_index("image_id").loc[chosen].reset_index()
  v=metric(sites)
  if v is not None:bins["site_median_li"].append(v["roc_auc"])
  sample=pd.concat([grouped[k] for k in chosen],ignore_index=True)
  v=metric(sample)
  if v is not None:bins["sample_cluster"].append(v["roc_auc"])
 results["cluster_bootstrap_auc"]={k:{"n":len(v),"lower_95":float(np.quantile(v,.025)),
  "median":float(np.median(v)),"upper_95":float(np.quantile(v,.975))} for k,v in bins.items()}
 # Leave-one-publication-out effect assesses source concentration.
 rows=[]
 for cit,g in ok.groupby("citation_number"):
  rest=ok.loc[ok.citation_number.ne(cit)]
  val=metric(rest)
  rows.append({"citation_number":int(cit),"rows_removed":len(g),"high_removed":int(g.label_21_6.sum()),
   "auc_after_removal":val["roc_auc"] if val else np.nan,
   "ap_after_removal":val["ap"] if val else np.nan})
 pd.DataFrame(rows).sort_values("rows_removed",ascending=False).to_csv(B/"diagnostic_leave_citation_out.csv",index=False)
 vals=[z["auc_after_removal"] for z in rows if np.isfinite(z["auc_after_removal"])]
 results["source_sensitivity"]={"source_count":len(rows),"min_leave_one_out_auc":float(min(vals)),
  "max_leave_one_out_auc":float(max(vals))}
 summary=frame.groupby("scored").agg(samples=("li_ppm","size"),high=("label_21_6","sum"),
  median_li=("li_ppm","median"),distinct_sites=("image_id","nunique")).reset_index()
 summary.to_csv(B/"diagnostic_coverage_bias.csv",index=False)
 results["coverage"]=summary.to_dict("records")
 results["interpretation"]="DIAGNOSTIC_ONLY; no sample-level independent labels verified"
 (B/"diagnostic_grouped_statistics.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf8")
 print(json.dumps(results,indent=2,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
