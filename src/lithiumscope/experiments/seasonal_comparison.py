"""Compare repeated seasonal baselines without mistaking stress scores for Li accuracy."""
from pathlib import Path
import json
import hashlib
import pandas as pd


def compare_seasons(root: Path) -> Path:
    base=root/"results"/"scientific_simulations"
    output=base/"seasonal_comparison_2024_2025"
    output.mkdir(parents=True,exist_ok=True)
    sources={
        2024:(base/"seasonal_2024/seasonal_manifest.csv",base/"seasonal_2024/batch/batch_m2.csv"),
        2025:(base/"campaign_2025/campaign_manifest.csv",base/"campaign_2025/batch/batch_m2.csv"),
    }
    rows=[]
    hashes={}
    for year,(manifest_file,batch_file) in sources.items():
        manifest=pd.read_csv(manifest_file)
        metric=pd.read_csv(batch_file)
        for path in (manifest_file,batch_file):
            hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        labels=manifest[["image_id","status","scene_id","scene_datetime","local_cloud_fraction"]]
        labels=labels.merge(metric.loc[(metric.scenario=="baseline") & (metric.status=="SCORED"),
                        ["image_id","score","ood_fraction"]],on="image_id",how="left",validate="one_to_one")
        labels["year"]=year
        labels["region"]=labels.image_id.str.split("_").str[0]
        labels["season"]=labels.image_id.str.extract(r"_(summer|winter)_")[0]
        labels=labels.loc[labels.season.isin(("summer","winter"))].copy()
        labels["satellite"]=labels.scene_id.str.split("_").str[0]
        rows.append(labels)
    combined=pd.concat(rows,ignore_index=True)
    combined.to_csv(output/"seasonal_baselines.csv",index=False)
    paired=combined.pivot(index=["region","year"],columns="season",values="score").reset_index()
    paired["winter_minus_summer"]=paired["winter"]-paired["summer"]
    paired["absolute_seasonal_change"]=paired.winter_minus_summer.abs()
    paired.to_csv(output/"paired_seasonal_differences.csv",index=False)
    variability=paired.groupby("region").agg(
        year_pairs=("year","nunique"),
        seasonal_change_mean=("winter_minus_summer","mean"),
        seasonal_change_std=("winter_minus_summer",lambda x:x.std(ddof=0)),
        max_abs_change=("absolute_seasonal_change","max"),
    ).reset_index()
    variability.to_csv(output/"seasonal_variability_by_region.csv",index=False)
    payload={"observations":len(combined),"pairs":len(paired),
             "distinct_regions":int(combined.region.nunique()),
             "source_sha256":hashes,
             "warning":"Different acquisitions/sensors/atmospheric states; changes are descriptive, not lithium accuracy or causal seasonal effects.",
             "external_geochemical_labels":"NONE"}
    (output/"manifest.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    return output


if __name__=="__main__":
    print(compare_seasons(Path.cwd()))
