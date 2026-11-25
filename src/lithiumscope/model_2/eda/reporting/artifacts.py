"""Export spectral EDA statistics, sample identity and measured local QA."""
from __future__ import annotations

import pandas as pd


def write_descriptive_tables(tables,combined,target,group,metadata,qa,directory):
    for name,frame in tables.items():
        frame.to_csv(directory/f"{name}.csv",index=name=="spearman_correlations")
    combined[[target,group,*metadata]].to_csv(directory/"sample_provenance.csv",index=False)
    if "sentinel_scene_id" in combined:
        scene=combined.groupby("sentinel_scene_id",dropna=False)[target].agg(["count","median"])
        for column in ("sentinel_cloud_cover","sixband_joint_valid_fraction","rededge_joint_valid_fraction"):
            if column in combined:
                scene[column+"_mean"]=pd.to_numeric(combined[column],errors="coerce").groupby(combined.sentinel_scene_id).mean()
        scene.to_csv(directory/"scene_quality.csv")
    if not qa.empty:
        qa.to_csv(directory/"patch_quality.csv",index=False)
