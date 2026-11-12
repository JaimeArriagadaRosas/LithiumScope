"""Report top-1 selection between three distinct sites for all M2 algorithms.

This metric is diagnostic: it never picks the winner or promotes a release.
Only held-out predictions from a common outer fold are compared.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from lithiumscope.model_2.methodology.three_zone_ranking import ranking_evaluation


def save_three_zone_competition(
    manifest_path:Path,
    probabilities_by_algorithm:dict[str,np.ndarray],
    fold_ids:np.ndarray,
    test_labels:np.ndarray,
    destination:Path,
)->dict:
    frame=pd.read_csv(manifest_path)
    required={"sample_id","Li_icpms","latitude","longitude"}
    if not required.issubset(frame):
        raise ValueError("Three-zone diagnostic requires identified Li points")
    n=len(frame)
    fold_ids=np.asarray(fold_ids,dtype=int)
    test_labels=np.asarray(test_labels,dtype=int)
    if len(fold_ids)!=n or np.any(fold_ids<1) or len(test_labels)!=n:
        raise ValueError("Outer fold indices or test labels incomplete")
    if not probabilities_by_algorithm:
        raise ValueError("No successful M2 algorithms to evaluate")
    site=(
        frame.latitude.astype(float).round(5).astype(str)+"_"+
        frame.longitude.astype(float).round(5).astype(str)
    )
    samples=[]
    for algorithm,prob in probabilities_by_algorithm.items():
        score=np.asarray(prob,dtype=float)
        if score.shape!=(n,) or not np.isfinite(score).all():
            raise ValueError(f"Algorithm {algorithm} lacks complete OOF predictions")
        samples.append(pd.DataFrame({
            "arm":algorithm,"fold":fold_ids,"test_index":np.arange(n),
            "sample_id":frame.sample_id.astype(str),
            "site_screening_id":site,"li_ppm":frame.Li_icpms.astype(float),
            "y_true":test_labels,"score":score,
        }))
    summary,cases=ranking_evaluation(
        pd.concat(samples,ignore_index=True),triples_per_fold=250,seed=42,
    )
    destination.mkdir(parents=True,exist_ok=True)
    cases.to_csv(destination/"three_zone_competition_cases.csv",index=False)
    (destination/"three_zone_competition_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8",
    )
    return summary
