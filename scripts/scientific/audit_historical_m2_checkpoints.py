"""Verify all 30 historical M2 v4 checkpoints against archived OOF; read-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.checkpoints import read_checkpoint_evidence
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.scientific_checks import assert_partition_coverage, assert_group_isolation
from lithiumscope.model_2.evaluation.metrics import classification_metrics


def run(run_dir, manifest_path, output):
    manifest=pd.read_csv(manifest_path)
    if len(manifest)!=706 or file_sha256(manifest_path)!="5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1":
        raise ValueError("Frozen M2 manifest differs")
    folds_path=run_dir/"manifests/scientific_folds.json"
    folds=json.loads(folds_path.read_text(encoding="utf-8"))
    splits=[(np.asarray(f["train_indices"]),np.asarray(f["test_indices"])) for f in folds]
    assert_partition_coverage(splits,len(manifest))
    y=np.full(len(manifest),-1,dtype=int)
    ids=np.zeros(len(manifest),dtype=int)
    for i,(tr,te) in enumerate(splits,1):
        assert_group_isolation(tr,te,manifest.spatial_group)
        cutoff=float(folds[i-1]["label_threshold_ppm"])
        if not np.isclose(cutoff,np.quantile(manifest.Li_icpms.iloc[tr],.75),rtol=0,atol=1e-10):
            raise ValueError("Fold endpoint differs from TRAIN quantile")
        y[te]=(manifest.Li_icpms.iloc[te]>=cutoff).astype(int)
        ids[te]=i
    sources={str(folds_path):file_sha256(folds_path),str(manifest_path):file_sha256(manifest_path)}
    rows=[]
    for algorithm in ("catboost","extra_trees","hist_gradient_boosting","random_forest","svm_rbf","xgboost"):
        path=run_dir/"tables"/f"oof_predictions_{algorithm}.csv"
        archived=pd.read_csv(path)
        sources[str(path)]=file_sha256(path)
        if not (len(archived)==len(manifest) and np.array_equal(archived.y_true,y)
                and np.array_equal(archived.outer_fold,ids)
                and np.allclose(archived.li_ppm,manifest.Li_icpms,rtol=0,atol=1e-10)):
            raise ValueError("Historical OOF identity differs")
        for i,(_,te) in enumerate(splits,1):
            path=run_dir/"checkpoints"/algorithm/f"fold_{i:02d}.json"
            sources[str(path)]=file_sha256(path)
            record=read_checkpoint_evidence(path,allow_legacy=True)
            if record["fold"]!=i or not np.array_equal(record["test_indices"],te):
                raise ValueError("Historical checkpoint fold differs")
            scores=np.asarray(record["predictions"],float)
            delta=float(np.max(np.abs(scores-archived.prospectivity_score.iloc[te])))
            if delta>1e-12 or not isinstance(record["params"],dict):
                raise ValueError("Historical checkpoint scores or parameters differ")
            metrics=classification_metrics(y[te],scores)
            for name in ("roc_auc","average_precision","brier_score"):
                limit=1e-7 if name=="brier_score" else 1e-12
                if not np.isclose(record["metrics"][name],metrics[name],rtol=0,atol=limit):
                    raise ValueError(f"Material historical metric drift: {algorithm}/{i}/{name}")
                if not np.isclose(record["metrics"][name],metrics[name],rtol=0,atol=1e-12):
                    print("METRIC_DIFFERENCE",algorithm,i,name,record["metrics"][name],metrics[name],flush=True)
            rows.append({"algorithm":algorithm,"fold":i,"test_rows":len(te),"maximum_absolute_delta":delta,
                         "roc_auc":metrics["roc_auc"],"average_precision":metrics["average_precision"],
                         "brier_score":metrics["brier_score"],"checkpoint_sha256":sources[str(path)],
                         **{f"archived_{name}":record["metrics"][name] for name in ("roc_auc","average_precision","brier_score")},
                         **{f"delta_{name}":float(metrics[name]-record["metrics"][name]) for name in ("roc_auc","average_precision","brier_score")}})
    if any(file_sha256(Path(p))!=h for p,h in sources.items()):
        raise ValueError("Historical files changed during audit")
    output.mkdir(parents=True,exist_ok=False)
    table=pd.DataFrame(rows)
    table.to_csv(output/"fold_comparison.csv",index=False)
    report={"scope":"Historical unsealed evidence only, never resumable",
            "checkpoints":len(table),"predictions":int(table.test_rows.sum()),
            "maximum_absolute_delta":float(table.maximum_absolute_delta.max()),
            "maximum_metric_deltas":{name:float(table[f"delta_{name}"].abs().max()) for name in ("roc_auc","average_precision","brier_score")},
            "metric_tolerances":{"roc_auc":1e-12,"average_precision":1e-12,"brier_score":1e-7},
            "sources_sha256":sources,"all_sources_unchanged":True,"fit":False}
    (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print({k:v for k,v in report.items() if k!="sources_sha256"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir","manifest-path","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.run_dir,args.manifest_path,args.output)
