"""Small synthetic estimator resource benchmark. No real dataset, bundle or activation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from threadpoolctl import threadpool_info

from lithiumscope.resources.manager import resource_snapshot,resource_operation,worker_limit,svm_cache_limit
from lithiumscope.resources.allocation.policy import Workload
from lithiumscope.resources.detection.device import detect_device


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    host=resource_snapshot()
    device=detect_device(prefer_gpu=True)
    rng=np.random.default_rng(42)
    x=rng.normal(size=(1500,33))
    y=20+4*x[:,0]-2*x[:,1]+rng.normal(size=len(x))
    # Explicitly synthetic and small; no geochemical accuracy claims.
    builders={
        "random_forest":lambda:RandomForestRegressor(n_estimators=24,max_depth=6,random_state=42,n_jobs=worker_limit(-1)),
        "hgb":lambda:HistGradientBoostingRegressor(max_iter=24,early_stopping=False,random_state=42),
        "svm":lambda:SVR(cache_size=svm_cache_limit(200.)),
    }
    results=[]
    for name,builder in builders.items():
        path=output/f"{name}.resources.jsonl"
        with resource_operation(path,Workload(memory_per_task_bytes=64*1024**2)) as plan:
            model=builder()
            start=time.perf_counter()
            model.fit(x[:1200],y[:1200])
            elapsed=time.perf_counter()-start
            pred=model.predict(x[1200:])
            if pred.shape!=(300,) or not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite synthetic prediction: {name}")
            pools=[{"library":p["internal_api"],"threads":p["num_threads"]} for p in threadpool_info()]
            if any(p["threads"]>plan.threads_per_task for p in pools):
                raise ValueError(f"Native thread limit not applied: {name}")
        events=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        end=events[-1]
        results.append({"algorithm":name,"elapsed_fit_seconds":elapsed,"threads":plan.threads_per_task,
            "native_pools":pools,"peak_rss_bytes":end["peak_rss_bytes"],"peak_tree_rss_bytes":end["peak_tree_rss_bytes"],
            "peak_process_cpu_percent":end["peak_process_cpu_percent"],"samples":end["samples"],
            "finite_predictions":len(pred),"status":end["state"]})
    summary={"scope":"synthetic resource smoke only; not a training performance comparison",
        "host":host,"device":asdict(device),"estimators":results,"real_data_used":False,
        "bundle_created":False,"hard_ram_vram_cap":False}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args().output)
