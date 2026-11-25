"""Paired synthetic local pipeline benchmark; opt-in, no trained bundle or remote data."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.resources.allocation.policy import ResourcePolicy,Workload,plan_workload
from lithiumscope.resources.detection.host import detect_host
from lithiumscope.resources.manager import worker_plan_scope
from lithiumscope.resources.monitoring.monitor import monitor_operation
from lithiumscope.results.atomic_exports import write_json


def _timed(events,name,call):
    start=time.perf_counter()
    result=call()
    events[name]=time.perf_counter()-start
    return result


def run(output:Path,*,execute=False,rows=240,seed=42,repeats=2):
    if rows<60 or repeats<2:
        raise ValueError("At least 60 rows and two paired repetitions")
    output.mkdir(parents=True,exist_ok=False)
    host=detect_host(include_gpu=False)
    base=plan_workload(host,ResourcePolicy(max_cpu_fraction=1.),Workload())
    budgets=sorted({1,base.cpu_budget})
    report={"state":"preflight","scope":"synthetic local I/O and CPU benchmark; no satellite network or GPU benchmark",
            "host":asdict(host),"seed":seed,"rows":rows,"repeats":repeats,"budgets":budgets,
            "real_data":False,"bundle_saved":False,"results":[]}
    path=output/"benchmark.json"
    write_json(path,report)
    if not execute:
        return report
    rng=np.random.default_rng(seed)
    cubes=rng.normal(size=(rows,6,32,32)).astype(np.float32)
    y=20+3*cubes[:,0].mean(axis=(1,2))+rng.normal(size=rows)
    source=output/"synthetic_patches.npy"
    np.save(source,cubes,allow_pickle=False)
    report["source_sha256"]=file_sha256(source)
    groups=np.arange(rows)//5
    train,test=next(GroupKFold(n_splits=3).split(cubes,y,groups))
    inner=list(GroupKFold(n_splits=3).split(train,groups=groups[train]))
    reference=None
    from dataclasses import replace
    try:
        for repeat in range(repeats):
            # Reverse order on alternating repetitions to reduce systematic warm-cache bias.
            for threads in (budgets if repeat%2==0 else list(reversed(budgets))):
                events={}
                plan=replace(base,parallel_tasks=1,threads_per_task=threads)
                resource_path=output/f"repeat_{repeat}_threads_{threads}.jsonl"
                with worker_plan_scope(plan),monitor_operation(resource_path,interval_seconds=.05):
                    loaded=_timed(events,"local_acquisition",lambda:np.load(source,allow_pickle=False))
                    x=_timed(events,"preparation",lambda:np.concatenate([loaded.mean(axis=(2,3)),loaded.std(axis=(2,3))],axis=1))
                    def fit(params,indices):
                        model=RandomForestRegressor(n_estimators=12,max_depth=params,random_state=seed,n_jobs=threads)
                        return model.fit(x[indices],y[indices])
                    def optimize():
                        values=[]
                        for depth in (3,6):
                            errors=[]
                            for a,b in inner:
                                pred=fit(depth,train[a]).predict(x[train[b]])
                                errors.append(float(np.mean((pred-y[train[b]])**2)))
                            values.append((float(np.mean(errors)),depth))
                        return min(values)[1]
                    selected=_timed(events,"optimization_fixed_grid",optimize)
                    model=_timed(events,"training",lambda:fit(selected,train))
                    pred=_timed(events,"inference",lambda:model.predict(x[test]))
                    if reference is None:
                        reference=(selected,pred.copy())
                    equivalent=selected==reference[0] and np.allclose(pred,reference[1],rtol=1e-10,atol=1e-10)
                report["results"].append({"repeat":repeat,"threads":threads,"selected_depth":selected,
                    "equivalent_predictions":bool(equivalent),"seconds":events,"resource_log":resource_path.name})
                write_json(path,report)
        report["state"]="complete" if all(r["equivalent_predictions"] for r in report["results"]) else "non_equivalent"
    except BaseException:
        report["state"]="interrupted"
        raise
    finally:
        write_json(path,report)
    return report
