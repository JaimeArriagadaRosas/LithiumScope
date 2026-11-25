"""Controlled synthetic HTTP, Optuna, process and GPU benchmark; no production data."""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import hashlib
import json
import threading
import time
from urllib.request import build_opener, ProxyHandler

import numpy as np

from lithiumscope.resources.allocation.policy import ResourcePolicy
from lithiumscope.resources.allocation.scheduler import Task, run_tasks
from lithiumscope.resources.detection.host import detect_host
from lithiumscope.resources.monitoring.monitor import monitor_operation
from lithiumscope.results.atomic_exports import write_json


def benchmark_worker(url, expected_sha, seed, backend):
    """Fixed synthetic cohort and fixed search grid in an independent process."""
    from urllib.parse import urlparse
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
    from lithiumscope.resources.manager import cpu_worker_budget
    import optuna
    if urlparse(url).hostname!="127.0.0.1":
        raise ValueError("Benchmark only reads its controlled loopback HTTP fixture")
    times={}
    def timed(name,fn):
        start=time.perf_counter()
        result=fn()
        times[name]=time.perf_counter()-start
        return result
    def retrieve():
        opener=build_opener(ProxyHandler({}))
        with opener.open(url,timeout=10) as response:
            return response.read()
    raw=timed("acquisition_http_loopback",retrieve)
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError("Benchmark input drift")
    data=np.load(BytesIO(raw),allow_pickle=False)
    x=timed("feature_preparation",lambda:np.concatenate((data.mean(axis=(2,3)),data.std(axis=(2,3))),axis=1))
    y=20+3*x[:,0]+np.random.default_rng(seed).normal(size=len(x))
    groups=np.arange(len(y))//4
    train,test=next(GroupKFold(n_splits=3).split(x,y,groups))
    splits=list(GroupKFold(n_splits=2).split(x[train],y[train],groups[train]))
    threads=cpu_worker_budget()
    def fit(depth,indices):
        model=XGBRegressor(n_estimators=8,max_depth=depth,tree_method="hist",device=backend,
                           random_state=seed,n_jobs=threads,verbosity=0)
        return model.fit(x[indices],y[indices])
    def optimize():
        def objective(trial):
            depth=trial.suggest_categorical("max_depth",[2,3])
            errors=[float(np.mean((fit(depth,train[a]).predict(x[train[b]])-y[train[b]])**2)) for a,b in splits]
            return float(np.mean(errors))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(direction="minimize",sampler=optuna.samplers.GridSampler({"max_depth":[2,3]},seed=seed))
        study.optimize(objective,n_trials=2,n_jobs=1)
        return sorted((int(t.params["max_depth"]),float(t.value)) for t in study.trials)
    scores=timed("optuna_fixed_grid",optimize)
    depth=min(scores,key=lambda p:(p[1],p[0]))[0]
    model=timed("training",lambda:fit(depth,train))
    pred=timed("inference",lambda:model.predict(x[test]))
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite benchmark predictions")
    config=json.loads(model.get_booster().save_config())
    device=config["learner"]["generic_param"].get("device","unknown")
    if backend=="cuda" and not str(device).startswith("cuda"):
        raise RuntimeError("Requested GPU benchmark fell back to CPU")
    return {"predictions":pred.tolist(),"selected_depth":depth,"scores":scores,
            "threads":threads,"backend_requested":backend,"backend_observed":device,"seconds":times}


class QuietHTTP(SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass


def _results(report):
    records=[r for wave in report["waves"] for r in wave["results"]]
    return {r["task_id"]:r.get("result") for r in records if r["state"]=="completed"}


def compare(output,*,execute=False,rows=96,seed=42):
    """Same two tasks under one-worker/two-worker policy, then optional CUDA."""
    if rows<48:
        raise ValueError("Benchmark requires at least 48 synthetic rows")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    host=detect_host()
    report={"state":"preflight","scope":"synthetic benchmark; HTTP loopback, not satellite/provider network",
            "host":asdict(host),"rows":rows,"seed":seed,"real_data":False,"runs":[],
            "gpu_state":"not_executed","hard_memory_limit":False}
    path=output/"comparison.json"
    write_json(path,report)
    if not execute:
        return report
    buffer=BytesIO()
    np.save(buffer,np.random.default_rng(seed).normal(size=(rows,6,8,8)).astype(np.float32),allow_pickle=False)
    content=buffer.getvalue()
    fixture=output/"fixture.npy"
    fixture.write_bytes(content)
    sha=hashlib.sha256(content).hexdigest()
    report["fixture_sha256"]=sha
    server=ThreadingHTTPServer(("127.0.0.1",0),partial(QuietHTTP,directory=str(output)))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        url=f"http://127.0.0.1:{server.server_port}/fixture.npy"
        policies=[("serial",ResourcePolicy(max_parallel_tasks=1)),
                  ("parallel",ResourcePolicy(max_parallel_tasks=2))]
        results={}
        def run_arm(name,policy,backend):
            import multiprocessing as mp
            baseline={p.pid for p in mp.active_children()}
            tasks=[Task(f"task_{i}","lithiumscope.resources.monitoring.acceleration_benchmark:benchmark_worker",
                        {"url":url,"expected_sha":sha,"seed":seed+i,"backend":backend},
                        128*1024**2,require_gpu=backend=="cuda",
                        vram_bytes=256*1024**2 if backend=="cuda" else None) for i in range(2)]
            start=time.perf_counter()
            log=output/f"{name}.resources.jsonl"
            with monitor_operation(log,interval_seconds=.5):
                evidence=run_tasks(tasks,output/name,policy=policy,execute=True)
            released=not ({p.pid for p in mp.active_children()}-baseline)
            metrics=json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            record={"arm":name,"state":evidence["state"],"wall_seconds":time.perf_counter()-start,
                    "worker_processes_released":released,"actual_parallel_tasks":[w["plan"]["parallel_tasks"] for w in evidence["waves"]],
                    "resource_summary":{k:v for k,v in metrics.items() if k.startswith("peak_") or k in {"samples","state"}}}
            report["runs"].append(record)
            write_json(path,report)
            return _results(evidence)
        for name,policy in policies:
            results[name]=run_arm(name,policy,"cpu")
        a,b=results["serial"],results["parallel"]
        report["cpu_equivalent"]=len(a)==len(b)==2 and all(
            a[key]["selected_depth"]==b[key]["selected_depth"] and
            np.allclose(a[key]["predictions"],b[key]["predictions"],rtol=1e-6,atol=1e-6)
            for key in a)
        from lithiumscope.resources.detection.device import detect_device
        device=detect_device(prefer_gpu=True)
        if device.xgboost_cuda_available and len(host.gpus)==1 and host.gpus[0].get("vram_available_bytes",0)>=512*1024**2:
            g=run_arm("cuda",ResourcePolicy(max_parallel_tasks=1),"cuda")
            report["gpu_state"]="measured" if len(g)==2 else "failed"
            report["gpu_cpu_prediction_max_abs_difference"]={
                key:float(np.max(np.abs(np.asarray(a[key]["predictions"])-np.asarray(g[key]["predictions"]))))
                for key in a.keys()&g.keys()}
        else:
            report["gpu_state"]="skipped_unverified_backend_or_vram"
        report["state"]="complete" if report["cpu_equivalent"] and all(r["worker_processes_released"] and r["state"]=="completed" for r in report["runs"]) else "partial"
    except BaseException:
        report["state"]="interrupted"
        raise
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        fixture.unlink(missing_ok=True)
        write_json(path,report)
    return report
