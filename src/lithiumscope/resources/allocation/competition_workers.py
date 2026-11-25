"""Opt-in CPU competition scheduling with isolated, hash-verified local results."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import joblib
import re

from lithiumscope.core.hashing import file_sha256
from lithiumscope.resources.allocation.scheduler import Task,run_tasks
from lithiumscope.resources.manager import load_policy

RUNNERS={
    "model_1":"lithiumscope.model_1.training.execution.cv_runner:run_nested_cv",
    "model_2":"lithiumscope.model_2.training.execution.cv_runner:run_classification_cv",
}


def enabled(device,policy):
    # Backend selection remains unchanged: no concurrent GPU or MPS fits.
    return policy.max_parallel_tasks>1 and not (
        device.is_gpu or device.any_cuda_backend or device.mps_available)


def _publish(path,value):
    temporary=path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(value,temporary,compress=1)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def evaluate_job(input_path,expected_sha,output_path,model_group):
    """Only inputs created in this trusted local run may be deserialized."""
    from importlib import import_module
    source,target=Path(input_path),Path(output_path)
    if source.parent!=target.parent or target.exists() or file_sha256(source)!=expected_sha:
        raise ValueError("Competition worker staging contract mismatch")
    args=joblib.load(source)
    module,name=RUNNERS[model_group].split(":")
    result=getattr(import_module(module),name)(**args)
    return {"result_sha256":_publish(target,result)}


def schedule_candidates(model_group,requests,root,device,*,policy=None):
    """Parent performs ranking and evidence publication; workers perform CV only."""
    policy=policy or load_policy()
    if not enabled(device,policy):
        return None
    root=Path(root)/f"parallel_{uuid4().hex}"
    root.mkdir(parents=True,exist_ok=False)
    tasks=[]
    files={}
    keys=[key for key,_,_ in requests]
    if not keys or len(set(keys))!=len(keys) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",key) for key in keys):
        raise ValueError("Unique safe candidate identifiers required")
    try:
        for key,arguments,memory in requests:
            source=root/f"{key}.input.joblib"
            target=root/f"{key}.result.joblib"
            files[key]=(source,target)
            task=Task(key,"lithiumscope.resources.allocation.competition_workers:evaluate_job",
                      {"input_path":str(source),"output_path":str(target),"expected_sha":_publish(source,arguments),
                       "model_group":model_group},memory)
            tasks.append(task)
        report=run_tasks(tasks,root/"execution",policy=policy,execute=True)
        records={r["task_id"]:r for wave in report["waves"] for r in wave["results"]}
        output={}
        for key,(_,target) in files.items():
            record=records.get(key,{})
            try:
                if record.get("state")!="completed" or file_sha256(target)!=record["result"]["result_sha256"]:
                    raise ValueError("Missing or invalid worker result")
                output[key]=joblib.load(target)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                output[key]=RuntimeError(f"Scheduled candidate {key} failed: {record.get('error_type',type(exc).__name__)}")
        return output
    finally:
        # Keep scheduler manifest and the canonical CV checkpoints, not serialized matrices.
        for source,target in files.values():
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)


def candidate_result(results,key):
    value=results[key]
    if isinstance(value,Exception):
        raise value
    return value
