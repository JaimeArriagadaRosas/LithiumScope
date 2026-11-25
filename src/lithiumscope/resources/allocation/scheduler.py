"""Adaptive independent-process scheduling; no nested outer parallelism."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib import import_module
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from lithiumscope.resources.allocation.policy import Workload, plan_workload
from lithiumscope.resources.detection.host import detect_host


@dataclass(frozen=True)
class Task:
    task_id: str
    callable_path: str
    arguments: dict
    memory_bytes: int
    require_gpu: bool = False
    vram_bytes: int | None = None

    def __post_init__(self):
        if (not re.fullmatch(r"[A-Za-z0-9_-]+",self.task_id)
                or not self.callable_path.startswith("lithiumscope.")
                or ":" not in self.callable_path or self.memory_bytes<1):
            raise ValueError("Invalid task contract")


def wave_plan(tasks,host,policy):
    """Conservative homogeneous wave; remeasure host before the next wave."""
    if not tasks:
        raise ValueError("No tasks")
    first=tasks[0]
    candidates=[first]
    if not first.require_gpu:
        for task in tasks[1:]:
            if task.require_gpu:
                break
            candidates.append(task)
            if len(candidates)>=policy.max_parallel_tasks:
                break
    # A conservative maximum estimate is charged to every concurrent worker.
    while len(candidates)>1:
        try:
            plan=plan_workload(host,policy,Workload(tasks=len(candidates),
                    memory_per_task_bytes=max(t.memory_bytes for t in candidates)))
            if plan.parallel_tasks>=len(candidates):
                return candidates,plan
        except ValueError:
            pass
        candidates.pop()
    plan=plan_workload(host,policy,Workload(tasks=1,memory_per_task_bytes=first.memory_bytes,
                        require_gpu=first.require_gpu,vram_per_task_bytes=first.vram_bytes))
    return candidates,plan


def _atomic_json(path,value):
    tmp=path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _execute(task,plan,parent_session):
    from lithiumscope.core.logging.session import finalize_logging,get_logger
    from lithiumscope.core.logging.context import scientific_log_context
    from lithiumscope.resources.manager import worker_plan_scope
    finalize_logging()
    os.environ["LITHIUMSCOPE_WORKER_SESSION_DIR"]=str(parent_session)
    os.environ["LITHIUMSCOPE_SCHEDULER_WORKER"]="1"
    start=time.monotonic()
    with scientific_log_context(stage="scheduled_task",run_id=task.task_id):
        log=get_logger("resources.scheduler")
        try:
            module,name=task.callable_path.split(":",1)
            with worker_plan_scope(plan):
                result=getattr(import_module(module),name)(**task.arguments)
            log.info("TASK_COMPLETE %s",task.task_id)
            return {"task_id":task.task_id,"state":"completed","result":result,
                    "duration_seconds":time.monotonic()-start,"pid":os.getpid()}
        except Exception as exc:
            log.exception("TASK_FAILED %s",task.task_id)
            return {"task_id":task.task_id,"state":"failed","error_type":type(exc).__name__,
                    "duration_seconds":time.monotonic()-start,"pid":os.getpid()}
        finally:
            finalize_logging()


def run_tasks(tasks,output:Path,*,policy=None,execute=False):
    """Only explicit execution spawns workers; default creates a preflight plan."""
    if os.environ.get("LITHIUMSCOPE_SCHEDULER_WORKER"):
        raise RuntimeError("Nested outer scheduling is forbidden")
    from lithiumscope.resources.manager import load_policy
    from lithiumscope.core.logging.session import current_session_directory
    tasks=list(tasks)
    if not tasks or len({t.task_id for t in tasks})!=len(tasks):
        raise ValueError("Unique task IDs required")
    policy=policy or load_policy()
    output.mkdir(parents=True,exist_ok=False)
    manifest={"state":"preflight","tasks":[asdict(t) for t in tasks],"waves":[],
              "ram_vram_hard_cap":False,"nested_parallelism":False}
    path=output/"scheduler.json"
    _atomic_json(path,manifest)
    if not execute:
        return manifest
    parent=current_session_directory()
    pending=tasks.copy()
    manifest["state"]="running"
    try:
        while pending:
            host=detect_host()
            wave,plan=wave_plan(pending,host,policy)
            entry={"tasks":[t.task_id for t in wave],"plan":asdict(plan),"results":[]}
            manifest["waves"].append(entry)
            _atomic_json(path,manifest)
            # Spawn is portable and avoids inheriting active native estimator pools.
            with ProcessPoolExecutor(max_workers=len(wave),mp_context=mp.get_context("spawn")) as pool:
                futures=[pool.submit(_execute,task,plan,parent) for task in wave]
                for task,future in zip(wave,futures):
                    try:
                        record=future.result()
                    except Exception as exc:
                        record={"task_id":task.task_id,"state":"failed","error_type":type(exc).__name__}
                    entry["results"].append(record)
                    _atomic_json(path,manifest)
            del pending[:len(wave)]
        manifest["state"]="completed" if all(r["state"]=="completed" for w in manifest["waves"] for r in w["results"]) else "partial"
    except BaseException:
        manifest["state"]="interrupted"
        raise
    finally:
        _atomic_json(path,manifest)
    return manifest
