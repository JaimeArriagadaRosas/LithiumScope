from lithiumscope.resources.allocation.scheduler import Task,wave_plan,run_tasks
from lithiumscope.resources.allocation.policy import ResourcePolicy
from lithiumscope.resources.detection.host import HostResources


def test_scheduler_limits_cpu_memory_and_preserves_task_order(tmp_path):
    tasks=[Task(str(i),"lithiumscope.resources.manager:cpu_worker_budget",{},2000) for i in range(5)]
    host=HostResources(8,4,8,16000,8000,100000,())
    policy=ResourcePolicy(max_cpu_fraction=.75,max_parallel_tasks=4,max_ram_fraction=.5)
    wave,plan=wave_plan(tasks,host,policy)
    assert [t.task_id for t in wave]==["0","1"]
    assert plan.parallel_tasks==2 and plan.threads_per_task==3
    manifest=run_tasks(tasks,tmp_path/"preflight",policy=policy)
    assert manifest["state"]=="preflight" and manifest["waves"]==[]
