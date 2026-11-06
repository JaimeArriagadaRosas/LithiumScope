from lithiumscope.core.resources import cpu_worker_budget


def test_cpu_worker_budget_is_positive():
    assert cpu_worker_budget() >= 1
