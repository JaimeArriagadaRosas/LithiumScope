from lithiumscope.resources.monitoring.acceleration_benchmark import compare, benchmark_worker
import pytest

def test_acceleration_benchmark_is_preflight_without_permission(tmp_path):
    output=tmp_path/"benchmark"
    result=compare(output)
    assert result["state"]=="preflight"
    assert result["runs"]==[] and not (output/"fixture.npy").exists()

def test_benchmark_worker_rejects_external_network():
    with pytest.raises(ValueError,match="loopback"):
        benchmark_worker("https://example.org/fixture.npy","0"*64,42,"cpu")
