from lithiumscope.resources.monitoring.pipeline_benchmark import run

def test_benchmark_default_is_preflight(tmp_path):
    path=tmp_path/"bench"
    result=run(path,rows=60)
    assert result["state"]=="preflight"
    assert result["results"]==[]
    assert not (path/"synthetic_patches.npy").exists()
