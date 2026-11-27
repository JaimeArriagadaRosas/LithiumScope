from types import SimpleNamespace
import pytest
from lithiumscope.resources.allocation.optimization_parallel import OptimizationExecution, choose_execution, map_calls
from lithiumscope.resources.allocation.policy import ResourcePolicy

def test_default_and_pruning_do_not_parallelize():
    assert choose_execution(None,False).mode=="serial"
    assert choose_execution(SimpleNamespace(),True).mode=="serial"
    with pytest.raises(ValueError):
        ResourcePolicy(parallel_level="all")
    assert map_calls(lambda x:x+1,[((1,),0,1),((2,),0,2)],OptimizationExecution())==[2,3]
