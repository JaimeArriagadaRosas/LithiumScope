import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.shared.design import budget_arms,choose_algorithms
from lithiumscope.experiments.model_2.studies.paired import representation_arms,policy_arms


def test_budget_is_factorial_and_references_same_family_and_seed():
    x=pd.DataFrame({"x":[1.,2.]})
    arms=budget_arms(["a","b"],x)
    assert len(arms)==24
    mapping={a.name:a for a in arms}
    assert sum(a.reference is None for a in arms)==6
    for a in arms:
        if a.reference:
            ref=mapping[a.reference]
            assert ref.algorithm==a.algorithm
            assert ref.policy["seed"]==a.policy["seed"]
    with pytest.raises(ValueError):
        choose_algorithms(["a"],["b"])


def test_m2_representation_has_exact_common_cohort_and_deduplicates():
    base={f"x{i}":[1.,2.] for i in range(114)}
    for i in range(6):
        base[f"nbr_{i}"]=[3.,4.]
        base[f"nir_swir22_nd_{i}"]=[3.,4.]
    base.update({f"rededge_{i}":[1.,2.] for i in range(44)})
    x=pd.DataFrame(base)
    keep=np.array([True,False])
    arms=representation_arms(x,["random_forest"],keep,seed=42,mode="legacy")
    assert [a.x.shape[1] for a in arms]==[170,126,164]
    assert all(np.array_equal(a.eligible,keep) for a in arms)
    with pytest.raises(ValueError,match="redundant"):
        representation_arms(x.assign(nir_swir22_nd_0=[0.,1.]),["random_forest"],keep,seed=42,mode="legacy")


def test_calibration_and_quality_are_changed_one_at_a_time():
    x=pd.DataFrame({"x":[1.,2.]})
    keep=np.array([True,False])
    ref,cal,qa=policy_arms(x,["random_forest"],keep,seed=42,mode="legacy")
    assert cal.reference==qa.reference==ref.name
    assert cal.policy["calibration"]=="spatial_sigmoid"
    assert np.array_equal(cal.eligible,ref.eligible)
    assert qa.policy["calibration"]==ref.policy["calibration"] and qa.eligible is None
