import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor
from lithiumscope.core.scientific.learning_curves import build_curve_plan,evaluate_curve


def cohort():
    groups=np.repeat(np.arange(12),4)
    y=np.arange(len(groups),dtype=float)
    splits=list(GroupKFold(3).split(y,y,groups))
    return y,groups,splits


def test_nested_group_curve_never_changes_test_and_respects_qa():
    y,g,splits=cohort()
    qa=np.ones(len(y),dtype=bool);qa[::5]=False
    cuts=[float(np.quantile(y[t],.75)) for t,_ in splits]
    plan=build_curve_plan(y,g,splits,kind="classification",eligible=qa,endpoints=cuts)
    assert len(plan)==30
    for fold in range(1,4):
        for seed in (42,43,44):
            rows=[p for p in plan if p.fold==fold and p.seed==seed]
            for prev,p in zip(rows,rows[1:]):
                assert set(prev.train)<=set(p.train)
            for p in rows:
                assert np.array_equal(p.test,splits[fold-1][1])
                assert qa[p.train].all()
                assert not set(g[p.train])&set(g[p.test])
                assert p.endpoint_ppm==cuts[fold-1]
    assert sum(p.fraction==1 for p in plan)==3


def test_curve_records_train_and_test_and_failed_rows():
    y,g,splits=cohort()
    x=pd.DataFrame({"x":y})
    plan=build_curve_plan(y,g,splits,kind="regression",fractions=(.5,1.),seeds=(42,))
    allowed={p.fold:set(p.train) for p in plan if p.fraction==1}
    def fit(fold,a,b,groups):
        assert set(a.index)<=allowed[fold]
        return DummyRegressor().fit(a,b)
    table=evaluate_curve(x,y,g,plan,kind="regression",algorithm="dummy",fit=fit,predict=lambda m,x:m.predict(x))
    assert len(table)==6 and set(table.status)=={"ok"}
    assert table.train_rmse.notna().all() and table.test_rmse.notna().all()
    def failed(*args):
        raise RuntimeError("diagnostic unavailable")
    table=evaluate_curve(x,y,g,plan,kind="regression",algorithm="failed",fit=failed,predict=lambda m,x:m.predict(x))
    assert set(table.status)=={"failed"}
