import pytest
from lithiumscope.evaluation.holdout import registry, workflow


def test_holdout_phase_rejects_cross_model_dispatch(monkeypatch,tmp_path):
    monkeypatch.setattr(registry,"verified_registration",lambda root:({"model_group":"model_2"},None))
    with pytest.raises(ValueError,match="another model"):
        workflow.run_holdout_phase("model_1","score",tmp_path,trusted_local_bundles=True)


def test_registration_and_score_never_receive_labels(monkeypatch,tmp_path):
    monkeypatch.setattr(registry,"verified_registration",lambda root:({"model_group":"model_1"},None))
    for phase in ("register","score"):
        with pytest.raises(ValueError,match="label"):
            workflow.run_holdout_phase("model_1",phase,tmp_path,labels_path=tmp_path/"labels.csv")
