from lithiumscope.model_1.training.optimizer import _sqlite_storage


def test_optuna_storage_uses_sqlite_url(tmp_path):
    path = tmp_path / "checkpoints" / "study.db"
    url = _sqlite_storage(path)

    assert url.startswith("sqlite:///")
    assert url.endswith("study.db")
    assert path.parent.exists()
