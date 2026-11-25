from lithiumscope.experiments.generalization.progress import ScreenProgress


def test_progress_counts_attempts_errors_and_preflight_skips(capsys):
    progress=ScreenProgress("model_1",["rf"],["a","b"],1,2)
    progress.start("Fold interior 1/2")
    progress.finish(result={"rmse":4.0,"train_rmse":3.0},metric="rmse")
    progress.start("Fold interior 2/2")
    progress.finish(error="failed")
    progress.skip(2,"blocked")
    for label in ("reference","selected"):
        progress.start(label)
        progress.finish(result={"rmse":4.0,"train_rmse":3.0},metric="rmse")
    assert (progress.total,progress.done,progress.ok,progress.failed,progress.skipped)==(6,6,3,1,2)
    assert "100.0 %" in capsys.readouterr().out
