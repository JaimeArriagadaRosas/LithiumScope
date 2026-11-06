from lithiumscope.core.run_resume import _scientific_code_files


def test_model_1_scientific_fingerprint_excludes_logging_and_plots():
    paths = {path.as_posix() for path in _scientific_code_files("model_1")}
    assert not any(path.endswith("core/logger.py") for path in paths)
    assert not any(path.endswith("evaluation/plots.py") for path in paths)
    assert any(path.endswith("model_1/training/cv_runner.py") for path in paths)
