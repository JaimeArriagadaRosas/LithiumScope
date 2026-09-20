import warnings

import lithiumscope.core.logger as logger_module


def test_warning_capture_deduplicates_repeated_warning(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "LITHIUMSCOPE_LOG_DIR",
        str(tmp_path),
    )
    logger_module.finalize_logging()
    logger_module._WARNING_COUNTS.clear()
    logger_module.configure_logging()
    logger_module.install_warning_capture()

    for _ in range(5):
        warnings.warn(
            "repeated test warning",
            UserWarning,
        )

    summary = logger_module.warning_summary()
    logger_module.finalize_logging()

    assert summary["unique"] == 1
    assert summary["total"] == 5
    assert summary["repeated"] == 4
