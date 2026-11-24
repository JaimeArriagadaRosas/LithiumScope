"""Backward-compatible API; logging implementation is in core/logging/."""
from lithiumscope.core.logging.session import (
    configure_logging, current_error_log_path, current_session_log_path,
    current_session_directory, current_session_id, error_count, finalize_logging,
    flush_logging, get_logger, install_warning_capture, logging_summary,
    run_log, warning_summary,
)
from lithiumscope.core.logging.retention import cleanup_expired_logs

__all__ = ["configure_logging", "current_error_log_path", "current_session_log_path",
           "current_session_directory", "current_session_id", "error_count", "finalize_logging",
           "flush_logging", "get_logger", "install_warning_capture", "logging_summary",
           "run_log", "warning_summary", "cleanup_expired_logs"]
