"""Application operations share one logging session; no parallel legacy log roots."""
import inspect
from lithiumscope.runtime import preboot_save_report
from lithiumscope.cli import scientific_lab_command, model_release_command

def test_operation_entrypoints_use_session_scoped_logging():
    assert 'current_session_directory() / "initialization"' in inspect.getsource(preboot_save_report._save_report)
    assert 'run_log("installation","versioned_model")' in inspect.getsource(model_release_command.run)
    assert 'run_log("eda",label)' in inspect.getsource(scientific_lab_command._eda_menu)
    assert 'run_log("simulations","registered")' in inspect.getsource(scientific_lab_command._run_simulations)
    assert 'LITHIUMSCOPE_SESSION_DIR' in inspect.getsource(scientific_lab_command._run_tests)
