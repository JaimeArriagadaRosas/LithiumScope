from pathlib import Path

from lithiumscope.runtime.graceful_shutdown import GracefulShutdownManager
from lithiumscope.runtime.preboot import CORE_DEPENDENCIES, cleanup_gitkeep_placeholders


def test_core_dependency_registry_is_not_empty():
    assert "pandas" in CORE_DEPENDENCIES
    assert "scikit-learn" in CORE_DEPENDENCIES


def test_shutdown_manager_is_idempotent():
    manager = GracefulShutdownManager(child_timeout_seconds=0.01)
    manager.request_shutdown("test", 0)
    manager.request_shutdown("test_again", 0)
    assert manager.requested


def test_gitkeep_cleanup_contract_exists():
    assert callable(cleanup_gitkeep_placeholders)
