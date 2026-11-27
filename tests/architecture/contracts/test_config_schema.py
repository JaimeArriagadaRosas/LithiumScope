import pytest

from lithiumscope.core.config import load_config
from lithiumscope.core.config_schema import ConfigurationSchemaError, validate_config


def test_repository_configs_use_current_schema():
    assert load_config("app")["schema_version"] == 1
    assert load_config("model_1")["schema_version"] == 1
    assert load_config("model_2")["schema_version"] == 1
    assert load_config("logging")["schema_version"] == 1


def test_old_config_schema_is_rejected():
    with pytest.raises(ConfigurationSchemaError):
        validate_config("model_1", {"schema_version": 0})
