import pytest

from llmUnify.utils._get_env import EnvironmentVariableError, get_env, get_required_env


def test_get_required_env_var_exists(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test_value")

    assert get_required_env("TEST_VAR") == "test_value"

    monkeypatch.delenv("TEST_VAR", raising=False)


def test_get_required_env_var_not_exists(monkeypatch):
    monkeypatch.delenv("TEST_VAR", raising=False)

    with pytest.raises(EnvironmentVariableError):
        get_required_env("TEST_VAR")


def test_get_env_var_exists(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test_value")

    assert get_env("TEST_VAR") == "test_value"

    monkeypatch.delenv("TEST_VAR", raising=False)


def test_get_env_var_not_exists(monkeypatch):
    monkeypatch.delenv("TEST_VAR", raising=False)

    assert get_env("TEST_VAR") is None
