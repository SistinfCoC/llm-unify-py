import pytest

from llmUnify.providers.watsonx._connector import EnvironmentVariableError, WatsonxConnector


def test_watson_connector_init_env_error(monkeypatch):
    monkeypatch.delenv("WATSONX_SPACE_ID", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

    with pytest.raises(EnvironmentVariableError):
        WatsonxConnector(host="test_host", api_key="test_api_key")
