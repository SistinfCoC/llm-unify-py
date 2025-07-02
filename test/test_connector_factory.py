import importlib

import pytest

from llmUnify._connector_factory import LlmOptions, LlmResponse, LlmUnify


@pytest.fixture
def mock_options():
    return LlmOptions(prompt="Test prompt")


def test_get_connector(monkeypatch):
    monkeypatch.setattr(LlmUnify, "_get_providers", lambda: {"provider"})

    mock_class = type("providerConnector", (), {"__init__": lambda self, **kwargs: None})
    mock_module = type("MockModule", (), {"ProviderConnector": mock_class})
    monkeypatch.setattr(importlib, "import_module", lambda name: mock_module)

    connector = LlmUnify.get_connector("provider", auth_key="abc123")
    assert isinstance(connector, mock_class)


def test_get_connector_invalid_provider(monkeypatch):
    monkeypatch.setattr(LlmUnify, "_get_providers", lambda: {})

    mock_class = type("providerConnector", (), {"__init__": lambda self, **kwargs: None})
    mock_module = type("MockModule", (), {"ProviderConnector": mock_class})
    monkeypatch.setattr(importlib, "import_module", lambda name: mock_module)

    with pytest.raises(ValueError):
        LlmUnify.get_connector("provider", auth_key="abc123")


def test_get_connector_import_error(monkeypatch):
    monkeypatch.setattr(LlmUnify, "_get_providers", lambda: {"provider"})

    def raise_():
        raise ImportError()

    monkeypatch.setattr(importlib, "import_module", lambda name: raise_())

    with pytest.raises(ImportError):
        LlmUnify.get_connector("provider", auth_key="abc123")


def test_generate(mock_options, monkeypatch):
    class MockConnector:
        def generate(self, model_name, options, call_name):
            return LlmResponse(generated_text="Generated response")

    monkeypatch.setattr(LlmUnify, "get_connector", lambda provider, **kwargs: MockConnector())

    response = LlmUnify.generate("provider:model_name", mock_options)
    assert isinstance(response, LlmResponse)
    assert response.generated_text == "Generated response"


def test_generate_stream(mock_options, monkeypatch):
    class MockConnector:
        def generate_stream(self, model_name, options, call_name):
            yield LlmResponse(generated_text="Streamed response 1")
            yield LlmResponse(generated_text="Streamed response 2")

    monkeypatch.setattr(LlmUnify, "get_connector", lambda provider, **kwargs: MockConnector())

    responses = list(LlmUnify.generate_stream("provider:model_name", mock_options))
    assert len(responses) == 2
    assert all(isinstance(resp, LlmResponse) for resp in responses)
    assert responses[0].generated_text == "Streamed response 1"


def test_validate_model():
    provider, model_name = LlmUnify._validate_model("provider:model_name")
    assert provider == "provider"
    assert model_name == "model_name"

    with pytest.raises(ValueError, match="Invalid model string"):
        LlmUnify._validate_model("invalid_model_string")
