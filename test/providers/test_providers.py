from pathlib import Path

import pytest

from llmUnify import LlmOptions, LlmResponse, LlmUnify

# To execute integration tests, make sure to fill in the fields with valid model and credentials
# for the respective providers before running the tests.
PROVIDERS_DATA = {
    "aws": (
        "model",
        {
            "region": "<region>",
            "access_key_id": "<access_key_id>",
            "secret_access_key": "<secret_access_key>",
        },
        {
            "LLM_UNIFY_AWS_REGION": "<region>",
            "LLM_UNIFY_AWS_ACCESS_KEY_ID": "<access_key_id>",
            "LLM_UNIFY_AWS_SECRET_ACCESS_KEY": "<secret_access_key>",
        },
    ),
    "azure": (
        "model",
        {
            "base_url": "<base_url>",
            "api_key": "<api_key>",
            "api_version": "<api_version>",
        },
        {
            "LLM_UNIFY_AZURE_BASE_URL": "<regbase_urlion>",
            "LLM_UNIFY_AZURE_API_KEY": "<api_key>",
            "LLM_UNIFY_AZURE_API_VERSION": "<api_version>",
        },
    ),
    "google": (
        "model",
        {
            "region": "<region>",
            "project_id": "<project_id>",
            "application_credentials": "<application_credentials>",
        },
        {
            "LLM_UNIFY_GOOGLE_REGION": "<region>",
            "LLM_UNIFY_GOOGLE_PROJECT_ID": "<project_id>",
            "LLM_UNIFY_GOOGLE_APPLICATION_CREDENTIALS": "<application_credentials>",
        },
    ),
    "ollama": (
        "model",
        {"host": "<host>"},
        {"LLM_UNIFY_OLLAMA_HOST": "<host>"},
    ),
    "watsonx": (
        "model",
        {
            "host": "<host>",
            "project_id": "<project_id>",
            "space_id": "<space_id>",
            "api_key": "<api_key>",
        },
        {
            "LLM_UNIFY_WATSONX_HOST": "<host>",
            "LLM_UNIFY_WATSONX_SPACE_ID": "<project_id>",
            "LLM_UNIFY_WATSONX_PROJECT_ID": "<space_id>",
            "LLM_UNIFY_WATSONX_API_KEY": "<api_key>",
        },
    ),
}


@pytest.mark.parametrize(
    argnames=("provider"),
    argvalues=[d.name for d in Path("llmUnify/providers").iterdir() if d.is_dir() and d.name != "__pycache__"],
)
def test_provider(provider, monkeypatch):
    options = LlmOptions(
        prompt="Write a greeting.",
        max_tokens=20,
        temperature=0.5,
        stop_sequences=["\\n"],
        top_p=1,
    )

    assert provider in PROVIDERS_DATA.keys()

    model_name, params, env = PROVIDERS_DATA[provider]

    provider_model_str = f"{provider}:{model_name}"

    simple_generate_params(provider_model_str, options, params)
    simple_generate_stream_params(provider_model_str, options, params)

    connector_generate_params(provider, model_name, options, params)
    connector_generate_stream_params(provider, model_name, options, params)

    for key, value in env.items():
        if value:
            monkeypatch.setenv(key, value)

    simple_generate_env(provider_model_str, options)
    simple_generate_stream_env(provider_model_str, options)

    connector_generate_env(provider, model_name, options)
    connector_generate_stream_env(provider, model_name, options)

    for key, value in env.items():
        if value:
            monkeypatch.delenv(key, raising=False)


def simple_generate_params(provider_model_str, options, params):
    response = LlmUnify.generate(provider_model_str, options, **params)
    assert isinstance(response, LlmResponse)


def simple_generate_stream_params(provider_model_str, options, params):
    response = LlmUnify.generate_stream(provider_model_str, options, **params)
    assert all(isinstance(resp, LlmResponse) for resp in response)


def connector_generate_params(provider, model, options, params):
    connector = LlmUnify.get_connector(provider, **params)
    response = connector.generate(model, options)
    assert isinstance(response, LlmResponse)


def connector_generate_stream_params(provider, model, options, params):
    connector = LlmUnify.get_connector(provider, **params)
    response = connector.generate_stream(model, options)
    assert all(isinstance(resp, LlmResponse) for resp in response)


def simple_generate_env(provider_model_str, options):
    response = LlmUnify.generate(provider_model_str, options)
    assert isinstance(response, LlmResponse)


def simple_generate_stream_env(provider_model_str, options):
    response = LlmUnify.generate_stream(provider_model_str, options)
    assert all(isinstance(resp, LlmResponse) for resp in response)


def connector_generate_env(provider, model, options):
    connector = LlmUnify.get_connector(provider)
    response = connector.generate(model, options)
    assert isinstance(response, LlmResponse)


def connector_generate_stream_env(provider, model, options):
    connector = LlmUnify.get_connector(provider)
    response = connector.generate_stream(model, options)
    assert all(isinstance(resp, LlmResponse) for resp in response)
