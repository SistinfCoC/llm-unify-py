# See "Prerequisites" in the Azure AI Inference Python guide for how to configure
# the required endpoint URL, api_key, and api_version when initializing this connector:
# https://github.com/MicrosoftDocs/azure-docs-sdk-python/blob/main/docs-ref-services/preview/ai-inference-readme.md#prerequisites
from typing import Iterator

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from ..._abstract_connector import LlmConnector
from ..._options import LlmOptions
from ..._response import LlmResponse
from ...utils._get_env import get_env, get_required_env
from ._options import AzureOptions
from ._response import AzureResponse

AZURE_BASE_URL = "LLM_UNIFY_AZURE_BASE_URL"
AZURE_API_KEY = "LLM_UNIFY_AZURE_API_KEY"
AZURE_API_VERSION = "LLM_UNIFY_AZURE_API_VERSION"


class AzureConnector(LlmConnector):
    def __init__(self, base_url: str = None, api_key: str = None, api_version: str = None):
        self.base_url = base_url or get_required_env(AZURE_BASE_URL)
        self.api_key = api_key or get_required_env(AZURE_API_KEY)
        self.api_version = api_version or get_env(AZURE_API_VERSION)

        kwargs = {
            "endpoint": self.base_url,
            "credential": AzureKeyCredential(self.api_key),
        }
        if self.api_version:
            kwargs["api_version"] = self.api_version

        self.client = ChatCompletionsClient(**kwargs)

    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        azure_options = AzureOptions.from_generic(options)

        response = self.client.complete(model=model_name, **azure_options.model_dump(exclude_none=True))

        return AzureResponse(**response).to_generic()

    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        azure_options = AzureOptions.from_generic(options)

        response = self.client.complete(model=model_name, stream=True, **azure_options.model_dump(exclude_none=True))

        for chunk in response:
            yield AzureResponse(**chunk).to_generic()

        self.client.close()
