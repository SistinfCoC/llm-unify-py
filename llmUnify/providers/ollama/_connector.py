# Based on the documentation at https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

import json
from typing import Iterator

import requests

from ..._abstract_connector import LlmConnector
from ..._options import LlmOptions
from ..._response import LlmResponse
from ...utils._get_env import get_required_env
from ._options import OllamaOptions
from ._response import OllamaResponse

OLLAMA_HOST = "LLM_UNIFY_OLLAMA_HOST"


class OllamaConnector(LlmConnector):
    def __init__(self, host: str = None):
        self.host = host or get_required_env(OLLAMA_HOST)

    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        ollama_options = OllamaOptions.from_generic(options)

        response = requests.post(
            self.host.rstrip("/") + "/api/chat",
            json={**ollama_options.model_dump(exclude_none=True), "model": model_name, "stream": False},
        )
        response.raise_for_status()
        return OllamaResponse(**response.json()).to_generic()

    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        ollama_options = OllamaOptions.from_generic(options)

        response = requests.post(
            self.host.rstrip("/") + "/api/chat",
            json={**ollama_options.model_dump(exclude_none=True), "model": model_name, "stream": True},
            stream=True,
        )
        response.raise_for_status()
        for chunk in response.iter_lines():
            response_data = chunk.decode("utf-8")
            json_data = json.loads(response_data)
            yield OllamaResponse(**json_data).to_generic()
