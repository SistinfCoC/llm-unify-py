# Based on the documentation at https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/generateContent

import os
from typing import Iterator

import vertexai
from vertexai.generative_models import GenerativeModel

from ..._abstract_connector import LlmConnector
from ..._options import LlmOptions
from ..._response import LlmResponse
from ...utils._get_env import get_env, get_required_env
from ._options import GoogleOptions
from ._response import GoogleResponse

GOOGLE_REGION = "LLM_UNIFY_GOOGLE_REGION"
GOOGLE_PROJECT_ID = "LLM_UNIFY_GOOGLE_PROJECT_ID"
GOOGLE_APPLICATION_CREDENTIALS = "LLM_UNIFY_GOOGLE_APPLICATION_CREDENTIALS"


class GoogleConnector(LlmConnector):
    def __init__(self, region: str = None, project_id: str = None, application_credentials: str = None):
        self.region = region or get_required_env(GOOGLE_REGION)
        self.project_id = project_id or get_required_env(GOOGLE_PROJECT_ID)
        self.application_credentials = application_credentials or get_env(GOOGLE_APPLICATION_CREDENTIALS)

        if self.application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.application_credentials

        vertexai.init(project=self.project_id, location=self.region)

    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        google_options = GoogleOptions.from_generic(options)

        google_options_dict = google_options.model_dump(exclude_none=True)

        response = GenerativeModel(model_name=model_name).generate_content(
            contents=google_options_dict.get("contents"),
            generation_config=google_options_dict.get("generationConfig"),
            safety_settings=google_options_dict.get("safetySettings"),
            tools=google_options_dict.get("tools"),
            tool_config=google_options_dict.get("tool_config"),
            labels=google_options_dict.get("labels"),
            stream=False,
        )

        return GoogleResponse(**self._normalize_response(response)).to_generic()

    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        google_options = GoogleOptions.from_generic(options)

        google_options_dict = google_options.model_dump(exclude_none=True)

        response = GenerativeModel(model_name=model_name).generate_content(
            contents=google_options_dict.get("contents"),
            generation_config=google_options_dict.get("generationConfig"),
            safety_settings=google_options_dict.get("safetySettings"),
            tools=google_options_dict.get("tools"),
            tool_config=google_options_dict.get("tool_config"),
            labels=google_options_dict.get("labels"),
            stream=True,
        )

        for chunk in response:
            yield GoogleResponse(**self._normalize_response(chunk)).to_generic()

    def _normalize_response(self, response) -> dict:
        if isinstance(response, dict):
            return {key: self._normalize_response(value) for key, value in response.items()}
        elif isinstance(response, list):
            return [self._normalize_response(item) for item in response]
        elif hasattr(response, "__dict__"):
            return {key: self._normalize_response(value) for key, value in vars(response).items()}
        elif hasattr(response, "__iter__") and not isinstance(response, (str, bytes)):
            return [self._normalize_response(item) for item in response]
        else:
            return response
