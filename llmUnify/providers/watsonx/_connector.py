# Based on the documentation at https://cloud.ibm.com/apidocs/watsonx-ai#text-generation

from typing import Iterator

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from ..._abstract_connector import LlmConnector
from ..._options import LlmOptions
from ..._response import LlmResponse
from ...utils._get_env import EnvironmentVariableError, get_env, get_required_env
from ._options import WatsonxOptions
from ._response import WatsonxResponse

WATSONX_HOST = "LLM_UNIFY_WATSONX_HOST"
WATSONX_SPACE_ID = "LLM_UNIFY_WATSONX_SPACE_ID"
WATSONX_PROJECT_ID = "LLM_UNIFY_WATSONX_PROJECT_ID"
WATSONX_API_KEY = "LLM_UNIFY_WATSONX_API_KEY"


class WatsonxConnector(LlmConnector):
    def __init__(self, host: str = None, space_id: str = None, project_id: str = None, api_key: str = None):
        self.host = host or get_required_env(WATSONX_HOST)
        self.space_id = space_id or get_env(WATSONX_SPACE_ID)
        self.project_id = project_id or get_env(WATSONX_PROJECT_ID)
        self.api_key = api_key or get_required_env(WATSONX_API_KEY)

        if not (self.space_id or self.project_id):
            raise EnvironmentVariableError(
                f"The environment variables '{WATSONX_SPACE_ID}' and '{WATSONX_PROJECT_ID}' are not set."
                " At least one is required."
            )

    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        watsonx_options = WatsonxOptions.from_generic(options)

        model_inference = ModelInference(
            model_id=model_name,
            credentials=Credentials(
                api_key=self.api_key,
                url=self.host,
            ),
            project_id=self.project_id,
            space_id=self.space_id,
        )

        response = model_inference.generate(
            prompt=watsonx_options.input, params=watsonx_options.parameters.model_dump(exclude_none=True)
        )
        return WatsonxResponse(**response).to_generic()

    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        watsonx_options = WatsonxOptions.from_generic(options)

        model_inference = ModelInference(
            model_id=model_name,
            credentials=Credentials(
                api_key=self.api_key,
                url=self.host,
            ),
            project_id=self.project_id,
            space_id=self.space_id,
        )

        response = model_inference.generate_text_stream(
            prompt=watsonx_options.input,
            params=watsonx_options.parameters.model_dump(exclude_none=True),
            raw_response=True,
        )

        for chunk in response:
            yield WatsonxResponse(**chunk).to_generic()
