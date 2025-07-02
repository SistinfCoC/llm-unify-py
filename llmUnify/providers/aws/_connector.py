from typing import Iterator

import boto3

from ..._abstract_connector import LlmConnector
from ..._options import LlmOptions
from ..._response import LlmResponse
from ...utils._get_env import get_required_env
from ._options import AwsOptions
from ._response import AwsResponse

AWS_REGION = "LLM_UNIFY_AWS_REGION"
AWS_ACCESS_KEY_ID = "LLM_UNIFY_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "LLM_UNIFY_AWS_SECRET_ACCESS_KEY"


class AwsConnector(LlmConnector):
    def __init__(self, region: str = None, access_key_id: str = None, secret_access_key: str = None):
        self.region = region or get_required_env(AWS_REGION)
        self.access_key_id = access_key_id or get_required_env(AWS_ACCESS_KEY_ID)
        self.secret_access_key = secret_access_key or get_required_env(AWS_SECRET_ACCESS_KEY)

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        aws_options = AwsOptions.from_generic(options)

        response = self.client.converse(modelId=model_name, **aws_options.model_dump(exclude_none=True))
        return AwsResponse(**response).to_generic()

    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        aws_options = AwsOptions.from_generic(options)

        response = self.client.converse_stream(modelId=model_name, **aws_options.model_dump(exclude_none=True))

        for chunk in response["stream"]:
            yield AwsResponse(**self._normalize_stream_response(chunk)).to_generic()

    def _normalize_stream_response(self, chunk: dict) -> dict:
        return {
            "output": (
                {"message": {"role": "assistant", "content": [{"text": chunk["contentBlockDelta"]["delta"]["text"]}]}}
                if "contentBlockDelta" in chunk
                else None
            ),
            "stopReason": chunk.get("messageStop", {}).get("stopReason", None),
            "additionalModelResponseFields": chunk.get("messageStop", {}).get("additionalModelResponseFields", None),
            **chunk.get("metadata", {}),
        }
