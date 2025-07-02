# Based on the documentation at https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/generateContent#request-body

from typing import Any, Literal, Optional

from pydantic import BaseModel

from ..._options import LlmOptions


class GoogleBlob(BaseModel):
    mime_type: str
    data: str


class GoogleFileData(BaseModel):
    mime_type: str
    file_uri: str


class GoogleFunctionCall(BaseModel):
    name: str
    args: dict[str, Any]  # https://protobuf.dev/reference/protobuf/google.protobuf/#struct


class GoogleFunctionResponse(BaseModel):
    name: str
    response: dict[str, Any]  # https://protobuf.dev/reference/protobuf/google.protobuf/#struct


class GoogleVideoMetadata(BaseModel):
    start_offset: Optional[str] = None
    end_offset: Optional[str] = None


class GooglePart(BaseModel):
    text: Optional[str] = None
    inline_data: Optional[GoogleBlob] = None
    file_data: Optional[GoogleFileData] = None
    function_call: Optional[GoogleFunctionCall] = None
    function_response: Optional[GoogleFunctionResponse] = None
    video_metadata: Optional[GoogleVideoMetadata] = None


class GoogleContent(BaseModel):
    role: Optional[Literal["user", "model"]] = None
    parts: Optional[list[GooglePart]] = None


class GoogleAutoRoutingMode(BaseModel):
    model_routing_preference: Literal[
        "UNKNOWN",
        "PRIORITIZE_QUALITY",
        "BALANCED",
        "PRIORITIZE_COST",
    ]


class GoogleManualRoutingMode(BaseModel):
    model_name: str


class GoogleRoutingConfig(BaseModel):
    auto_mode: Optional[GoogleAutoRoutingMode]
    manual_mode: Optional[GoogleManualRoutingMode]


class GoogleGenerationConfig(BaseModel):
    stop_sequences: Optional[list[str]] = None
    response_mime_type: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    max_output_tokens: Optional[int] = None
    response_logprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    response_schema: Optional[dict[str, Any]] = None  # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Schema
    routing_config: Optional[GoogleRoutingConfig] = None


class GoogleSafetySetting(BaseModel):
    category: Literal[
        "HARM_CATEGORY_UNSPECIFIED",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    threshold: Literal[
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE",
        "OFF",
    ]
    method: Optional[
        Literal[
            "HARM_BLOCK_METHOD_UNSPECIFIED",
            "SEVERITY",
            "PROBABILITY",
        ]
    ] = None


class GoogleFunctionDeclaration(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None  # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Schema


class GoogleDynamicRetrievalConfig(BaseModel):
    mode: Literal["MODE_UNSPECIFIED", "MODE_DYNAMIC"]
    dynamic_threshold: Optional[float]


class GoogleSearchRetrieval(BaseModel):
    dynamic_retrieval_config: GoogleDynamicRetrievalConfig


class GoogleTool(BaseModel):
    function_declarations: Optional[list[GoogleFunctionDeclaration]] = None
    google_search_retrieval: Optional[GoogleSearchRetrieval] = None

    # Deprecated
    retrieval: Optional[Any] = None  # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/Tool#Retrieval


class GoogleFunctionCallingConfig(BaseModel):
    mode: Optional[Literal["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE"]] = None
    allowed_function_names: Optional[list[str]]


class GoogleToolConfig(BaseModel):
    function_calling_config: Optional[GoogleFunctionCallingConfig]


class GoogleOptions(BaseModel):
    contents: Optional[list[GoogleContent]] = None
    generation_config: Optional[GoogleGenerationConfig] = None
    safety_settings: Optional[GoogleSafetySetting] = None
    tools: Optional[list[GoogleTool]] = None
    tool_config: Optional[GoogleToolConfig] = None
    labels: Optional[dict[str, str]] = None

    @classmethod
    def from_generic(cls, generic_options: LlmOptions) -> "GoogleOptions":
        """Creates an GoogleOptions instance from a generic LlmOptions."""

        mapping_obj = {
            "contents": [{"role": "user", "parts": [{"text": generic_options.prompt}]}],
            **(generic_options.google.model_dump() if generic_options.google else {}),
            "generation_config": {
                "max_output_tokens": generic_options.max_tokens,
                "temperature": generic_options.temperature,
                "top_p": generic_options.top_p,
                "stop_sequences": generic_options.stop_sequences,
                **(
                    generic_options.google.generation_config.model_dump()
                    if generic_options.google and generic_options.google.generation_config
                    else {}
                ),
            },
        }

        return cls(**mapping_obj)
