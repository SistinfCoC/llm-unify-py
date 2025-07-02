# Based on the documentation at https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..._options import LlmOptions


class AwsBytesSource(BaseModel):
    bytes: bytes


class AwsS3Location(BaseModel):
    uri: str
    bucketOwner: Optional[str] = None


class AwsImageContent(BaseModel):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: AwsBytesSource


class AwsDocumentContent(BaseModel):
    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: AwsBytesSource


class AwsVideoSource(BaseModel):
    bytes: Optional["bytes"] = None
    s3Location: Optional[AwsS3Location] = None


class AwsVideoContent(BaseModel):
    format: Literal["mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp"]
    source: AwsVideoSource


class AwsToolUse(BaseModel):
    toolUseId: str
    name: str
    input: Union[dict[str, Any], list[Any], int, float, str, bool, None]


class AwsToolResultContent(BaseModel):
    json_: Optional[Union[dict[str, Any], list[Any], int, float, str, bool, None]] = Field(None, alias="json")
    text: Optional[str] = None
    image: Optional[AwsImageContent] = None
    document: Optional[AwsDocumentContent] = None
    video: Optional[AwsVideoContent] = None


class AwsToolResult(BaseModel):
    toolUseId: str
    content: list[AwsToolResultContent]
    status: Literal["success", "error"]


class AwsGuardText(BaseModel):
    text: str
    qualifiers: list[Literal["grounding_source", "query", "guard_content"]]


class AwsGuardImage(BaseModel):
    format: Literal["png", "jpeg"]
    source: AwsBytesSource


class AwsGuardContent(BaseModel):
    text: Optional[AwsGuardText] = None
    image: Optional[AwsGuardImage] = None


class AwsContentBlock(BaseModel):
    text: Optional[str] = None
    image: Optional[AwsImageContent] = None
    document: Optional[AwsDocumentContent] = None
    video: Optional[AwsVideoContent] = None
    toolUse: Optional[AwsToolUse] = None
    toolResult: Optional[AwsToolResult] = None
    guardContent: Optional[AwsGuardContent] = None


class AwsMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[AwsContentBlock]


class AwsSystem(BaseModel):
    guardContent: Optional[AwsGuardContent] = None


class AwsInferenceConfig(BaseModel):
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[list[str]] = None


class AwsToolSpecInputSchema(BaseModel):
    json_: Optional[Union[dict[str, Any], list[Any], int, float, str, bool, None]] = Field(None, alias="json")


class AwsToolSpec(BaseModel):
    name: str
    description: str
    inputSchema: Optional[AwsToolSpecInputSchema] = None


class AwsTool(BaseModel):
    toolSpec: AwsToolSpec


class AwsToolChoiceTool(BaseModel):
    name: Optional[str] = None


class AwsToolChoice(BaseModel):
    auto: Optional[dict[str, Any]] = None
    any: Optional[dict[str, Any]] = None
    tool: Optional[AwsToolChoiceTool] = None


class AwsToolConfig(BaseModel):
    tools: Optional[list[AwsTool]] = None
    toolChoice: Optional[AwsToolChoice] = None


class AwsGuardrailConfig(BaseModel):
    guardrailIdentifier: str
    guardrailVersion: str
    trace: Literal["enabled", "disabled"]


class AwsPromptVariable(BaseModel):
    text: str


class AwsPerformanceConfig(BaseModel):
    latency: Literal["standard", "optimized"]


class AwsOptions(BaseModel):
    messages: list[AwsMessage]
    system: Optional[AwsSystem] = None
    inferenceConfig: Optional[AwsInferenceConfig] = None
    toolConfig: Optional[AwsToolConfig] = None
    guardrailConfig: Optional[AwsGuardrailConfig] = None
    additionalModelRequestFields: Optional[Union[dict[str, Any], list[Any], int, float, str, bool, None]] = None
    promptVariables: Optional[dict[str, AwsPromptVariable]] = None
    additionalModelResponseFieldPaths: Optional[list[str]] = None
    requestMetadata: Optional[dict[str, str]] = None
    performanceConfig: Optional[dict[str, str]] = None

    @classmethod
    def from_generic(cls, generic_options: LlmOptions) -> "AwsOptions":
        """Creates an AwsOptions instance from a generic LlmOptions."""

        mapping_obj = {
            "messages": [{"role": "user", "content": [{"text": generic_options.prompt}]}],
            **(generic_options.aws.model_dump(by_alias=True) if generic_options.aws else {}),
            "inferenceConfig": {
                "maxTokens": generic_options.max_tokens,
                "temperature": generic_options.temperature,
                "topP": generic_options.top_p,
                "stopSequences": generic_options.stop_sequences,
                **(
                    generic_options.aws.inferenceConfig.model_dump(by_alias=True)
                    if generic_options.aws and generic_options.aws.inferenceConfig
                    else {}
                ),
            },
        }

        return cls(**mapping_obj)
