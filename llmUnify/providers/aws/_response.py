# Based on the documentation at https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..._response import LlmResponse


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


class AwsOutput(BaseModel):
    message: Optional[AwsMessage] = None


class AwsUsage(BaseModel):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class AwsMetrics(BaseModel):
    latencyMs: int


class AwsTopicPolicy(BaseModel):
    name: str
    type: Literal["DENY"]
    action: Literal["BLOCKED"]


class AwsContentPolicyFilter(BaseModel):
    type: Literal["INSULTS", "HATE", "SEXUAL", "VIOLENCE", "MISCONDUCT", "PROMPT_ATTACK"]
    confidence: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    filterStrength: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    action: Literal["BLOCKED"]


class AwsWordPolicyCustomWord(BaseModel):
    match: str
    action: Literal["BLOCKED"]


class AwsSensitiveInformationPolicyEntity(BaseModel):
    match: str
    type: Literal[
        "ADDRESS",
        "AGE",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "CA_HEALTH_NUMBER",
        "CA_SOCIAL_INSURANCE_NUMBER",
        "CREDIT_DEBIT_CARD_CVV",
        "CREDIT_DEBIT_CARD_EXPIRY",
        "CREDIT_DEBIT_CARD_NUMBER",
        "DRIVER_ID",
        "EMAIL",
        "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
        "IP_ADDRESS",
        "LICENSE_PLATE",
        "MAC_ADDRESS",
        "NAME",
        "PASSWORD",
        "PHONE",
        "PIN",
        "SWIFT_CODE",
        "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
        "UK_NATIONAL_INSURANCE_NUMBER",
        "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER",
        "URL",
        "USERNAME",
        "US_BANK_ACCOUNT_NUMBER",
        "US_BANK_ROUTING_NUMBER",
        "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
        "US_PASSPORT_NUMBER",
        "US_SOCIAL_SECURITY_NUMBER",
        "VEHICLE_IDENTIFICATION_NUMBER",
    ]
    action: Literal["ANONYMIZED", "BLOCKED"]


class AwsSensitiveInformationPolicyRegex(BaseModel):
    name: str
    match: str
    regex: str
    action: Literal["ANONYMIZED", "BLOCKED"]


class AwsContextualGroundingFilter(BaseModel):
    type: Literal["GROUNDING", "RELEVANCE"]
    threshold: float
    score: float
    action: Literal["BLOCKED", "NONE"]


class AwsInvocationMetrics(BaseModel):
    guardrailProcessingLatency: int
    usage: dict[str, int]
    guardrailCoverage: dict[str, dict[str, int]]


class AwsInputAssessment(BaseModel):
    topicPolicy: Optional[list[AwsTopicPolicy]] = None
    contentPolicy: Optional[list[AwsContentPolicyFilter]] = None
    wordPolicy: Optional[dict[str, list[Union[AwsWordPolicyCustomWord, dict[str, str]]]]] = None
    sensitiveInformationPolicy: Optional[
        dict[str, list[Union[AwsSensitiveInformationPolicyEntity, AwsSensitiveInformationPolicyRegex]]]
    ] = None
    contextualGroundingPolicy: Optional[list[AwsContextualGroundingFilter]] = None
    invocationMetrics: Optional[AwsInvocationMetrics] = None


class AwsPromptRouter(BaseModel):
    invokedModelId: Optional[str] = None


class AwsGuardrail(BaseModel):
    modelOutput: list[str]
    inputAssessment: dict[str, AwsInputAssessment]
    outputAssessments: dict[str, list[AwsInputAssessment]]
    promptRouter: Optional[AwsPromptRouter] = None


class AwsPerformanceConfig(BaseModel):
    latency: Literal["standard", "optimized"]


class AwsResponse(BaseModel):
    output: Optional[AwsOutput] = None
    stopReason: Optional[
        Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "guardrail_intervened", "content_filtered"]
    ] = None
    usage: Optional[AwsUsage] = None
    metrics: Optional[AwsMetrics] = None
    additionalModelResponseFields: Optional[Union[dict[str, Any], list[Any], int, float, str, bool, None]] = None
    trace: Optional[AwsGuardrail] = None
    performanceConfig: Optional[AwsPerformanceConfig] = None

    def to_generic(self) -> LlmResponse:
        """Converts an AwsResponse into a generic LlmResponse instance."""
        return LlmResponse(
            generated_text=(
                self.output.message.content[0].text
                if getattr(self.output, "message", None)
                and getattr(self.output.message, "content", None)
                and len(self.output.message.content) > 0
                else None
            ),
            model=getattr(getattr(self.trace, "promptRouter", None), "invokedModelId", None),
            generated_token_count=getattr(self.usage, "outputTokens", None),
            input_token_count=getattr(self.usage, "inputTokens", None),
            aws=self,
        )
