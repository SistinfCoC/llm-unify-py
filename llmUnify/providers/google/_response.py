# Based on the documentation at https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse

from typing import Any, Literal, Optional

from pydantic import BaseModel

from ..._response import LlmResponse


class GoogleBlob(BaseModel):
    mime_type: str
    data: str


class GoogleFileData(BaseModel):
    mime_type: str
    fileUri: str


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


class GoogleCandidate(BaseModel):
    token: Optional[str] = None
    token_id: Optional[int] = None
    log_probability: Optional[float] = None


class GoogleTopCandidate(BaseModel):
    candidates: Optional[list[GoogleCandidate]] = None


class GoogleLogprobResult(BaseModel):
    top_candidates: Optional[list[GoogleTopCandidate]] = None
    chosen_candidates: Optional[list[GoogleCandidate]] = None


class GoogleSafetyRating(BaseModel):
    category: Optional[
        Literal[
            "HARM_CATEGORY_UNSPECIFIED",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
    ] = None
    probability: Optional[
        Literal[
            "HARM_PROBABILITY_UNSPECIFIED",
            "NEGLIGIBLE",
            "LOW",
            "MEDIUM",
            "HIGH",
        ]
    ] = None
    probability_score: Optional[float] = None
    severity: Optional[
        Literal[
            "HARM_SEVERITY_UNSPECIFIED",
            "HARM_SEVERITY_NEGLIGIBLE",
            "HARM_SEVERITY_LOW",
            "HARM_SEVERITY_MEDIUM",
            "HARM_SEVERITY_HIGH",
        ]
    ] = None
    severity_score: Optional[float] = None
    blocked: Optional[bool] = None


class GoogleDate(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


class GoogleCitation(BaseModel):
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    uri: Optional[str] = None
    title: Optional[str] = None
    license: Optional[str] = None
    publication_date: Optional[GoogleDate] = None


class GoogleCitationMetadata(BaseModel):
    citations: Optional[list[GoogleCitation]] = None


class GoogleWeb(BaseModel):
    uri: Optional[str] = None
    title: Optional[str] = None


class GoogleRetrievedContext(BaseModel):
    uri: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None


class GoogleGroundingChunk(BaseModel):
    web: Optional[GoogleWeb] = None
    retrieved_context: Optional[GoogleRetrievedContext] = None


class GoogleSegment(BaseModel):
    part_index: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    text: Optional[str] = None


class GoogleGroundingSupport(BaseModel):
    grounding_chunk_indices: Optional[list[int]] = None
    confidence_scores: Optional[list[float]] = None
    segment: Optional[GoogleSegment] = None


class GoogleSearchEntryPoint(BaseModel):
    rendered_content: Optional[str] = None
    sdk_blob: Optional[str] = None


class GoogleRetrievalMetadata(BaseModel):
    googleSearchDynamicRetrievalScore: Optional[float] = None


class GoogleGroundingMetadata(BaseModel):
    web_search_queries: Optional[list[str]] = None
    grounding_chunks: Optional[list[GoogleGroundingChunk]] = None
    grounding_supports: Optional[list[GoogleGroundingSupport]] = None
    search_entry_point: Optional[GoogleSearchEntryPoint] = None
    retrieval_metadata: Optional[GoogleRetrievalMetadata] = None


class GoogleCandidate(BaseModel):
    index: Optional[int] = None
    content: Optional[GoogleContent] = None
    avg_logprobs: Optional[float] = None
    logprobs_result: Optional[GoogleLogprobResult] = None
    finish_reason: Optional[
        Literal[
            "FINISH_REASON_UNSPECIFIED",
            "STOP",
            "MAX_TOKENS",
            "SAFETY",
            "RECITATION",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
        ]
    ]
    safety_ratings: Optional[list[GoogleSafetyRating]] = None
    citation_metadata: Optional[GoogleCitationMetadata] = None
    grounding_metadata: Optional[GoogleGroundingMetadata] = None
    finish_message: Optional[str] = None


class GooglePromptFeedback(BaseModel):
    block_reason: Optional[
        Literal[
            "BLOCKED_REASON_UNSPECIFIED",
            "SAFETY",
            "OTHER",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
        ]
    ]
    safety_ratings: Optional[list[GoogleSafetyRating]] = None
    block_reason_message: Optional[str] = None


class GoogleUsageMetadata(BaseModel):
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    total_token_count: Optional[int] = None
    cached_content_token_count: Optional[int] = None


class GoogleResponse(BaseModel):
    candidates: Optional[list[GoogleCandidate]] = None
    model_version: Optional[str] = None
    prompt_feedback: Optional[GooglePromptFeedback] = None
    usage_metadata: Optional[GoogleUsageMetadata] = None

    def to_generic(self) -> LlmResponse:
        """Converts an GoogleResponse into a generic LlmResponse instance."""
        return LlmResponse(
            generated_text=(
                self.candidates[0].content.parts[0].text
                if self.candidates
                and len(self.candidates) > 0
                and getattr(self.candidates[0], "content", None)
                and getattr(self.candidates[0].content, "parts", None)
                and len(self.candidates[0].content.parts[0]) > 0
                and isinstance(self.candidates[0].content.parts[0].text, str)
                else None
            ),
            model=self.model_version,
            generated_token_count=getattr(self.usage_metadata, "candidates_token_count", None),
            input_token_count=getattr(self.usage_metadata, "prompt_token_count", None),
            google=self,
        )
