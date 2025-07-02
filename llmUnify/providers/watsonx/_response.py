# Based on the documentation at https://cloud.ibm.com/apidocs/watsonx-ai#text-generation

from typing import Optional

from pydantic import BaseModel

from ..._response import LlmResponse


class WatsonxTextGenTopTokenInfo(BaseModel):
    text: Optional[str] = None
    logprob: Optional[float] = None


class WatsonxTextGenTokenInfo(BaseModel):
    text: Optional[str] = None
    logprob: Optional[float] = None
    rank: Optional[int] = None
    top_tokens: Optional[list[WatsonxTextGenTopTokenInfo]] = None


class WatsonxModerationTextRange(BaseModel):
    start: int
    end: int


class WatsonxModerationResult(BaseModel):
    score: float
    input: bool
    position: WatsonxModerationTextRange
    entity: str
    word: Optional[str] = None


class WatsonxModerationResults(BaseModel):
    hap: Optional[list[WatsonxModerationResult]] = None
    pii: Optional[list[WatsonxModerationResult]] = None
    other_property: Optional[list[WatsonxModerationResult]] = None


class WatsonxResults(BaseModel):
    generated_text: str
    stop_reason: str
    generated_token_count: Optional[int] = None
    input_token_count: Optional[int] = None
    seed: Optional[int] = None
    generated_tokens: Optional[list[WatsonxTextGenTokenInfo]] = None
    input_tokens: Optional[list[WatsonxTextGenTokenInfo]] = None
    moderations: Optional[WatsonxModerationResults] = None


class WatsonxWatsonxWarning(BaseModel):
    message: str
    id: Optional[str] = None
    more_info: Optional[str] = None
    additional_properties: Optional[dict] = None


class WatsonxSystemDetails(BaseModel):
    warnings: Optional[list[WatsonxWatsonxWarning]] = None


class WatsonxResponse(BaseModel):
    model_id: str
    created_at: str
    results: list[WatsonxResults]
    model_version: Optional[str] = None
    system: Optional[WatsonxSystemDetails] = None

    def to_generic(self) -> LlmResponse:
        """Converts an WatsonxResponse into a generic LlmResponse instance."""
        return LlmResponse(
            generated_text=self.results[0].generated_text,
            model=self.model_id,
            generated_token_count=self.results[0].generated_token_count,
            input_token_count=self.results[0].input_token_count,
            watsonx=self,
        )
