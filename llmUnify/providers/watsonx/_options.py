# Based on the documentation at https://cloud.ibm.com/apidocs/watsonx-ai#text-generation

from typing import Literal, Optional

from pydantic import BaseModel

from ..._options import LlmOptions


class WatsonxReturnOptionProperties(BaseModel):
    input_text: Optional[bool] = None
    generated_tokens: Optional[bool] = None
    input_tokens: Optional[bool] = None
    token_logprobs: Optional[bool] = None
    token_ranks: Optional[bool] = None
    top_n_tokens: Optional[bool] = None


class WatsonxTextGenLengthPenalty(BaseModel):
    decay_factor: Optional[float] = None
    start_index: Optional[int] = None


class WatsonxTextGenParameters(BaseModel):
    decoding_method: Optional[Literal["sample", "greedy"]] = None
    length_penalty: Optional[WatsonxTextGenLengthPenalty] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    temperature: Optional[float] = None
    time_limit: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    return_options: Optional[WatsonxReturnOptionProperties] = None
    include_stop_sequence: Optional[bool] = None


class WatsonxTextModeration(BaseModel):
    enabled: Optional[bool] = None
    threshold: Optional[float] = None
    other_property: Optional[dict] = None


class WatsonxMaskProperties(BaseModel):
    remove_entity_value: Optional[bool] = None


class WatsonxModerationTextRange(BaseModel):
    start: int
    end: int


class WatsonxModerationProperties(BaseModel):
    input: Optional[WatsonxTextModeration] = None
    output: Optional[WatsonxTextModeration] = None
    other_property: Optional[dict] = None


class WatsonxModerationHapProperties(WatsonxModerationProperties):
    mask: Optional[WatsonxMaskProperties] = None


class WatsonxModerationPiiProperties(WatsonxModerationProperties):
    mask: Optional[WatsonxMaskProperties] = None


class WatsonxModerations(BaseModel):
    hap: Optional[WatsonxModerationHapProperties] = None
    pii: Optional[WatsonxModerationPiiProperties] = None
    input_ranges: Optional[list[WatsonxModerationTextRange]] = None
    other_property: Optional[dict] = None


class WatsonxOptions(BaseModel):
    input: Optional[str] = None
    parameters: Optional[WatsonxTextGenParameters] = None
    moderations: Optional[WatsonxModerations] = None

    @classmethod
    def from_generic(cls, generic_options: LlmOptions) -> "WatsonxOptions":
        """Creates an WatsonxOptions instance from a generic LlmOptions."""

        mapping_obj = {
            "input": generic_options.prompt,
            **(generic_options.watsonx.model_dump(exclude_none=True) if generic_options.watsonx else {}),
            "parameters": {
                "max_new_tokens": generic_options.max_tokens,
                "temperature": generic_options.temperature,
                "top_p": generic_options.top_p,
                "stop_sequences": generic_options.stop_sequences,
                **(
                    generic_options.watsonx.parameters.model_dump(exclude_none=True)
                    if generic_options.watsonx and generic_options.watsonx.parameters
                    else {}
                ),
            },
        }

        return cls(**mapping_obj)
