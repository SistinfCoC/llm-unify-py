# Based on the documentation at https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
# and ollama sdk for python

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

from ..._options import LlmOptions


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    thinking: Optional[str] = None
    images: Optional[list[str]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class Property(BaseModel):
    type: Optional[Union[str, list[str]]] = None
    items: Optional[Any] = None
    description: Optional[str] = None
    enum: Optional[list[Any]] = None


class Parameters(BaseModel):
    type: Optional[Literal["object"]] = "object"
    defs: Optional[Any] = None
    items: Optional[Any] = None
    required: Optional[list[str]] = None
    properties: Optional[dict[str, Property]] = None


class Function(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Parameters] = None


class Tool(BaseModel):
    type: Optional[Literal["function"]] = "function"
    function: Optional[Function] = None


class OllamaModelParameters(BaseModel):
    mirostat: Optional[int] = None
    mirostat_eta: Optional[float] = None
    mirostat_tau: Optional[float] = None
    num_ctx: Optional[int] = None
    repeat_last_n: Optional[int] = None
    repeat_penalty: Optional[float] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[list[str]] = None
    tfs_z: Optional[float] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None


class OllamaOptions(BaseModel):
    messages: Optional[list[Message]] = None
    tools: Optional[list[Tool]] = None
    think: Optional[bool] = None
    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    options: Optional[OllamaModelParameters] = None
    keep_alive: Optional[Union[float, str]] = None

    @classmethod
    def from_generic(cls, generic_options: LlmOptions) -> "OllamaOptions":
        """Creates an OllamaOptions instance from a generic LlmOptions."""

        mapping_obj = {
            "messages": [{"role": "user", "content": generic_options.prompt}],
            **(generic_options.ollama.model_dump(exclude_none=True) if generic_options.ollama else {}),
            "options": {
                "num_predict": generic_options.max_tokens,
                "temperature": generic_options.temperature,
                "top_p": generic_options.top_p,
                "stop": generic_options.stop_sequences,
                **(
                    generic_options.ollama.options.model_dump(exclude_none=True)
                    if generic_options.ollama and generic_options.ollama.options
                    else {}
                ),
            },
        }

        return cls(**mapping_obj)
