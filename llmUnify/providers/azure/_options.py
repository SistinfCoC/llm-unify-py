# Based on the documentation at https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/get-chat-completions

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..._options import LlmOptions


class ChatRequestAudioReference(BaseModel):
    id: str


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ChatCompletionsToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatRequestAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    audio: ChatRequestAudioReference
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionsToolCall]] = None


class ChatRequestSystemMessage(BaseModel):
    role: Literal["system", "developer"] = "system"  # Based on azure python SDK
    content: str


class ChatRequestToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: Optional[str]
    tool_call_id: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


class ImageContentItem(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class InputAudio(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class AudioContentItem(BaseModel):
    type: Literal["image_url"] = "image_url"
    input_audio: InputAudio


class TextContentItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatRequestUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[
        str,
        ImageContentItem,
        AudioContentItem,
        TextContentItem,
    ]


class JsonSchemaFormat(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(None, alias="json")
    description: Optional[str] = None
    strict: Optional[bool] = None


class ChatCompletionsNamedToolChoiceFunction(BaseModel):
    name: str


class ChatCompletionsNamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ChatCompletionsNamedToolChoiceFunction


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Any] = None


class ChatCompletionsToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class AzureOptions(BaseModel):
    messages: list[
        Union[ChatRequestAssistantMessage, ChatRequestSystemMessage, ChatRequestToolMessage, ChatRequestUserMessage]
    ]
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Union[Literal["text", "json_object"], JsonSchemaFormat]] = None
    seed: Optional[int] = None
    stop: Optional[list[str]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], ChatCompletionsNamedToolChoice]] = None
    tools: Optional[list[ChatCompletionsToolDefinition]] = None
    top_p: Optional[float] = None
    model_extras: Optional[dict[str, Any]] = None

    @classmethod
    def from_generic(cls, generic_options: LlmOptions) -> "AzureOptions":
        """Creates an AzureOptions instance from a generic LlmOptions."""

        mapping_obj = {
            "messages": [{"role": "user", "content": generic_options.prompt}],
            "max_tokens": generic_options.max_tokens,
            "temperature": generic_options.temperature,
            "top_p": generic_options.top_p,
            "stop": generic_options.stop_sequences,
            **(generic_options.azure.model_dump(exclude_none=True, by_alias=True) if generic_options.azure else {}),
        }

        return cls(**mapping_obj)
