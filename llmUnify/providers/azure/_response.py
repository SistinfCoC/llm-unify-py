# Based on the documentation at https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/get-chat-completions

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel

from ..._response import LlmResponse


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ChatCompletionsToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatResponseMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool", "developer"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionsToolCall]] = None


class ChatChoice(BaseModel):
    index: int
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]]
    message: Optional[ChatResponseMessage] = None


class StreamingChatChoiceUpdate(BaseModel):
    index: int
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None
    delta: ChatResponseMessage


class CompletionsUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class AzureResponse(BaseModel):
    id: str
    created: datetime
    model: str
    choices: list[Union[ChatChoice, StreamingChatChoiceUpdate]]
    usage: Optional[CompletionsUsage] = None

    def to_generic(self) -> LlmResponse:
        """Converts an AwsResponse into a generic LlmResponse instance."""
        generated_text = None

        if self.choices:
            choice = self.choices[0]
            # Se è un ChatChoice
            if hasattr(choice, "message") and choice.message and choice.message.content:
                generated_text = choice.message.content
            # Se è un StreamingChatChoiceUpdate
            elif hasattr(choice, "delta") and choice.delta and choice.delta.content:
                generated_text = choice.delta.content

        return LlmResponse(
            generated_text=generated_text,
            model=self.model,
            generated_token_count=getattr(self.usage, "completion_tokens", None),
            input_token_count=getattr(self.usage, "prompt_tokens", None),
            azure=self,
        )
