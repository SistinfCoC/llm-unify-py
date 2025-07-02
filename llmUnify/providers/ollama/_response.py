# Based on the documentation at https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
# and ollama sdk for python

from typing import Any, Literal, Optional

from pydantic import BaseModel

from ..._response import LlmResponse


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    thinking: Optional[str] = None
    images: Optional[list[str]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class OllamaResponse(BaseModel):
    message: Message
    model: str
    created_at: str
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    def to_generic(self) -> LlmResponse:
        """Converts an OllamaResponse into a generic LlmResponse instance."""
        return LlmResponse(
            generated_text=self.message.content,
            model=self.model,
            generated_token_count=self.eval_count,
            input_token_count=self.prompt_eval_count,
            ollama=self,
        )
