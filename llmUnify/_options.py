from typing import Optional

from pydantic import BaseModel


class LlmOptions(BaseModel):
    """
    Represents the standardized options for generating text with an LLM provider.

    This model provides a unified interface for specifying parameters used during text generation,
    while also allowing provider-specific configurations to be included.
    Provider-specific options allow fine-tuning or
    overriding the generic options for a specific provider.

    Attributes:
        prompt (str): The input prompt for text generation.
        max_tokens (Optional[int]): The maximum number of tokens to generate in the response.
        temperature (Optional[float]): The sampling temperature for text generation.
            Lower values make outputs more deterministic, while higher values increase randomness.
        top_p (Optional[float]): The nucleus sampling parameter. Specifies the cumulative probability
            threshold for token selection.
        stop_sequences (Optional[list[str]]): A list of sequences that will terminate text generation
            if encountered in the output.

        aws (Optional[AwsOptions]): Provider-specific options for AWS Bedrock.
        azure (Optional[AzureOptions]): Provider-specific options for Azure Foundry.
        google (Optional[GoogleOptions]): Provider-specific options for Google Vertex AI.
        ollama (Optional[OllamaOptions]): Provider-specific options for Ollama.
        watsonx (Optional[WatsonxOptions]): Provider-specific options for IBM WatsonX.
    """

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None

    aws: Optional["AwsOptions"] = None
    azure: Optional["AzureOptions"] = None
    google: Optional["GoogleOptions"] = None
    ollama: Optional["OllamaOptions"] = None
    watsonx: Optional["WatsonxOptions"] = None


from .providers.aws._options import AwsOptions  # noqa: E402
from .providers.azure._options import AzureOptions  # noqa: E402
from .providers.google._options import GoogleOptions  # noqa: E402
from .providers.ollama._options import OllamaOptions  # noqa: E402
from .providers.watsonx._options import WatsonxOptions  # noqa: E402

LlmOptions.model_rebuild()
