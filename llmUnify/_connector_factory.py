import functools
import importlib
import re
from pathlib import Path
from typing import Iterator, Optional

from ._abstract_connector import LlmConnector
from ._options import LlmOptions
from ._response import LlmResponse

PROVIDER_PATTERN = r"^([a-zA-Z0-9_-]+):(.+)$"

PROVIDERS_DIR = Path(__file__).parent / "providers"


class LlmUnify:
    @classmethod
    def get_connector(cls, provider: str, **kwargs) -> "LlmConnector":
        """
        Creates an instance of the connector for the specified LLM provider.

        Parameters:
            provider (str): The name of the desired LLM provider. This should match one of the supported providers.
            **kwargs (dict): Additional parameters specific to the provider, such as
                authentication details, or custom configuration options.

        Returns:
            LlmConnector: An instance of the connector for the specified provider.

        Raises:
            ValueError: If the specified provider is not supported or invalid.
            ImportError: If the connector module for the provider cannot be imported.
        """

        if provider not in cls._get_providers():
            raise ValueError(
                f"Provider '{provider}' is not supported. Supported providers are: {cls._get_providers()}."
            )

        module_path = f"llmUnify.providers.{provider}._connector"
        class_name = f"{provider.capitalize()}Connector"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import the module '{module_path}': {str(e)}.\n"
                f"Ensure the required dependencies for the provider '{provider}' are installed.\n"
                f"You may need to run 'pip install llmUnify[{provider}]'."
            )

        _class = getattr(module, class_name)
        return _class(**kwargs)

    @classmethod
    def generate(cls, model: str, options: LlmOptions, call_name: Optional[str] = None, **kwargs) -> LlmResponse:
        """
        Generates a single response using the specified LLM model.

        Parameters:
            model (str): A string that specifies the provider and model name, formatted as `<provider>:<model-name>`.
            options (LlmOptions): An instance of `LlmOptions` containing the parameters for text generation.
            call_name (str, optional): An optional identifier for the call to be logged.
            **kwargs (dict): Additional parameters specific to the provider, such as
                authentication details, or custom configuration options.

        Returns:
            LlmResponse: The generated response.

        Raises:
            ValueError: If the model string format is invalid.
        """
        provider, model_name = cls._validate_model(model)
        return cls.get_connector(provider, **kwargs).generate(model_name, options, call_name=call_name)

    @classmethod
    def generate_stream(
        cls,
        model,
        options: LlmOptions,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> Iterator[LlmResponse]:
        """
        Generates responses as a stream using the specified LLM model.

        Parameters:
            model (str): A string that specifies the provider and model name, formatted as `<provider>:<model-name>`.
            options (LlmOptions): An instance of `LlmOptions` containing the parameters for text generation.
            call_name (str, optional): An optional identifier for the call to be logged.
            **kwargs (dict): Additional parameters specific to the provider, such as
                authentication details, or custom configuration options.

        Returns:
            Iterator[LlmResponse]: An iterator of generated responses.

        Raises:
            ValueError: If the model string format is invalid.
        """
        provider, model_name = cls._validate_model(model)
        yield from cls.get_connector(provider, **kwargs).generate_stream(model_name, options, call_name=call_name)

    @classmethod
    def _validate_model(cls, model: str) -> tuple[str, str]:
        """
        Validates and parses the input model string into provider and model name.

        The model string must be formatted as `<provider>:<model-name>`. This method extracts the components
        and ensures the format is correct.

        Parameters:
            model (str): A string in the format `<provider>:<model-name>`.

        Returns:
            tuple[str, str]: A tuple containing the provider name and model name.

        Raises:
            ValueError: If the model string format is invalid.
        """
        match = re.match(PROVIDER_PATTERN, model)
        if match:
            return match.group(1), match.group(2)

        raise ValueError("Invalid model string")

    @classmethod
    @functools.cache
    def _get_providers(cls) -> set[str]:
        """
        Retrieves a set of supported provider names by inspecting the providers directory.

        This method scans the `providers` directory for subdirectories containing a `_connector.py` file,
        which indicates support for a provider.

        Returns:
            set[str]: A set of supported provider names.

        Notes:
            - Results are cached for efficiency, so subsequent calls do not re-scan the directory.
        """
        provider_dirs = [d for d in PROVIDERS_DIR.iterdir() if d.is_dir()]

        supported_providers = {d.name for d in provider_dirs if (d / "_connector.py").exists()}

        return supported_providers
