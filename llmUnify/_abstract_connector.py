from abc import ABC, abstractmethod
from time import perf_counter
from typing import Iterator, Optional

from ._options import LlmOptions
from ._response import LlmResponse
from ._usage_metrics_logger import UsageMetricsLogger


class LlmConnector(ABC):
    def generate(self, model_name: str, options: LlmOptions, call_name: Optional[str] = None) -> LlmResponse:
        """
        Generates a single response using the specified LLM model.

        Parameters:
            model_name (str): The name of the model to be used for generating responses.
            options (LlmOptions): An instance of `LlmOptions` containing the parameters for text generation.
            call_name (str, optional): An optional identifier for the call to be logged.

        Returns:
            LlmResponse: The generated response.
        """
        start = perf_counter()
        response: LlmResponse = self._generate(model_name, options)
        elapsed = perf_counter() - start

        provider = self._get_provider_name()
        UsageMetricsLogger.log_usage_metrics(response, elapsed=elapsed, provider=provider, call_name=call_name)

        return response

    def generate_stream(
        self,
        model_name: str,
        options: LlmOptions,
        call_name: Optional[str] = None,
    ) -> Iterator[LlmResponse]:
        """
        Generates responses as a stream using the specified LLM model.

        Parameters:
            model (str): The name of the model to be used for generating responses.
            options (LlmOptions): An instance of `LlmOptions` containing the parameters for text generation.
            call_name (str, optional): An optional identifier for the call to be logged.

        Returns:
            Iterator[LlmResponse]: An iterator of generated responses.

        """
        start = perf_counter()
        response_iterator: Iterator[LlmResponse] = self._generate_stream(model_name, options)

        for response in response_iterator:
            yield response

        elapsed = perf_counter() - start
        provider = self._get_provider_name()
        UsageMetricsLogger.log_usage_metrics(response, elapsed=elapsed, provider=provider, call_name=call_name)

    def _get_provider_name(self) -> str:
        class_name = self.__class__.__name__
        provider = class_name.replace("Connector", "").lower()
        return provider

    @abstractmethod
    def _generate(self, model_name: str, options: LlmOptions) -> LlmResponse:
        raise NotImplementedError("LlmConnector Interface has not implemented _generate()")

    @abstractmethod
    def _generate_stream(self, model_name: str, options: LlmOptions) -> Iterator[LlmResponse]:
        raise NotImplementedError("LlmConnector Interface has not implemented _generate_stream()")
