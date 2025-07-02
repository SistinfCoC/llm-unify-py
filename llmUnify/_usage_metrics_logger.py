import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

from ._response import LlmResponse
from .utils._get_env import get_env


class UsageMetricsLogger:
    _lock = threading.Lock()
    _enabled = get_env("LLM_UNIFY_ENABLE_USAGE_METRICS") in {"1", "true", "True"}
    _output_path = Path(get_env("LLM_UNIFY_USAGE_METRICS_OUTPUT_PATH") or "llm_unify_usage_metrics.csv")

    @staticmethod
    def log_usage_metrics(response: LlmResponse, elapsed: float = None, provider: str = None, call_name: str = None):
        """
        Log the generation statistics to the CSV file if logging is enabled.

        Args:
            response (LlmResponse): The LLM response object to log.
        """
        if not UsageMetricsLogger._enabled:
            return

        data = {
            "timestamp": datetime.now(timezone.utc),
            "call_name": call_name or "None",
            "provider": provider or "",
            "model": response.model or "",
            "input_token_count": response.input_token_count or 0,
            "generated_token_count": response.generated_token_count or 0,
            "elapsed": elapsed or 0.0,
        }

        with UsageMetricsLogger._lock:
            file_exists = UsageMetricsLogger._output_path.exists()

            with open(UsageMetricsLogger._output_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                if not file_exists and UsageMetricsLogger._output_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(data)


def configure_usage_metrics(enabled: bool = True, output_path: str = None):
    """
    Enable/disable logging and optionally set a custom output path.

    Args:
        enabled (bool): Enable or disable metrics logging.
        output_path (str, optional): Custom path for the CSV file.
    """
    UsageMetricsLogger._enabled = enabled
    if output_path:
        UsageMetricsLogger._output_path = Path(output_path)
