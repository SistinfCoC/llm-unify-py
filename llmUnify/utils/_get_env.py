import os


class EnvironmentVariableError(Exception):
    """Raised when a required environment variable is not set."""

    pass


def get_required_env(var_name: str) -> str:
    """
    Retrieves the value of the specified environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable.

    Raises:
        EnvironmentVariableError: If the specified environment variable is not set.
    """
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentVariableError(f"The required environment variable '{var_name}' is not set.")
    return value


def get_env(var_name) -> str | None:
    """
    Retrieves the value of the specified environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.

    Returns:
        str | None: The value of the environment variable, or None if the variable does not exist.
    """
    return os.getenv(var_name)
