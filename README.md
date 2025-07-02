# LlmUnify

**LlmUnify** is a Python library designed to simplify and standardize interactions with multiple Large Language Model (LLM) providers. By offering a unified interface, it enables seamless integration and allows you to switch between providers or models without modifying your code. The library abstracts the complexity of invoking LLMs, supports streaming responses, and can be easily configured via environment variables or method arguments.

## Supported Providers

- **AWS Bedrock** (via `boto3`)
- **Azure AI Foundry** (via `azure-ai-inference`)
- **Google Vertex Ai** (via `vertexai`)
- **Ollama** (via direct API calls)
- **IBM WatsonX** (via `ibm-watsonx-ai`)

Future versions will include support for additional providers.

## Installation

### Install the core library from PyPI

To install the library directly from [PyPI](https://pypi.org/project/llmUnify/), run:

```bash
pip install llmUnify
```

---

### Install with dependencies for a specific provider

To install the library with **AWS-specific** dependencies, use:

```bash
pip install 'llmUnify[aws]'
```

---

### Install with all provider-specific dependencies

To install the library with **all available provider-specific dependencies**, use:

```bash
pip install 'llmUnify[all]'
```

## Quickstart

### Configuration and Authentication

LlmUnify retrieves provider-specific credentials and configuration from environment variables, with the option to override them using method arguments.

Example `.env` configuration:

```dotenv
OLLAMA_API_KEY=your_ollama_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### Minimal Example

```python
from llmUnify import LlmOptions, LlmUnify
from dotenv import load_dotenv

# Load configurations from .env file
load_dotenv()

# Define options for text generation
options = LlmOptions(
    temperature=0.7,
    prompt="Write a motivational poem:",
)

# Generate a response specifying provider and model
result = LlmUnify.generate(
    model="<provider>:<model-name>",  # Provider and model name separated by ":"
    options=options,
)

print(result.generated_text)
```

### Reusing Connectors

For repeated calls to the same provider, you can use a reusable connector:

```python
from llmUnify import LlmOptions, LlmUnify
from dotenv import load_dotenv

# Load configurations from .env file
load_dotenv()

# Create a connector for a specific provider
connector = LlmUnify.get_connector(
    provider="<provider>",
)

# Define options for text generation
options = LlmOptions(prompt="List three ways to stay productive:")

# Generate a response in streaming mode
result = connector.generate_stream(model_name="model_name", options=options)

print(result.generated_text)
```

### Usage Metrics Logging

LlmUnify can log usage statistics such as token counts, elapsed time, and optional call identifiers.

#### Enable logging via environment variables:

```dotenv
LLM_UNIFY_ENABLE_USAGE_METRICS=true
LLM_UNIFY_USAGE_METRICS_OUTPUT_PATH=llm_unify_usage_metrics.csv  # optional, default is ./llm_unify_usage_metrics.csv
```

#### Enable logging via code:

```python
from llmUnify import configure_usage_metrics

UsageMetricsLogger.enable_usage_metrics(enabled=True, output_path="my_metrics.csv")
```

#### Pass a custom call name to identify calls in the logs:

```python
result = LlmUnify.generate(
    provider="<provider>:<model-name>",
    options=options,
    call_name="<call_name>"
)
```

After each call with logging enabled, a new row will be appended to the CSV file at the specified path.
