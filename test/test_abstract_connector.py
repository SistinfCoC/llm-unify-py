from unittest.mock import patch

from pytest import raises

from llmUnify._abstract_connector import LlmConnector, LlmOptions


@patch.object(LlmConnector, "__abstractmethods__", set())
def test_abs_generate():
    connector = LlmConnector()

    with raises(NotImplementedError):
        connector._generate("test-model", LlmOptions(prompt="Test"))


@patch.object(LlmConnector, "__abstractmethods__", set())
def test_abs_generate_stream():
    connector = LlmConnector()

    with raises(NotImplementedError):
        connector._generate_stream("test-model", LlmOptions(prompt="Test"))
