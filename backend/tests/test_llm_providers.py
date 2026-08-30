"""Unit and integration tests for LLM provider abstraction."""

import json
from unittest.mock import MagicMock, patch
import pytest

from backend.app.generation.models import LLMMessage
from backend.app.generation.providers import (
    LLMConfigurationError,
    LLMProviderError,
    LLMTimeoutError,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    get_llm_provider
)


def test_mock_provider_generation():
    """Test deterministic MockLLMProvider generation and token tracking."""
    provider = MockLLMProvider(
        default_response="Murder is defined under [BNS s.103].",
        prompt_tokens=100,
        completion_tokens=25
    )
    messages = [
        LLMMessage(role="system", content="You are a legal assistant."),
        LLMMessage(role="user", content="Explain murder punishment.")
    ]
    resp = provider.generate(messages)
    assert resp.content == "Murder is defined under [BNS s.103]."
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 25
    assert resp.total_tokens == 125
    assert len(provider.call_history) == 1


def test_mock_provider_queue_and_streaming():
    """Test sequential response queuing and token streaming."""
    provider = MockLLMProvider()
    provider.set_responses([
        "First response [BNS s.999]",
        "Second response [BNS s.103]"
    ])
    messages = [LLMMessage(role="user", content="test")]
    
    resp1 = provider.generate(messages)
    assert resp1.content == "First response [BNS s.999]"
    
    resp2 = provider.generate(messages)
    assert resp2.content == "Second response [BNS s.103]"

    # Test streaming
    tokens = list(provider.stream(messages))
    assert len(tokens) > 0
    assert "".join(tokens).strip() == provider.default_response.strip()


def test_mock_provider_errors():
    """Test mock provider simulated timeout and error handling."""
    timeout_provider = MockLLMProvider(simulate_timeout=True)
    with pytest.raises(LLMTimeoutError):
        timeout_provider.generate([LLMMessage(role="user", content="hi")])

    err_provider = MockLLMProvider(simulate_error="Simulated GPU OOM")
    with pytest.raises(LLMProviderError) as exc_info:
        err_provider.generate([LLMMessage(role="user", content="hi")])
    assert "Simulated GPU OOM" in str(exc_info.value)


def test_get_llm_provider_factory():
    """Test factory function resolves correct provider classes."""
    ollama_p = get_llm_provider("ollama", model="llama3.2")
    assert isinstance(ollama_p, OllamaProvider)
    assert ollama_p.model == "llama3.2"

    openai_p = get_llm_provider("openai", model="gpt-4o", base_url="https://api.openai.com/v1")
    assert isinstance(openai_p, OpenAICompatibleProvider)
    assert openai_p.model == "gpt-4o"

    mock_p = get_llm_provider("mock")
    assert isinstance(mock_p, MockLLMProvider)

    with pytest.raises(LLMConfigurationError):
        get_llm_provider("unsupported_xyz")


@patch("urllib.request.urlopen")
def test_ollama_provider_http_success(mock_urlopen):
    """Test Ollama provider executes HTTP POST and parses response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {"content": "As per [BNSS s.35], police may arrest without warrant."},
        "prompt_eval_count": 80,
        "eval_count": 30
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2", base_url="http://localhost:11434")
    messages = [LLMMessage(role="user", content="When can police arrest?")]
    
    resp = provider.generate(messages)
    assert "As per [BNSS s.35]" in resp.content
    assert resp.prompt_tokens == 80
    assert resp.completion_tokens == 30
    assert resp.total_tokens == 110


@patch("urllib.request.urlopen")
def test_openai_provider_http_success(mock_urlopen):
    """Test OpenAI-compatible provider executes HTTP POST with auth headers."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "Under [BNS s.105], culpable homicide is punishable."}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key"
    )
    messages = [LLMMessage(role="user", content="What is culpable homicide?")]
    
    resp = provider.generate(messages)
    assert "[BNS s.105]" in resp.content
    assert resp.prompt_tokens == 120
    assert resp.completion_tokens == 40
    assert resp.total_tokens == 160
