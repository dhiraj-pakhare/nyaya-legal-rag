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
def test_ollama_provider_thinking_field_never_returned_as_content(mock_urlopen):
    """Verify that Ollama message['thinking'] is NEVER returned as answer content."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {
            "content": "",
            "thinking": "Internal reasoning about rules, cannot fabricate, 303 vs 303(2)"
        },
        "prompt_eval_count": 50,
        "eval_count": 40
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OllamaProvider(model="qwen3:4b", base_url="http://localhost:11434")
    resp = provider.generate([LLMMessage(role="user", content="What is Section 303(2)?")])

    assert resp.content == ""
    assert "Internal reasoning" not in resp.content


@patch("urllib.request.urlopen")
def test_ollama_provider_think_tags_stripped_from_content(mock_urlopen):
    """Verify that <think>...</think> blocks inside message content are stripped."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "message": {
            "content": "<think>\nDebating Section 303 vs 303(2)...\n</think>\n\nAccording to [BNS s.303(2)], theft is cognizable."
        },
        "prompt_eval_count": 60,
        "eval_count": 50
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OllamaProvider(model="qwen3:4b", base_url="http://localhost:11434")
    resp = provider.generate([LLMMessage(role="user", content="What is Section 303(2)?")])

    assert "<think>" not in resp.content
    assert "</think>" not in resp.content
    assert "Debating Section 303" not in resp.content
    assert resp.content == "According to [BNS s.303(2)], theft is cognizable."


@patch("urllib.request.urlopen")
def test_ollama_provider_streaming_does_not_emit_thinking(mock_urlopen):
    """Verify that streamed thinking chunks and <think>...</think> blocks are not yielded."""
    # Simulate stream chunks:
    # 1. Thinking chunk in message.thinking
    # 2. <think> tag in content
    # 3. internal thought in content
    # 4. </think> tag in content
    # 5. actual answer token in content
    lines = [
        json.dumps({"message": {"content": "", "thinking": "Internal thought 1"}}).encode("utf-8") + b"\n",
        json.dumps({"message": {"content": "<think>Internal "}}).encode("utf-8") + b"\n",
        json.dumps({"message": {"content": "monologue</think>Grounded "}}).encode("utf-8") + b"\n",
        json.dumps({"message": {"content": "answer [BNS s.303(2)]"}}).encode("utf-8") + b"\n",
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__.return_value = iter(lines)
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OllamaProvider(model="qwen3:4b", base_url="http://localhost:11434")
    tokens = list(provider.stream([LLMMessage(role="user", content="What is Section 303(2)?")]))

    combined = "".join(tokens)
    assert "Internal thought 1" not in combined
    assert "Internal monologue" not in combined
    assert "<think>" not in combined
    assert "</think>" not in combined
    assert "Grounded answer [BNS s.303(2)]" in combined


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

    # Verify custom User-Agent and Authorization headers were sent
    req_arg = mock_urlopen.call_args[0][0]
    assert req_arg.get_header("User-agent") == "NyayaLegalRAG/1.0 (OpenAI-Compatible Client)"
    assert req_arg.get_header("Authorization") == "Bearer sk-test-key"


@patch("urllib.request.urlopen")
def test_openai_provider_reasoning_content_and_think_tags(mock_urlopen):
    """Verify OpenAI-compatible provider handles reasoning_content and strips <think> tags."""
    # Test case 1: Reasoning content fallback when content is empty/None
    mock_resp1 = MagicMock()
    mock_resp1.read.return_value = json.dumps({
        "choices": [{"message": {"content": None, "reasoning_content": "Under [BNS s.103], murder is defined."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    }).encode("utf-8")
    mock_resp1.__enter__.return_value = mock_resp1

    # Test case 2: <think> tags stripped from content
    mock_resp2 = MagicMock()
    mock_resp2.read.return_value = json.dumps({
        "choices": [{"message": {"content": "<think>Deliberating statutory text</think>Under [BNS s.103], murder is defined."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    }).encode("utf-8")
    mock_resp2.__enter__.return_value = mock_resp2

    mock_urlopen.side_effect = [mock_resp1, mock_resp2]

    provider = OpenAICompatibleProvider(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_key"
    )

    # Call 1: reasoning_content
    resp1 = provider.generate([LLMMessage(role="user", content="Define murder.")])
    assert resp1.content == "Under [BNS s.103], murder is defined."

    # Call 2: <think> tags stripped
    resp2 = provider.generate([LLMMessage(role="user", content="Define murder.")])
    assert "<think>" not in resp2.content
    assert "</think>" not in resp2.content
    assert resp2.content == "Under [BNS s.103], murder is defined."


@patch("urllib.request.urlopen")
def test_openai_provider_streaming_with_user_agent(mock_urlopen):
    """Verify OpenAI-compatible streaming sends User-Agent and handles delta content."""
    lines = [
        b"data: " + json.dumps({"choices": [{"delta": {"content": "Under "}}]}).encode("utf-8") + b"\n",
        b"data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "[BNS s.103]"}}]}).encode("utf-8") + b"\n",
        b"data: [DONE]\n"
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__.return_value = iter(lines)
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    provider = OpenAICompatibleProvider(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_key"
    )

    tokens = list(provider.stream([LLMMessage(role="user", content="Define murder.")]))
    assert "".join(tokens) == "Under [BNS s.103]"

    req_arg = mock_urlopen.call_args[0][0]
    assert req_arg.get_header("User-agent") == "NyayaLegalRAG/1.0 (OpenAI-Compatible Client)"
