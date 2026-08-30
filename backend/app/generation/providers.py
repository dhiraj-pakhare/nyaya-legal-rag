"""Provider-independent LLM interfaces and implementations."""

from abc import ABC, abstractmethod
import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional
import urllib.error
import urllib.request

from backend.app.core.config import settings
from backend.app.generation.models import LLMMessage, LLMResponse

logger = logging.getLogger("nyaya.generation.providers")


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider call times out."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when provider configuration is invalid or missing."""
    pass


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a complete completion for given messages."""
        pass

    @abstractmethod
    def stream(self, messages: List[LLMMessage], **kwargs) -> Generator[str, None, None]:
        """Stream completion tokens for given messages."""
        pass


class OllamaProvider(LLMProvider):
    """Provider for local Ollama instances (default: http://localhost:11434)."""

    def __init__(
        self,
        model: str = settings.llm_model,
        base_url: str = settings.llm_base_url,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
        timeout: float = settings.llm_timeout,
        max_retries: int = settings.llm_max_retries
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise LLMConfigurationError("Ollama base URL cannot be empty")

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    content = resp_data.get("message", {}).get("content", "")
                    prompt_tokens = resp_data.get("prompt_eval_count")
                    completion_tokens = resp_data.get("eval_count")
                    total_tokens = None
                    if prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_ms=round(latency_ms, 2),
                        raw_response=resp_data
                    )
            except urllib.error.HTTPError as e:
                err_msg = f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"
                logger.warning(f"Ollama attempt {attempt + 1} failed: {err_msg}")
                last_err = LLMProviderError(err_msg)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if "timed out" in str(e).lower():
                    last_err = LLMTimeoutError(f"Ollama call timed out after {self.timeout}s")
                else:
                    last_err = LLMProviderError(f"Ollama connection error: {str(e)}")
                logger.warning(f"Ollama attempt {attempt + 1} failed: {str(last_err)}")

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        raise last_err or LLMProviderError("Ollama generation failed after all retries")

    def stream(self, messages: List[LLMMessage], **kwargs) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
        except Exception as e:
            raise LLMProviderError(f"Ollama streaming failed: {str(e)}") from e


class OpenAICompatibleProvider(LLMProvider):
    """Provider for standard OpenAI-compatible endpoints (OpenAI, Groq, Together, vLLM, etc.)."""

    def __init__(
        self,
        model: str = settings.llm_model,
        base_url: str = settings.llm_base_url,
        api_key: str = settings.llm_api_key,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
        timeout: float = settings.llm_timeout,
        max_retries: int = settings.llm_max_retries
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if not self.base_url:
            raise LLMConfigurationError("OpenAI-compatible base URL cannot be empty")

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    choice = resp_data.get("choices", [{}])[0]
                    content = choice.get("message", {}).get("content", "")
                    usage = resp_data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")

                    return LLMResponse(
                        content=content,
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_ms=round(latency_ms, 2),
                        raw_response=resp_data
                    )
            except urllib.error.HTTPError as e:
                err_msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"
                logger.warning(f"Provider attempt {attempt + 1} failed: {err_msg}")
                last_err = LLMProviderError(err_msg)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if "timed out" in str(e).lower():
                    last_err = LLMTimeoutError(f"Call timed out after {self.timeout}s")
                else:
                    last_err = LLMProviderError(f"Connection error: {str(e)}")
                logger.warning(f"Provider attempt {attempt + 1} failed: {str(last_err)}")

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        raise last_err or LLMProviderError("OpenAI-compatible generation failed after retries")

    def stream(self, messages: List[LLMMessage], **kwargs) -> Generator[str, None, None]:
        url = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        body = line[6:]
                        if body == "[DONE]":
                            break
                        chunk = json.loads(body)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
        except Exception as e:
            raise LLMProviderError(f"Streaming failed: {str(e)}") from e


class MockLLMProvider(LLMProvider):
    """Deterministic in-memory Mock LLM provider for unit tests, CI, and simulation."""

    def __init__(
        self,
        model: str = "mock-statutory-llm",
        default_response: str = "According to [BNS s.103], whoever commits murder shall be punished with death or imprisonment for life.",
        response_queue: Optional[List[str]] = None,
        simulate_timeout: bool = False,
        simulate_error: Optional[str] = None,
        prompt_tokens: int = 150,
        completion_tokens: int = 50
    ):
        super().__init__(model=model, temperature=0.0, max_tokens=1024, timeout=5.0, max_retries=1)
        self.default_response = default_response
        self.response_queue: List[str] = response_queue or []
        self.simulate_timeout = simulate_timeout
        self.simulate_error = simulate_error
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_history: List[List[LLMMessage]] = []

    def set_responses(self, responses: List[str]):
        """Queue a list of deterministic responses to return sequentially."""
        self.response_queue = list(responses)

    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        self.call_history.append(messages)

        if self.simulate_timeout:
            raise LLMTimeoutError("Mock provider simulated timeout")

        if self.simulate_error:
            raise LLMProviderError(self.simulate_error)

        if self.response_queue:
            content = self.response_queue.pop(0)
        else:
            content = self.default_response

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            latency_ms=1.5,
            raw_response={"mock": True}
        )

    def stream(self, messages: List[LLMMessage], **kwargs) -> Generator[str, None, None]:
        resp = self.generate(messages, **kwargs)
        # Yield words
        for word in resp.content.split(" "):
            yield word + " "


def get_llm_provider(
    provider_type: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None
) -> LLMProvider:
    """Factory creating an LLM provider based on settings or explicit overrides."""
    p_type = (provider_type or settings.llm_provider).lower().strip()
    m_name = model or settings.llm_model
    b_url = base_url or settings.llm_base_url
    temp = temperature if temperature is not None else settings.llm_temperature
    max_t = max_tokens if max_tokens is not None else settings.llm_max_tokens
    t_out = timeout if timeout is not None else settings.llm_timeout

    if p_type in ("ollama", "local"):
        return OllamaProvider(
            model=m_name,
            base_url=b_url,
            temperature=temp,
            max_tokens=max_t,
            timeout=t_out
        )
    elif p_type in ("openai", "openai_compatible", "groq", "hosted"):
        return OpenAICompatibleProvider(
            model=m_name,
            base_url=b_url,
            api_key=api_key or settings.llm_api_key,
            temperature=temp,
            max_tokens=max_t,
            timeout=t_out
        )
    elif p_type in ("mock", "test"):
        return MockLLMProvider(model=m_name)
    else:
        raise LLMConfigurationError(f"Unsupported LLM provider type: '{p_type}'. Must be 'ollama', 'openai', or 'mock'.")
