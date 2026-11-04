"""Provider clients and the endpoint policy, with stand-in SDK clients: no network call."""

import sys
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from text2icd.llm import providers
from text2icd.llm.providers import (
    ALLOW_LIST,
    PROVIDERS,
    BedrockModel,
    EndpointNotAllowedError,
    LLMUnavailableError,
    VertexModel,
    bedrock_model_id,
    check_endpoint,
    choice_model,
    fatal,
    thinks,
)


class Messages:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        usage = SimpleNamespace(input_tokens=40, output_tokens=6)
        content = [SimpleNamespace(type="text", text='{"choice": "2"}')]
        return SimpleNamespace(content=content, stop_reason="end_turn", usage=usage)


def test_endpoint_policy() -> None:
    assert PROVIDERS == ("anthropic", "bedrock", "vertex")
    for provider in ("anthropic", "bedrock"):
        check_endpoint(provider, restricted=True, require_compliant=True)
        assert "physionet.org" in ALLOW_LIST[provider]
        assert "Checked 2026-11-04" in (ALLOW_LIST[provider])
    # PhysioNet names Vertex AI for Gemini only: Claude on Vertex AI is not approved.
    with pytest.raises(EndpointNotAllowedError, match="'vertex' is not approved"):
        check_endpoint("vertex", restricted=True, require_compliant=True)
    with pytest.raises(EndpointNotAllowedError, match="REQUIRE_DUA_COMPLIANT_ENDPOINT=true"):
        check_endpoint("bedrock", restricted=True, require_compliant=False)
    check_endpoint("vertex", restricted=False, require_compliant=True)  # public data: any


def test_bedrock_and_vertex_requests() -> None:
    """Both platforms take the Anthropic API's request: cached system prompt, schema, effort."""
    bedrock = BedrockModel("claude-haiku-5-5", client=SimpleNamespace(messages=Messages()))
    vertex = VertexModel("claude-haiku-5-5", client=SimpleNamespace(messages=Messages()))
    assert (bedrock.provider, bedrock.model) == ("bedrock", "anthropic.claude-haiku-5-5")
    assert (vertex.provider, vertex.model) == ("vertex", "claude-haiku-5-5")
    for model in (bedrock, vertex):
        answer = model.ask(system="rules", prompt="phrase", schema={"type": "object"}, max_tokens=9)
        assert answer.data == {"choice": "2"}
        assert answer.usage.input_tokens == 40
        (request,) = model._client.messages.requests
        assert request["model"] == model.model
        assert request["thinking"] == {"type": "disabled"}
        assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert request["output_config"] == {
            "effort": "low",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        }


def test_model_ids_and_thinking() -> None:
    assert bedrock_model_id("claude-haiku-5-5") == "anthropic.claude-haiku-5-5"
    assert bedrock_model_id("anthropic.claude-haiku-5-5") == "anthropic.claude-haiku-5-5"
    assert not thinks("claude-haiku-5-5")
    assert not thinks("anthropic.claude-haiku-5-5")
    assert thinks("claude-sonnet-5-5")
    assert thinks("anthropic.claude-opus-5-5")


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def factory(self, name: str) -> Any:
        def make(**options: Any) -> Any:
            self.calls.append((name, options))
            return SimpleNamespace(messages=Messages())

        return make


def test_choice_model_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each provider builds its SDK client with the run's timeout and retry limits."""
    recorder = Recorder()
    for name in ("Anthropic", "AnthropicBedrockMantle", "AnthropicVertex"):
        monkeypatch.setattr(anthropic, name, recorder.factory(name))
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    limits = {"timeout": 12.0, "max_retries": 1}
    first = choice_model("anthropic", "claude-haiku-5-5", api_key="k", **limits)
    second = choice_model("bedrock", "claude-sonnet-5-5", region="eu-west-1", **limits)
    third = choice_model("vertex", "claude-haiku-5-5", project="proj", **limits)
    choice_model("vertex", "claude-haiku-5-5", region="us", **limits)
    monkeypatch.setenv("CLOUD_ML_REGION", "eu")
    choice_model("vertex", "claude-haiku-5-5", **limits)
    assert recorder.calls == [
        ("Anthropic", {"api_key": "k", **limits}),
        ("AnthropicBedrockMantle", {"aws_region": "eu-west-1", **limits}),
        ("AnthropicVertex", {"region": "global", "project_id": "proj", **limits}),
        ("AnthropicVertex", {"region": "us", **limits}),
        ("AnthropicVertex", {"region": "eu", **limits}),
    ]
    assert [m.provider for m in (first, second, third)] == ["anthropic", "bedrock", "vertex"]
    assert (second.model, second.thinking) == ("anthropic.claude-sonnet-5-5", True)
    assert third.thinking is False


def test_unavailable_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_region(**options: Any) -> Any:
        raise ValueError("No AWS region or base URL found")

    def no_boto(**options: Any) -> Any:
        raise ModuleNotFoundError("No module named 'boto3'")

    monkeypatch.setattr(anthropic, "AnthropicBedrockMantle", no_region)
    with pytest.raises(LLMUnavailableError, match="cannot create the Bedrock client: No AWS"):
        BedrockModel("claude-haiku-5-5")
    monkeypatch.setattr(anthropic, "AnthropicBedrockMantle", no_boto)
    with pytest.raises(LLMUnavailableError, match="Bedrock client needs its extra"):
        BedrockModel("claude-haiku-5-5")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # the SDK is not installed
    with pytest.raises(LLMUnavailableError, match="uv sync --extra vertex"):
        VertexModel("claude-haiku-5-5")
    with pytest.raises(LLMUnavailableError, match="unknown LLM provider 'azure'"):
        providers.choice_model("azure", "m")


def test_fatal_errors() -> None:
    def status(code: int) -> Exception:
        error = Exception("x")
        error.status_code = code  # type: ignore[attr-defined]
        return error

    assert [fatal(status(code)) for code in (401, 403, 404, 400, 429, 500, 529)] == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert not fatal(ConnectionError())
