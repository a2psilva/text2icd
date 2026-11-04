"""LLM providers behind one small interface, and the endpoint allow-list for restricted data
(ADR 0003).

A provider answers one constrained question: given a system prompt, a user prompt and a JSON
schema, it returns the JSON object the model produced and the call's token usage. Nothing else of
the response is kept, and no input text is logged.

Three platforms serve Claude through the same Messages API, each with the official `anthropic`
SDK client: the Anthropic API (`anthropic`), Amazon Bedrock (`bedrock`, `AnthropicBedrockMantle`,
model ids with an `anthropic.` prefix) and Google Cloud Vertex AI (`vertex`, `AnthropicVertex`).
The SDK retries connection errors, timeouts (408), conflicts (409), rate limits (429) and server
errors (5xx) up to `max_retries` times with exponential backoff, and nothing else; `timeout`
bounds each attempt.

Restricted data (MIMIC-IV) may go only to an endpoint on `ALLOW_LIST`, and only while
`TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT` is on. Each entry cites the basis for its approval; the
list is reviewed whenever PhysioNet updates its guidance.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from text2icd.llm.cost import Usage

PHYSIONET_GUIDANCE = (
    "PhysioNet, 'Responsible use of MIMIC data with online services like GPT' (2023-04-18), "
    "https://physionet.org/news/post/gpt-responsible-use"
)
# Google Cloud Vertex AI is not listed: the guidance names it for Google's Gemini models only, so
# Claude on Vertex AI needs its own check before it may receive restricted data.
ALLOW_LIST: dict[str, str] = {
    "anthropic": (
        f"{PHYSIONET_GUIDANCE}: Claude does not use prompts or responses to train its models by "
        "default, and routine human review of data is not performed. Checked 2026-11-04."
    ),
    "bedrock": (
        f"{PHYSIONET_GUIDANCE}: Amazon Bedrock is named as an acceptable service; data is not "
        "shared back to the base model for training. Claude in Amazon Bedrock runs on AWS-managed "
        "infrastructure with no Anthropic operator access, under Amazon Bedrock's data protection "
        "terms. Checked 2026-11-04."
    ),
}
PROVIDERS = ("anthropic", "bedrock", "vertex")
# Models that accept `thinking: disabled` (the others always think; effort bounds them).
CAN_DISABLE_THINKING = frozenset({"claude-haiku-5-5"})
BEDROCK_PREFIX = "anthropic."
VERTEX_REGION = "global"  # the endpoint Google recommends, with no regional price premium
TIMEOUT_S = 30.0
MAX_RETRIES = 2


class LLMUnavailableError(RuntimeError):
    """The provider is unknown, its SDK is not installed, or it has no credentials."""


class EndpointNotAllowedError(PermissionError):
    """Restricted data would be sent to an endpoint that is not approved for it."""


def check_endpoint(provider: str, *, restricted: bool, require_compliant: bool) -> None:
    """Refuse to send restricted data unless the policy is on and the provider is approved."""
    if not restricted:
        return
    if not require_compliant:
        raise EndpointNotAllowedError(
            "this dataset is restricted (MIMIC-IV): LLM runs on it need "
            "TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT=true"
        )
    if provider not in ALLOW_LIST:
        approved = ", ".join(sorted(ALLOW_LIST))
        raise EndpointNotAllowedError(
            f"provider {provider!r} is not approved for restricted data (approved: {approved}); "
            "see docs/data-and-compliance.md"
        )


@dataclass(frozen=True, slots=True)
class Answer:
    data: dict[str, Any] | None  # the JSON object, or None when there is none
    usage: Usage
    stop_reason: str
    error: str | None = None


class ChoiceModel(Protocol):
    provider: str
    model: str
    thinking: bool  # whether the model reasons before it answers (its output needs more room)

    def ask(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> Answer: ...


def fatal(exc: BaseException) -> bool:
    """An error no retry or later call can fix: bad credentials, no access, unknown model."""
    return getattr(exc, "status_code", None) in (401, 403, 404)


class MessagesModel:
    """Claude through a Messages API client, with a JSON-schema constrained answer.

    The system prompt carries a cache breakpoint, so repeated calls read it from the prompt cache
    when it is long enough to be cached. Thinking is turned off when `thinking` is False, which
    only some models accept; `effort` bounds the work of the others."""

    provider = "anthropic"

    def __init__(self, model: str, client: Any, *, effort: str = "low", thinking: bool) -> None:
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self._client = client

    def ask(self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> Answer:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if not self.thinking:
            request["thinking"] = {"type": "disabled"}
        response = self._client.messages.create(**request)
        usage = Usage.from_response(response.usage)
        stop = str(response.stop_reason)
        if stop == "refusal":
            return Answer(None, usage, stop, "the model refused")
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if stop == "max_tokens" or text is None:
            return Answer(None, usage, stop, f"no complete answer (stop reason {stop})")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return Answer(None, usage, stop, "the answer is not JSON")
        if not isinstance(data, dict):
            return Answer(None, usage, stop, "the answer is not a JSON object")
        return Answer(data, usage, stop)


class AnthropicModel(MessagesModel):
    """Claude through the Anthropic API (`anthropic.Anthropic`)."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        effort: str = "low",
        thinking: bool = False,
        timeout: float = TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
        client: Any = None,
    ) -> None:
        if client is None:
            anthropic = _sdk("the Anthropic provider", "llm")
            client = _client(
                lambda: anthropic.Anthropic(
                    api_key=api_key, timeout=timeout, max_retries=max_retries
                ),
                "Anthropic",
            )
        super().__init__(model, client, effort=effort, thinking=thinking)


class BedrockModel(MessagesModel):
    """Claude in Amazon Bedrock (`anthropic.AnthropicBedrockMantle`). The model id takes the
    `anthropic.` prefix Bedrock uses ("anthropic.claude-haiku-5-5"); AWS credentials and the
    region come from the standard AWS chain unless `region` is given."""

    provider = "bedrock"

    def __init__(
        self,
        model: str,
        *,
        region: str | None = None,
        effort: str = "low",
        thinking: bool = False,
        timeout: float = TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
        client: Any = None,
    ) -> None:
        if client is None:
            anthropic = _sdk("the Bedrock provider", "bedrock")
            client = _client(
                lambda: anthropic.AnthropicBedrockMantle(
                    aws_region=region, timeout=timeout, max_retries=max_retries
                ),
                "Bedrock",
            )
        super().__init__(bedrock_model_id(model), client, effort=effort, thinking=thinking)


class VertexModel(MessagesModel):
    """Claude on Google Cloud Vertex AI (`anthropic.AnthropicVertex`), with Application Default
    Credentials. The region is `region`, else `CLOUD_ML_REGION`, else the global endpoint."""

    provider = "vertex"

    def __init__(
        self,
        model: str,
        *,
        project: str | None = None,
        region: str | None = None,
        effort: str = "low",
        thinking: bool = False,
        timeout: float = TIMEOUT_S,
        max_retries: int = MAX_RETRIES,
        client: Any = None,
    ) -> None:
        if client is None:
            anthropic = _sdk("the Vertex AI provider", "vertex")
            where = region or os.environ.get("CLOUD_ML_REGION") or VERTEX_REGION
            options: dict[str, Any] = {"region": where}
            if project:
                options["project_id"] = project
            client = _client(
                lambda: anthropic.AnthropicVertex(
                    **options, timeout=timeout, max_retries=max_retries
                ),
                "Vertex AI",
            )
        super().__init__(model, client, effort=effort, thinking=thinking)


def bedrock_model_id(model: str) -> str:
    return model if model.startswith(BEDROCK_PREFIX) else BEDROCK_PREFIX + model


def thinks(model: str) -> bool:
    """Whether the model reasons before it answers, even when asked not to."""
    return model.removeprefix(BEDROCK_PREFIX) not in CAN_DISABLE_THINKING


def choice_model(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    effort: str = "low",
    region: str | None = None,
    project: str | None = None,
    timeout: float = TIMEOUT_S,
    max_retries: int = MAX_RETRIES,
) -> ChoiceModel:
    limits: dict[str, Any] = {"effort": effort, "timeout": timeout, "max_retries": max_retries}
    if provider == "anthropic":
        return AnthropicModel(model, api_key=api_key, thinking=thinks(model), **limits)
    if provider == "bedrock":
        return BedrockModel(model, region=region, thinking=thinks(model), **limits)
    if provider == "vertex":
        return VertexModel(model, project=project, region=region, thinking=thinks(model), **limits)
    known = ", ".join(PROVIDERS)
    raise LLMUnavailableError(
        f"unknown LLM provider {provider!r} (known: {known}); set TEXT2ICD_LLM_PROVIDER"
    )


def _sdk(what: str, extra: str) -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailableError(
            f"{what} needs the optional dependency: uv sync --extra {extra}"
        ) from exc
    return anthropic


def _client(make: Any, platform: str) -> Any:
    try:
        return make()
    except ImportError as exc:  # the platform's own dependencies (boto3, google-auth)
        raise LLMUnavailableError(f"the {platform} client needs its extra: {exc}") from exc
    except Exception as exc:  # no region, no credentials
        raise LLMUnavailableError(f"cannot create the {platform} client: {exc}") from exc
