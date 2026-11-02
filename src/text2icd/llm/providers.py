"""LLM providers behind one small interface, and the endpoint allow-list for restricted data
(ADR 0003).

A provider answers one constrained question: given a system prompt, a user prompt and a JSON
schema, it returns the JSON object the model produced and the call's token usage. Nothing else of
the response is kept, and no input text is logged.

Restricted data (MIMIC-IV) may go only to an endpoint on `ALLOW_LIST`, and only while
`TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT` is on. Each entry cites the basis for its approval; the
list is reviewed whenever PhysioNet updates its guidance.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from text2icd.llm.cost import Usage

PHYSIONET_GUIDANCE = (
    "PhysioNet, 'Responsible use of MIMIC data with online services like GPT' (2023-04-18), "
    "https://physionet.org/news/post/gpt-responsible-use"
)
ALLOW_LIST: dict[str, str] = {
    "anthropic": (
        f"{PHYSIONET_GUIDANCE}: Claude does not use prompts or responses to train its models by "
        "default, and routine human review of data is not performed. Checked 2026-11-01."
    ),
}


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

    def ask(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> Answer: ...


class AnthropicModel:
    """Claude through the Anthropic API, with a JSON-schema constrained answer.

    The system prompt carries a cache breakpoint, so repeated calls read it from the prompt cache
    when it is long enough to be cached. Thinking is turned off when `thinking` is False, which
    only some models accept; `effort` bounds the work of the others."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        effort: str = "low",
        thinking: bool = False,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.thinking = thinking
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "the Anthropic provider needs the optional dependency: uv sync --extra llm"
            ) from exc
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except anthropic.AnthropicError as exc:
            raise LLMUnavailableError(f"cannot create the Anthropic client: {exc}") from exc

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


PROVIDERS = ("anthropic",)
# Models that accept `thinking: disabled` (the others always think; effort bounds them).
CAN_DISABLE_THINKING = frozenset({"claude-haiku-5-5"})


def choice_model(provider: str, model: str, *, api_key: str | None, effort: str) -> ChoiceModel:
    if provider == "anthropic":
        return AnthropicModel(
            model, api_key=api_key, effort=effort, thinking=model not in CAN_DISABLE_THINKING
        )
    known = ", ".join(PROVIDERS)
    raise LLMUnavailableError(
        f"unknown LLM provider {provider!r} (known: {known}); set TEXT2ICD_LLM_PROVIDER"
    )
