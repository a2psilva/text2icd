"""The L3 navigator (task 8.1): `choose(decision) -> Choice`, behind one interface.

`LLMNavigator` asks a provider's model (`providers`) with the cached prompt and the answer schema
of `prompts`, and keeps only an answer that names one of the decision's options. `NoNavigator`
(provider `none`, the default) asks nothing, so the cascade runs without any LLM.

Limits (task 8.6): at most `calls_per_phrase` calls for one phrase (the caller passes the calls
made so far), and none once the run's budget (`CostMeter`) is spent; the provider's client bounds
each attempt with a timeout and retries only retryable errors. A limit reached, a failed call or
an unusable answer gives a choice without an option, and the coder keeps its own answer. Errors
that no later call can fix (bad credentials, no access, unknown model) stop the run instead.

Every call is charged to the meter from the response's `usage` (task 8.7); the choice carries
its calls, cost, usage and latency, so each phrase records what it spent. With a decision cache
(task 8.9), a decision answered before is answered from it, free and before any limit.
"""

import time
from typing import Protocol

from text2icd.config import Settings
from text2icd.llm.cache import FILE, DecisionCache, namespace
from text2icd.llm.cost import BudgetExceededError, CostMeter, load_prices
from text2icd.llm.decisions import NONE, Choice, Decision
from text2icd.llm.prompts import SYSTEM_PROMPT, choice_schema, decision_prompt, max_tokens
from text2icd.llm.providers import (
    ChoiceModel,
    LLMUnavailableError,
    bedrock_model_id,
    choice_model,
    fatal,
)

CALLS_PER_PHRASE = 4


class Navigator(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def choose(self, decision: Decision, *, calls: int = 0) -> Choice:
        """The choice for `decision`; `calls`, the calls already made for this phrase."""
        ...

    def close(self) -> None: ...


class NoNavigator:
    """No LLM: every decision keeps the coder's own answer."""

    provider = "none"
    model = ""

    def choose(self, decision: Decision, *, calls: int = 0) -> Choice:
        return Choice(by="off", reason="no LLM provider (TEXT2ICD_LLM_PROVIDER=none)")

    def close(self) -> None:
        pass


class LLMNavigator:
    def __init__(
        self,
        model: ChoiceModel,
        meter: CostMeter,
        *,
        calls_per_phrase: int = CALLS_PER_PHRASE,
        cache: DecisionCache | None = None,
    ) -> None:
        self.chooser = model
        self.meter = meter
        self.calls_per_phrase = calls_per_phrase
        self.cache = cache
        self.space = namespace(model.provider, model.model, getattr(model, "effort", ""))

    @property
    def provider(self) -> str:
        return self.chooser.provider

    @property
    def model(self) -> str:
        return self.chooser.model

    def choose(self, decision: Decision, *, calls: int = 0) -> Choice:
        if self.cache is not None and (stored := self.cache.get(decision, self.space)):
            return stored
        if calls >= self.calls_per_phrase:
            return Choice(by="limit", reason=f"{self.calls_per_phrase} calls for this phrase")
        try:
            self.meter.check()
        except BudgetExceededError as exc:
            return Choice(by="limit", reason=str(exc))
        started = time.perf_counter()
        try:
            answer = self.chooser.ask(
                system=SYSTEM_PROMPT,
                prompt=decision_prompt(decision),
                schema=choice_schema(decision),
                max_tokens=max_tokens(self.chooser.thinking),
            )
        except Exception as exc:  # after the client's own retries
            status = getattr(exc, "status_code", None)
            reason = type(exc).__name__ + (f" {status}" if status is not None else "")
            if fatal(exc):
                raise LLMUnavailableError(f"{self.provider} refused the call: {reason}") from exc
            return Choice(by="error", reason=reason, calls=1, latency_ms=_since(started))
        spent = {
            "calls": 1,
            "cost_usd": self.meter.charge(answer.usage),
            "usage": answer.usage.as_dict(),
            "latency_ms": _since(started),
        }
        if answer.data is None:
            return Choice(by="error", reason=answer.error, **spent)
        chosen = answer.data.get("choice")
        if chosen == NONE:
            choice = Choice(by="llm", reason="no option fits", **spent)
        elif not isinstance(chosen, str) or decision.option(chosen) is None:
            choice = Choice(by="llm", reason="the answer is not one of the options", **spent)
        else:
            choice = Choice(option=chosen, by="llm", **spent)
        if self.cache is not None:
            self.cache.put(decision, self.space, choice.option)
        return choice

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()


def provider_model(settings: Settings, **overrides: str | None) -> ChoiceModel:
    """The settings' provider and model (`TEXT2ICD_LLM_*`), or the ones `overrides` names."""
    provider, model = _named(settings, overrides)
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return choice_model(
        provider,
        model,
        api_key=key,
        effort=overrides.get("effort") or "low",
        region=settings.llm_region,
        project=settings.llm_project,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_navigator(
    settings: Settings,
    *,
    budget: float | None = None,
    cache: bool = True,
    **overrides: str | None,
) -> Navigator:
    """The navigator the settings describe: `NoNavigator` for provider `none`, else the model's,
    with its price (`PriceError` when the price table has none), the run's limits and, unless
    `cache` is False, the decision cache in the settings' cache directory."""
    if (overrides.get("provider") or settings.llm_provider) == "none":
        return NoNavigator()
    provider, model = _named(settings, overrides)
    priced = bedrock_model_id(model) if provider == "bedrock" else model
    price = load_prices().get(provider, priced)  # before any client is made
    chooser = provider_model(settings, **overrides)
    meter = CostMeter(price, settings.llm_budget_usd if budget is None else budget)
    return LLMNavigator(
        chooser,
        meter,
        calls_per_phrase=settings.llm_max_calls_per_phrase,
        cache=DecisionCache(settings.cache_dir / FILE) if cache else None,
    )


def _named(settings: Settings, overrides: dict[str, str | None]) -> tuple[str, str]:
    provider = overrides.get("provider") or settings.llm_provider
    model = overrides.get("model") or settings.llm_model
    if provider == "none" or not model:
        raise LLMUnavailableError(
            "an LLM needs a provider and a model: set TEXT2ICD_LLM_PROVIDER and "
            "TEXT2ICD_LLM_MODEL, or pass -O provider=... -O model=..."
        )
    return provider, model


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
