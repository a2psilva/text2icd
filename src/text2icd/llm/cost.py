"""Cost accounting for LLM calls: the usage fields of each response times the price table in
`resources/prices.toml`, accumulated per run against a budget (ADR 0003).
"""

import tomllib
from dataclasses import dataclass, field, fields
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

MILLION = 1_000_000
PRICES = "prices.toml"
WARM_UP = 3  # calls after which a cached system prompt must have been read from the cache


class PriceError(ValueError):
    """A price table that does not load, or a model with no price."""


class BudgetExceededError(RuntimeError):
    """The run's budget is spent; no more calls may be made."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts of one or more calls, as the provider reports them. `input_tokens` excludes
    the tokens read from or written to the prompt cache."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_write_1h_tokens: int = 0  # the part of the cache writes with a 1-hour lifetime

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    def as_dict(self) -> dict[str, int]:
        """The non-zero counts, as stored with each prediction."""
        return {f.name: value for f in fields(self) if (value := getattr(self, f.name))}

    @classmethod
    def from_response(cls, usage: Any) -> "Usage":
        """From a response's `usage`, an object or a dict; missing fields count as 0."""

        def read(source: Any, name: str) -> int:
            value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
            return int(value or 0)

        creation = (
            usage.get("cache_creation")
            if isinstance(usage, dict)
            else getattr(usage, "cache_creation", None)
        )
        return cls(
            input_tokens=read(usage, "input_tokens"),
            output_tokens=read(usage, "output_tokens"),
            cache_read_input_tokens=read(usage, "cache_read_input_tokens"),
            cache_creation_input_tokens=read(usage, "cache_creation_input_tokens"),
            cache_write_1h_tokens=read(creation, "ephemeral_1h_input_tokens") if creation else 0,
        )


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens for one model on one provider."""

    provider: str
    id: str
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float
    long_context_above: int | None = None
    long_input: float | None = None
    long_output: float | None = None
    batch_discount: float = 1.0

    def cost(self, usage: Usage, *, batch: bool = False) -> float:
        """The cost in US dollars of the calls `usage` sums up."""
        long = (
            self.long_context_above is not None
            and self.long_input is not None
            and self.long_output is not None
            and usage.prompt_tokens > self.long_context_above
        )
        input_price = self.long_input if long and self.long_input else self.input
        output_price = self.long_output if long and self.long_output else self.output
        scale = input_price / self.input if self.input else 1.0
        writes_1h = min(usage.cache_write_1h_tokens, usage.cache_creation_input_tokens)
        writes_5m = usage.cache_creation_input_tokens - writes_1h
        total = (
            usage.input_tokens * input_price
            + usage.output_tokens * output_price
            + usage.cache_read_input_tokens * self.cache_read * scale
            + writes_5m * self.cache_write_5m * scale
            + writes_1h * self.cache_write_1h * scale
        ) / MILLION
        return total * (self.batch_discount if batch else 1.0)


@dataclass(frozen=True, slots=True)
class PriceTable:
    checked: date
    prices: dict[tuple[str, str], Price]

    def get(self, provider: str, model: str) -> Price:
        try:
            return self.prices[(provider, model)]
        except KeyError:
            raise PriceError(
                f"no price for {model} on {provider}; add it to {PRICES} before running it"
            ) from None


def load_prices(path: Path | None = None) -> PriceTable:
    raw = (
        path.read_bytes()
        if path is not None
        else resources.files("text2icd.resources").joinpath(PRICES).read_bytes()
    )
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("schema") != 1:
        raise PriceError(f"unsupported price table schema {data.get('schema')!r}")
    checked = data.get("checked")
    if not isinstance(checked, date):
        raise PriceError("the price table needs a `checked` date")
    prices: dict[tuple[str, str], Price] = {}
    for entry in data.get("model", []):
        try:
            price = Price(**entry)
        except TypeError as exc:
            raise PriceError(f"bad price entry {entry.get('id')!r}: {exc}") from None
        numbers = [price.input, price.output, price.cache_read, price.cache_write_5m]
        numbers += [price.cache_write_1h, price.batch_discount]
        if any(not isinstance(n, int | float) or n < 0 for n in numbers):
            raise PriceError(f"prices of {price.id} must be numbers of 0 or more")
        key = (price.provider, price.id)
        if key in prices:
            raise PriceError(f"{price.id} on {price.provider} is listed twice")
        prices[key] = price
    return PriceTable(checked, prices)


@dataclass
class CostMeter:
    """Running totals of a run's calls, with a hard budget in US dollars: `charge` records a
    call, and `check` raises once the budget is spent, so no further call is made."""

    price: Price
    budget_usd: float
    batch: bool = False
    usage: Usage = field(default_factory=Usage)
    spent_usd: float = 0.0
    calls: int = 0

    def check(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"budget of ${self.budget_usd:.2f} spent (${self.spent_usd:.4f} on {self.calls} "
                "calls)"
            )

    @property
    def cache_rate(self) -> float | None:
        """The share of prompt tokens read from the prompt cache; None before any call."""
        prompt = self.usage.prompt_tokens
        return self.usage.cache_read_input_tokens / prompt if prompt else None

    def cache_warning(self, after: int = WARM_UP) -> str | None:
        """A warning when `after` calls or more read nothing from the prompt cache: the cached
        prefix is shorter than the model's minimum, or the provider does not cache it (8.7)."""
        if self.calls < after or self.usage.cache_read_input_tokens > 0:
            return None
        return (
            f"no prompt-cache reads in {self.calls} calls: the system prompt may be shorter than "
            f"the minimum cacheable prefix of {self.price.id}, so every call pays for it in full"
        )

    def charge(self, usage: Usage) -> float:
        """Record one call; returns its cost."""
        cost = self.price.cost(usage, batch=self.batch)
        self.usage = self.usage + usage
        self.spent_usd += cost
        self.calls += 1
        return cost
