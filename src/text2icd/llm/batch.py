"""Offline evaluation through the Message Batches API (task 8.8), on the Anthropic API only.

A batch run answers L3's decisions in rounds, at the batch price (`batch_discount` in the price
table, half the price on the Anthropic API):

1. Every phrase is coded with a `Collector` navigator, which answers from the decision cache and
   records the first decision of each phrase it cannot answer.
2. The recorded decisions go out as one Message Batch, keyed by their cache key (`custom_id`);
   the answers are read when the batch ends, checked against the options, stored in the cache,
   and the batch is deleted from the provider.
3. Repeat, since an answer can raise the phrase's next decision (an abbreviation's meaning, then
   the Index entry, then the Tabular child), up to the calls allowed per phrase.

Then the run codes every phrase with a `Replay` navigator, which takes each answer from the
cache and charges the phrase that first uses a batch answer with its cost.

The budget holds as in a direct run: before each batch, requests are sent only while an upper
bound of their cost (every prompt token uncached, the full `max_tokens`) fits in what is left.
"""

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from text2icd.llm.cache import DecisionCache, cache_key
from text2icd.llm.cost import CostMeter, Usage
from text2icd.llm.decisions import NONE, Choice, Decision
from text2icd.llm.prompts import SYSTEM_PROMPT, choice_schema, decision_prompt, max_tokens
from text2icd.llm.providers import MessagesModel, read_answer

POLL_S = 60.0  # between status checks; most batches end within the hour
WAIT_S = 24 * 3600.0  # the longest a batch may run
CHARS_PER_TOKEN = 2  # a safe floor, for the cost bound: no prompt has fewer characters a token


class BatchError(RuntimeError):
    """A batch that did not end in time, or a provider without batches."""


@dataclass
class BatchModel:
    """Sends decisions as one Message Batch and reads the answers when it ends."""

    model: MessagesModel
    meter: CostMeter  # with batch=True: the batch price
    cache: DecisionCache
    space: str  # the cache namespace (`cache.namespace`)
    poll_s: float = POLL_S
    wait_s: float = WAIT_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    batches: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model.provider != "anthropic":
            raise BatchError(
                f"Message Batches are offered on the Anthropic API only, not on "
                f"{self.model.provider}: run without -O llm=batch"
            )

    def run(self, decisions: Mapping[str, Decision]) -> dict[str, Choice]:
        """The answers by cache key; decisions over the budget get a `limit` choice."""
        requests, out = [], {}
        left = self.meter.budget_usd - self.meter.spent_usd
        for key, decision in decisions.items():
            params = self._params(decision)
            bound = self._bound(params)
            if bound > left:
                out[key] = Choice(by="limit", reason=f"budget of ${self.meter.budget_usd:.2f}")
                continue
            left -= bound
            requests.append({"custom_id": key, "params": params})
        if not requests:
            return out
        batches = self.model.client.messages.batches
        batch = batches.create(requests=requests)
        self.batches.append(batch.id)
        self._wait(batch.id)
        for entry in batches.results(batch.id):
            out[entry.custom_id] = self._read(entry, decisions[entry.custom_id])
        batches.delete(batch.id)  # the results stay on the provider otherwise
        return out

    def _params(self, decision: Decision) -> dict[str, Any]:
        return self.model.request(
            system=SYSTEM_PROMPT,
            prompt=decision_prompt(decision),
            schema=choice_schema(decision),
            max_tokens=max_tokens(self.model.thinking),
        )

    def _bound(self, params: dict[str, Any]) -> float:
        prompt = len(SYSTEM_PROMPT) + len(params["messages"][0]["content"])
        usage = Usage(input_tokens=prompt // CHARS_PER_TOKEN, output_tokens=params["max_tokens"])
        return self.meter.price.cost(usage, batch=True)

    def _wait(self, batch_id: str) -> None:
        batches = self.model.client.messages.batches
        started = self.clock()
        while batches.retrieve(batch_id).processing_status != "ended":
            if self.clock() - started > self.wait_s:
                batches.cancel(batch_id)
                raise BatchError(f"batch {batch_id} did not end in {self.wait_s:.0f} s: canceled")
            self.sleep(self.poll_s)

    def _read(self, entry: Any, decision: Decision) -> Choice:
        result = entry.result
        if result.type != "succeeded":
            return Choice(by="error", reason=f"batch request {result.type}")
        answer = read_answer(result.message)
        spent = {
            "calls": 1,
            "cost_usd": self.meter.charge(answer.usage),
            "usage": answer.usage.as_dict(),
        }
        if answer.data is None:
            return Choice(by="error", reason=answer.error, **spent)
        chosen = answer.data.get("choice")
        option = chosen if isinstance(chosen, str) and decision.option(chosen) else None
        reason = None if option else ("no option fits" if chosen == NONE else "not an option")
        self.cache.put_key(entry.custom_id, option)
        return Choice(option=option, by="llm", reason=reason, **spent)


class Collector:
    """A round's navigator: cached answers, else the decision is recorded for the batch and the
    phrase asks nothing more this round (its next decision depends on this answer)."""

    def __init__(self, cache: DecisionCache, space: str, provider: str, model: str) -> None:
        self.cache = cache
        self.space = space
        self.provider = provider
        self.model = model
        self.pending: dict[str, Decision] = {}
        self._waiting = False

    def start(self) -> None:
        """A new phrase."""
        self._waiting = False

    def choose(self, decision: Decision, *, calls: int = 0) -> Choice:
        if self._waiting:
            return Choice(by="limit", reason="waiting for the batch")
        stored = self.cache.get(decision, self.space)
        if stored is not None:
            return stored
        self.pending.setdefault(cache_key(decision, self.space), decision)
        self._waiting = True
        return Choice(by="limit", reason="waiting for the batch")

    def close(self) -> None:
        pass


class Replay:
    """The final pass's navigator: every answer from the cache. A phrase that first uses an
    answer of this run's batches is charged its call and cost; repeats are free."""

    def __init__(
        self,
        cache: DecisionCache,
        space: str,
        provider: str,
        model: str,
        answered: Mapping[str, Choice],
        calls_per_phrase: int,
    ) -> None:
        self.cache = cache
        self.space = space
        self.provider = provider
        self.model = model
        self.unpaid = dict(answered)
        self.calls_per_phrase = calls_per_phrase

    def choose(self, decision: Decision, *, calls: int = 0) -> Choice:
        key = cache_key(decision, self.space)
        if key in self.unpaid:
            return self.unpaid.pop(key)
        stored = self.cache.get(decision, self.space)
        if stored is not None:
            return stored
        if calls >= self.calls_per_phrase:
            return Choice(by="limit", reason=f"{self.calls_per_phrase} calls for this phrase")
        return Choice(by="limit", reason="no batch answer (budget or calls per phrase)")

    def close(self) -> None:
        pass


def answer_in_rounds(
    code: Callable[[str], object],
    phrases: Iterable[str],
    collector: Collector,
    batch: BatchModel,
    rounds: int,
) -> dict[str, Choice]:
    """Fill the cache with batch answers for the phrases' decisions, at most `rounds` batches;
    returns the answers this run paid for, by cache key."""
    distinct: Sequence[str] = list(dict.fromkeys(phrases))
    answered: dict[str, Choice] = {}
    for _ in range(rounds):
        collector.pending.clear()
        for phrase in distinct:
            collector.start()
            code(phrase)
        if not collector.pending:
            break
        answers = batch.run(collector.pending)
        paid = {key: choice for key, choice in answers.items() if choice.calls}
        answered.update(paid)
        if not any(choice.by == "llm" for choice in answers.values()):
            break  # nothing answered (budget, errors): another round would ask the same
    return answered
