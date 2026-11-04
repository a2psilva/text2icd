"""The Message Batches path (task 8.8) with a stand-in Batches API: no network call is made."""

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from text2icd.llm.batch import BatchError, BatchModel, Collector, Replay
from text2icd.llm.cache import FILE, DecisionCache, cache_key, namespace
from text2icd.llm.cost import CostMeter, Usage, load_prices
from text2icd.llm.decisions import Choice, Decision, Offer
from text2icd.llm.providers import AnthropicModel, BedrockModel

PRICE = load_prices().get("anthropic", "claude-haiku-5-5")
SPACE = namespace("anthropic", "claude-haiku-5-5", "low")
USAGE = {"input_tokens": 200, "output_tokens": 8, "cache_read_input_tokens": 900}


def message(text: str) -> Any:
    usage = SimpleNamespace(**USAGE, cache_creation_input_tokens=0, cache_creation=None)
    content = [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn", usage=usage)


class Batches:
    """Ends every batch on the second status check; `answer` gives each request's reply."""

    def __init__(self, answer: Callable[[dict[str, Any]], Any]) -> None:
        self.answer = answer
        self.created: list[list[dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.canceled: list[str] = []
        self.checks = 0

    def create(self, *, requests: list[dict[str, Any]]) -> Any:
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch-{len(self.created)}")

    def retrieve(self, batch_id: str) -> Any:
        self.checks += 1
        return SimpleNamespace(processing_status="ended" if self.checks % 2 == 0 else "in_progress")

    def results(self, batch_id: str) -> Iterator[Any]:
        for request in self.created[int(batch_id.split("-")[1]) - 1]:
            reply = self.answer(request)
            result = (
                SimpleNamespace(type="succeeded", message=message(reply))
                if isinstance(reply, str)
                else SimpleNamespace(type=reply["type"])
            )
            yield SimpleNamespace(custom_id=request["custom_id"], result=result)

    def delete(self, batch_id: str) -> None:
        self.deleted.append(batch_id)

    def cancel(self, batch_id: str) -> None:
        self.canceled.append(batch_id)


def decision(phrase: str) -> Decision:
    return Decision.of("subterm", phrase, [Offer("Amyloid heart", "E85.4"), Offer("Pyelitis")])


def batch_model(batches: Batches, cache: DecisionCache, budget: float = 1.0) -> BatchModel:
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    model = AnthropicModel("claude-haiku-5-5", client=client)
    sleeps: list[float] = []
    return BatchModel(
        model, CostMeter(PRICE, budget, batch=True), cache, SPACE, sleep=sleeps.append
    )


def test_batch_run(tmp_path: Path) -> None:
    replies = {"p1": '{"choice": "1"}', "p2": '{"choice": "none"}', "p3": '{"choice": "Z99"}'}

    def answer(request: dict[str, Any]) -> Any:
        prompt = request["params"]["messages"][0]["content"]
        phrase = prompt.split("</phrase>")[0].removeprefix("<phrase>")
        return replies.get(phrase, {"type": "errored"})

    batches = Batches(answer)
    with DecisionCache(tmp_path / FILE) as cache:
        model = batch_model(batches, cache)
        decisions = {cache_key(decision(p), SPACE): decision(p) for p in ("p1", "p2", "p3", "p4")}
        answers = model.run(decisions)
        (requests,) = batches.created
        assert [r["custom_id"] for r in requests] == list(decisions)
        params = requests[0]["params"]
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert params["output_config"]["format"]["schema"]["properties"]["choice"]["enum"] == [
            "1",
            "2",
            "none",
        ]
        assert params["thinking"] == {"type": "disabled"}
        got = [(c.option, c.by, c.reason, c.calls) for c in answers.values()]
        assert got == [
            ("1", "llm", None, 1),
            (None, "llm", "no option fits", 1),
            (None, "llm", "not an option", 1),
            (None, "error", "batch request errored", 0),
        ]
        cost = PRICE.cost(Usage(**USAGE), batch=True)
        assert answers[next(iter(decisions))].cost_usd == pytest.approx(cost)
        assert cost == pytest.approx(PRICE.cost(Usage(**USAGE)) / 2)  # half price
        assert model.meter.calls == 3
        assert batches.deleted == ["batch-1"]  # results do not stay on the provider
        assert batches.checks == 2
        assert len(cache) == 3  # answered ones, none included; not the error
        assert cache.get(decision("p1"), SPACE) == Choice(option="1", by="cache")


def test_batch_budget_and_timeout(tmp_path: Path) -> None:
    batches = Batches(lambda request: '{"choice": "1"}')
    with DecisionCache(tmp_path / FILE) as cache:
        poor = batch_model(batches, cache, budget=0.0)
        answers = poor.run({"k": decision("p1")})
        assert answers["k"].by == "limit"
        assert batches.created == []  # nothing sent

        never = Batches(lambda request: '{"choice": "1"}')
        never.retrieve = lambda batch_id: SimpleNamespace(processing_status="in_progress")  # type: ignore[method-assign]
        ticks = iter(range(0, 10_000, 100))
        slow = batch_model(never, cache)
        slow.clock = lambda: float(next(ticks))
        slow.wait_s = 250
        with pytest.raises(BatchError, match="did not end in 250 s: canceled"):
            slow.run({"k": decision("p1")})
        assert never.canceled == ["batch-1"]

        bedrock = BedrockModel("claude-haiku-5-5", client=SimpleNamespace())
        with pytest.raises(BatchError, match="Anthropic API only, not on bedrock"):
            BatchModel(bedrock, CostMeter(PRICE, 1.0, batch=True), cache, SPACE)


def test_collector_and_replay(tmp_path: Path) -> None:
    with DecisionCache(tmp_path / FILE) as cache:
        collector = Collector(cache, SPACE, "anthropic", "claude-haiku-5-5")
        collector.start()
        assert collector.choose(decision("p1")).reason == "waiting for the batch"
        assert collector.choose(decision("p2")).by == "limit"  # this phrase waits
        collector.start()
        collector.choose(decision("p2"))
        assert list(collector.pending) == [cache_key(decision(p), SPACE) for p in ("p1", "p2")]
        cache.put(decision("p1"), SPACE, "1")
        collector.start()
        assert collector.choose(decision("p1")) == Choice(option="1", by="cache")

        paid = Choice(option="1", by="llm", calls=1, cost_usd=0.002)
        replay = Replay(cache, SPACE, "anthropic", "m", {cache_key(decision("p1"), SPACE): paid}, 4)
        assert replay.choose(decision("p1")) == paid  # the first phrase that uses it pays
        assert replay.choose(decision("p1")) == Choice(option="1", by="cache")  # then free
        missing = replay.choose(decision("p9"))
        assert (missing.option, missing.by) == (None, "limit")
        assert replay.choose(decision("p9"), calls=4).reason == "4 calls for this phrase"
