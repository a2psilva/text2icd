"""B3/B4 with a stand-in model and retriever: no network call is ever made."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.llm_baseline import LLMChoiceSystem, answer_schema, user_prompt
from text2icd.eval.predictions import Prediction
from text2icd.eval.runner import RunError, evaluate
from text2icd.eval.systems import System, SystemSetupError, build_system
from text2icd.graph.model import Graph
from text2icd.llm.cost import Usage
from text2icd.llm.providers import (
    ALLOW_LIST,
    Answer,
    AnthropicModel,
    EndpointNotAllowedError,
    LLMUnavailableError,
    check_endpoint,
    choice_model,
)

CANDIDATES = ("E11.22", "E08.3211", "E08.3212")


class Retriever(System):
    name = "fake-retriever"

    def predict(self, phrase: str) -> Prediction:
        return Prediction(id="", codes=CANDIDATES[:1], alternates=CANDIDATES[1:])

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        for phrase in phrases:
            yield self.predict(phrase) if phrase != "nothing" else Prediction(id="")

    def settings(self) -> dict[str, Any]:
        return {"name": "fake"}


class Model:
    provider = "anthropic"
    model = "claude-haiku-5-5"

    def __init__(self, answers: list[Answer | Exception]) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    def ask(self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int) -> Answer:
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


USAGE = Usage(input_tokens=1000, output_tokens=20)


def system(graph: Graph, model: Model, **options: str) -> LLMChoiceSystem:
    return LLMChoiceSystem(graph, Settings(), options, retriever=Retriever(), model=model)


def test_choice(graph: Graph) -> None:
    model = Model([Answer({"primary": 2, "companions": [1, 2], "confidence": 0.8}, USAGE, "end")])
    (prediction,) = system(graph, model).predict_many(["diabetes with retinopathy"])
    assert prediction.codes == ("E08.3211", "E11.22")
    assert prediction.alternates == ("E08.3212",)
    assert prediction.confidence == 0.8
    assert prediction.llm_calls == 1
    assert prediction.cost_usd == pytest.approx((1000 * 0.10 + 20 * 0.50) / 1e6)
    assert prediction.usage == {"input_tokens": 1000, "output_tokens": 20}
    assert "1. E11.22 Type 2 diabetes mellitus" in model.prompts[0]
    assert model.prompts[0].startswith("Phrase: diabetes with retinopathy")


def test_no_fit_refusal_and_errors(graph: Graph) -> None:
    model = Model(
        [
            Answer({"primary": 0, "companions": [], "confidence": 0.1}, USAGE, "end"),
            Answer(None, USAGE, "refusal", "the model refused"),
            RuntimeError("timeout"),
        ]
    )
    llm = system(graph, model)
    none, refused, failed, empty = llm.predict_many(["a", "b", "c", "nothing"])
    assert (none.codes, none.confidence, none.llm_calls) == ((), 0.0, 1)
    assert (refused.error, refused.llm_calls) == ("the model refused", 1)
    assert failed.error == "RuntimeError: timeout"
    assert empty.error == "no candidates"
    assert llm.meter.calls == 2


def test_budget_stops_the_calls(graph: Graph) -> None:
    big = Usage(input_tokens=10_000_000)  # $1 at the Haiku input price
    model = Model([Answer({"primary": 1, "companions": [], "confidence": 1}, big, "end")] * 3)
    llm = system(graph, model, budget="0.5")
    first, second = llm.predict_many(["a", "b"])
    assert first.codes == ("E11.22",)
    assert second.error is not None
    assert "budget of $0.50 spent" in second.error
    assert len(model.prompts) == 1


def test_settings_and_options(graph: Graph) -> None:
    llm = system(graph, Model([]), candidates="10")
    assert llm.settings() == {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "effort": "low",
        "candidates": 10,
        "budget_usd": 1.0,
        "retriever": {"name": "fake"},
    }
    with pytest.raises(SystemSetupError, match="unknown options for llm: temperature"):
        system(graph, Model([]), temperature="0")
    with pytest.raises(SystemSetupError, match="needs a provider and a model"):
        LLMChoiceSystem(graph, Settings(llm_provider="none"), {}, retriever=Retriever())
    with pytest.raises(SystemSetupError, match="budget must be a number"):
        system(graph, Model([]), budget="lots")


def test_schema_and_prompt() -> None:
    schema = answer_schema(3)
    assert schema["properties"]["primary"]["enum"] == [0, 1, 2, 3]
    assert schema["properties"]["companions"]["items"]["enum"] == [1, 2, 3]
    assert schema["additionalProperties"] is False
    assert user_prompt("x", [("A00.0", "Cholera")]) == "Phrase: x\n\nCandidates:\n1. A00.0 Cholera"


def test_endpoint_policy() -> None:
    check_endpoint("anything", restricted=False, require_compliant=False)
    check_endpoint("anthropic", restricted=True, require_compliant=True)
    assert "physionet.org" in ALLOW_LIST["anthropic"]
    with pytest.raises(EndpointNotAllowedError, match="REQUIRE_DUA_COMPLIANT_ENDPOINT=true"):
        check_endpoint("anthropic", restricted=True, require_compliant=False)
    with pytest.raises(EndpointNotAllowedError, match="not approved for restricted data"):
        check_endpoint("other", restricted=True, require_compliant=True)


class Messages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return self.response


def response(text: str | None, stop: str = "end_turn") -> Any:
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    usage = SimpleNamespace(
        input_tokens=50,
        output_tokens=9,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=0,
        cache_creation=None,
    )
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage)


def test_anthropic_model_request_and_answers() -> None:
    messages = Messages(response('{"primary": 1, "companions": [], "confidence": 0.9}'))
    model = AnthropicModel("claude-haiku-5-5", client=SimpleNamespace(messages=messages))
    answer = model.ask(system="rules", prompt="phrase", schema={"type": "object"}, max_tokens=50)
    assert answer.data == {"primary": 1, "companions": [], "confidence": 0.9}
    assert answer.usage.cache_read_input_tokens == 600
    (request,) = messages.requests
    assert request["thinking"] == {"type": "disabled"}
    assert request["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    thinking = AnthropicModel(
        "claude-opus-5-5", thinking=True, client=SimpleNamespace(messages=messages)
    )
    thinking.ask(system="s", prompt="p", schema={}, max_tokens=10)
    assert "thinking" not in messages.requests[-1]


@pytest.mark.parametrize(
    ("reply", "stop", "error"),
    [
        (None, "refusal", "the model refused"),
        ('{"primary": 1', "max_tokens", "no complete answer (stop reason max_tokens)"),
        ("not json", "end_turn", "the answer is not JSON"),
        ("[1, 2]", "end_turn", "the answer is not a JSON object"),
    ],
)
def test_anthropic_model_bad_answers(reply: str | None, stop: str, error: str) -> None:
    model = AnthropicModel(
        "claude-haiku-5-5", client=SimpleNamespace(messages=Messages(response(reply, stop)))
    )
    answer = model.ask(system="s", prompt="p", schema={}, max_tokens=10)
    assert (answer.data, answer.error) == (None, error)


def test_choice_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMUnavailableError, match="unknown LLM provider 'other'"):
        choice_model("other", "m", api_key=None, effort="low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    haiku = choice_model("anthropic", "claude-haiku-5-5", api_key=None, effort="low")
    assert isinstance(haiku, AnthropicModel)
    assert haiku.thinking is False
    opus = choice_model("anthropic", "claude-opus-5-5", api_key=None, effort="low")
    assert isinstance(opus, AnthropicModel)
    assert opus.thinking is True


def restricted_dataset(tmp_path: Path, source: str) -> Path:
    path = tmp_path / f"{source}.jsonl"
    example = Example.model_validate(
        {
            "id": "x",
            "phrase": "type 2 diabetes mellitus with diabetic chronic kidney disease",
            "gold": [{"code": "E11.22", "role": "primary"}],
            "release": "fy2099",
            "source": source,
            "split": "dev",
        }
    )
    write_dataset(path, [example])
    return path


class Remote(Retriever):
    """A system that sends phrases to a service that is not on the allow-list."""

    name = "remote"
    uses_llm = True

    @property
    def endpoint(self) -> str | None:
        return "elsewhere"


def test_runner_endpoint_policy(graph: Graph, tmp_path: Path) -> None:
    answer = Answer({"primary": 1, "companions": [], "confidence": 1}, USAGE, "end")
    public = restricted_dataset(tmp_path, "cms-synthetic")
    mimic = restricted_dataset(tmp_path, "mimic-silver")
    run, _ = evaluate(system(graph, Model([answer])), graph, public, resamples=0)
    assert run.header["system"]["model"] == "claude-haiku-5-5"
    evaluate(system(graph, Model([answer])), graph, mimic, resamples=0)
    with pytest.raises(RunError, match="REQUIRE_DUA_COMPLIANT_ENDPOINT=true"):
        evaluate(
            system(graph, Model([answer])),
            graph,
            mimic,
            resamples=0,
            require_compliant_endpoint=False,
        )
    with pytest.raises(RunError, match="'elsewhere' is not approved"):
        evaluate(Remote(), graph, mimic, resamples=0)
    assert Retriever().endpoint is None


class Locked(Retriever):
    """A system whose provider refuses every call (bad credentials)."""

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        raise LLMUnavailableError("anthropic refused the call: AuthenticationError 401")
        yield  # pragma: no cover


def test_runner_stops_when_the_provider_refuses(graph: Graph, tmp_path: Path) -> None:
    """Every call would fail the same way: the run stops instead of recording an error each."""
    public = restricted_dataset(tmp_path, "cms-synthetic")
    with pytest.raises(LLMUnavailableError, match="AuthenticationError 401"):
        evaluate(Locked(), graph, public, resamples=0)


def test_llm_system_needs_settings(graph: Graph, graph_file: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemSetupError, match="needs a provider and a model"):
        build_system("llm", graph, Settings(llm_provider="none"), {})
    dataset = restricted_dataset(tmp_path, "cms-synthetic")
    base = ["eval", "run", "-d", str(dataset), "-s", "llm", "--graph", str(graph_file)]
    runner = CliRunner()
    unknown = runner.invoke(cli.app, [*base, "-O", "provider=other", "-O", "model=m"])
    assert unknown.exit_code == 1
    assert "unknown LLM provider 'other'" in unknown.output
    unpriced = runner.invoke(
        cli.app,
        [*base, "-O", "provider=anthropic", "-O", "model=claude-unpriced"],
        env={"ANTHROPIC_API_KEY": "test-key-not-used"},
    )
    assert unpriced.exit_code == 1
    assert "no price for claude-unpriced on anthropic" in unpriced.output
