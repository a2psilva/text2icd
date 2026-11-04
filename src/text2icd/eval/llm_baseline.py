"""B3 and B4: an LLM chooses among the top codes of the embedding baseline (B2).

The model never writes a code: it picks numbers from a candidate list, through a JSON schema
whose numbers are an enum, so every code it returns is a billable code of the release. One call
per phrase; the run's cost meter refuses further calls once the budget is spent, and the
remaining phrases get no prediction (counted as wrong).

B3 is a small, low-cost model; B4 a large one on a small sample (`--limit`). The provider, model
and budget come from the settings (`TEXT2ICD_LLM_*`) or `-O provider=... -O model=...
-O budget=...`.
"""

import time
from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import JsonValue

from text2icd.config import Settings
from text2icd.eval.predictions import Prediction
from text2icd.eval.systems import EmbeddingSystem, Options, System, SystemSetupError, option
from text2icd.graph.model import Graph
from text2icd.llm.cost import BudgetExceededError, CostMeter, load_prices
from text2icd.llm.navigator import provider_model
from text2icd.llm.providers import ChoiceModel, LLMUnavailableError

CANDIDATES = 20
MAX_TOKENS = 200

SYSTEM_PROMPT = """You are a certified medical coder assigning ICD-10-CM codes to one diagnosis \
phrase from a hospital discharge summary. You are given the phrase and a numbered list of \
candidate codes with their official descriptions. Choose from the list only.

How to choose:
- Pick the candidate that codes the phrase most accurately, as a coder would after checking the \
Alphabetic Index and the Tabular List. Prefer the most specific code that the phrase supports; \
when the phrase gives no detail, the unspecified code is correct. Do not assume details the \
phrase does not state.
- Read clinical abbreviations as a coder would in a discharge summary (for example HTN, CKD, \
CHF, AFib, UTI, AKI, COPD, DM2).
- Uncertain diagnoses such as "probable" or "suspected" are coded as if established, since \
these are inpatient records.
- "History of" a condition that no longer exists is a personal history code; a current \
condition is coded as the condition.
- When the phrase states two conditions that need two codes, or an instructional note requires \
a second code that the phrase supports, list the other candidates as companions, in order. \
Otherwise leave companions empty.
- If no candidate fits the phrase, answer 0 as the primary code.
- Confidence is your probability, from 0 to 1, that the primary code is the one a careful coder \
would assign.

Answer with the JSON object only."""


def answer_schema(count: int) -> dict[str, Any]:
    numbers = list(range(1, count + 1))
    return {
        "type": "object",
        "properties": {
            "primary": {"type": "integer", "enum": [0, *numbers]},
            "companions": {"type": "array", "items": {"type": "integer", "enum": numbers}},
            "confidence": {"type": "number"},
        },
        "required": ["primary", "companions", "confidence"],
        "additionalProperties": False,
    }


def user_prompt(phrase: str, candidates: Sequence[tuple[str, str]]) -> str:
    lines = [f"Phrase: {phrase}", "", "Candidates:"]
    lines += [f"{n}. {code} {description}" for n, (code, description) in enumerate(candidates, 1)]
    return "\n".join(lines)


class LLMChoiceSystem(System):
    name = "llm"
    uses_llm = True

    def __init__(
        self,
        graph: Graph,
        settings: Settings,
        options: Options,
        *,
        retriever: System | None = None,
        model: ChoiceModel | None = None,
    ) -> None:
        known = {"provider", "model", "effort", "budget", "candidates", "retriever_model"}
        unknown = set(options) - known
        if unknown:
            raise SystemSetupError(f"unknown options for llm: {', '.join(sorted(unknown))}")
        self.graph = graph
        self.candidates = option(options, "candidates", CANDIDATES)
        try:
            budget = float(options.get("budget", settings.llm_budget_usd))
        except ValueError:
            raise SystemSetupError("option budget must be a number of US dollars") from None
        effort = options.get("effort", "low")
        named = {key: options.get(key) for key in ("provider", "model")}
        if model is None:
            try:
                model = provider_model(settings, effort=effort, **named)
            except LLMUnavailableError as exc:
                raise SystemSetupError(f"the llm system: {exc}") from exc
        self.model = model
        self.meter = CostMeter(load_prices().get(self.model.provider, self.model.model), budget)
        retriever_options = {"top": str(self.candidates)}
        if "retriever_model" in options:
            retriever_options["model"] = options["retriever_model"]
        self.retriever = retriever or EmbeddingSystem(graph, retriever_options)
        self.effort = effort

    def predict(self, phrase: str) -> Prediction:
        return next(self.predict_many([phrase]))

    def predict_many(self, phrases: Sequence[str]) -> Iterator[Prediction]:
        for phrase, retrieved in zip(phrases, self.retriever.predict_many(phrases), strict=True):
            yield self._choose(phrase, retrieved)

    def _choose(self, phrase: str, retrieved: Prediction) -> Prediction:
        codes = list(retrieved.ranked[: self.candidates])
        if not codes:
            return Prediction(id="", confidence=0.0, error="no candidates")
        try:
            self.meter.check()
        except BudgetExceededError as exc:
            return Prediction(id="", error=str(exc))
        candidates = [(code, self.graph.code(code).description) for code in codes]
        started = time.perf_counter()
        try:
            answer = self.model.ask(
                system=SYSTEM_PROMPT,
                prompt=user_prompt(phrase, candidates),
                schema=answer_schema(len(candidates)),
                max_tokens=MAX_TOKENS,
            )
        except Exception as exc:  # an API failure is kept on this phrase
            return Prediction(id="", llm_calls=1, error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - started) * 1000
        cost = self.meter.charge(answer.usage)
        spent = {
            "llm_calls": 1,
            "cost_usd": cost,
            "usage": answer.usage.as_dict(),
            "latency_ms": latency,
        }
        if answer.data is None:
            return Prediction(id="", error=answer.error, **spent)
        primary = int(answer.data.get("primary", 0))
        if primary == 0:
            return Prediction(id="", confidence=0.0, **spent)
        chosen = [codes[primary - 1]]
        for number in answer.data.get("companions", []):
            code = codes[int(number) - 1]
            if code not in chosen:
                chosen.append(code)
        confidence = min(max(float(answer.data.get("confidence", 0.0)), 0.0), 1.0)
        alternates = tuple(code for code in codes if code not in chosen)[:4]
        return Prediction(
            id="", codes=tuple(chosen), alternates=alternates, confidence=confidence, **spent
        )

    def settings(self) -> dict[str, JsonValue]:
        return {
            "provider": self.model.provider,
            "model": self.model.model,
            "effort": self.effort,
            "candidates": self.candidates,
            "budget_usd": self.meter.budget_usd,
            "retriever": self.retriever.settings(),
        }

    @property
    def endpoint(self) -> str | None:
        return self.model.provider

    def warnings(self) -> list[str]:
        warning = self.meter.cache_warning()
        return [warning] if warning else []

    def close(self) -> None:
        self.retriever.close()
