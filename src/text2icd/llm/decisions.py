"""What the navigator is asked and what it answers (task 8.1).

A decision is one point where the graph offers several ways forward that the deterministic coder
cannot tell apart: which main term to look up, which Index entry codes the diagnosis, which child
of a Tabular code the phrase supports, or what an ambiguous abbreviation means. Its options are
numbered, and the answer names one of them or `none`; a choice is never anything else.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

DecisionKind = Literal["main_term", "subterm", "tabular_child", "abbreviation"]
KINDS: tuple[DecisionKind, ...] = ("main_term", "subterm", "tabular_child", "abbreviation")
NONE = "none"
# How a choice was made: by the model, from the decision cache, or not at all because a limit
# was reached, the call failed, or no provider is configured.
ChosenBy = Literal["llm", "cache", "limit", "error", "off"]


class Offer(NamedTuple):
    """An option before it is numbered."""

    text: str  # an Index path, a code title, a meaning
    code: str | None = None  # the billable code it leads to, if it leads to one
    title: str | None = None  # that code's title in the Tabular List
    note: str | None = None  # how it was found, when that matters ("retrieval")


class Option(BaseModel):
    """One way forward."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str  # what the answer names: "1", "2", ...
    text: str
    code: str | None = None
    title: str | None = None
    note: str | None = None


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DecisionKind
    phrase: str
    subject: str | None = None  # what is decided: the abbreviation as written, the parent code
    path: tuple[str, ...] = ()  # the path so far: the main term, the code above
    options: tuple[Option, ...] = Field(min_length=1)

    @classmethod
    def of(
        cls,
        kind: DecisionKind,
        phrase: str,
        offers: Sequence[Offer],
        *,
        subject: str | None = None,
        path: Sequence[str] = (),
    ) -> "Decision":
        """A decision with its options numbered from 1, in the order given."""
        numbered = tuple(
            Option(id=str(n), **Offer(*offer)._asdict()) for n, offer in enumerate(offers, start=1)
        )
        return cls(kind=kind, phrase=phrase, subject=subject, path=tuple(path), options=numbered)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(option.id for option in self.options)

    def option(self, id: str | None) -> Option | None:
        return next((option for option in self.options if option.id == id), None)

    @property
    def key(self) -> str:
        """A hash of everything the decision says (64 hex characters): equal decisions have
        equal keys, whatever phrase they come from."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Choice(BaseModel):
    """The answer to a decision, with what it cost. `option` is None when the answer is
    `none`, falls outside the options, or no answer was given: the coder then keeps its own."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option: str | None = None
    by: ChosenBy
    reason: str | None = None  # why there is no option; never the input text
    calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0)
