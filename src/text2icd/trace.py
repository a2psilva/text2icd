"""Trace steps: what each layer did to a phrase, in order, so every result can be audited."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Layer = Literal["text", "cache", "index", "tabular", "retrieval", "llm"]


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: Layer
    action: str  # e.g. "shorthand", "abbreviation", "split", "default_code"
    detail: str  # human-readable, e.g. '"w/" -> "with"'
    guideline: str | None = None  # the guideline section a rule applies, e.g. "II.H"


def text_step(action: str, detail: str, guideline: str | None = None) -> TraceStep:
    return TraceStep(layer="text", action=action, detail=detail, guideline=guideline)


def change(before: str, after: str) -> str:
    """The detail of a rewrite: `"w/" -> "with"`."""
    return f'"{before}" -> "{after}"'
