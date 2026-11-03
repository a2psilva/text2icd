"""What a system returns for one example, as scored by `text2icd.eval.metrics` and stored in result
files. Codes are kept as the system wrote them (dotted when well formed), so that malformed or
non-billable codes are counted rather than rejected."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from text2icd.ontology.codes import dot, is_code


def normalize_code(code: str) -> str:
    value = code.strip().upper()
    return dot(value) if is_code(value) else value


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    codes: tuple[str, ...] = ()  # the code set: the primary code first, then companion codes
    alternates: tuple[str, ...] = ()  # other candidate primary codes, best first
    confidence: float | None = Field(default=None, ge=0, le=1)
    # The layers that escalated the phrase, in order ("index": the coder's confidence was below
    # its threshold; "retrieval": L2's too); None when the system does not escalate.
    escalated: tuple[str, ...] | None = None
    llm_calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    usage: dict[str, int] = Field(default_factory=dict)  # provider token counts, summed
    error: str | None = None

    @field_validator("codes", "alternates")
    @classmethod
    def _normalize(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_code(code) for code in value)

    @property
    def primary(self) -> str | None:
        return self.codes[0] if self.codes else None

    @property
    def ranked(self) -> tuple[str, ...]:
        """The primary code then the alternates, without repeats: the list accuracy@k reads."""
        return tuple(dict.fromkeys(self.codes[:1] + self.alternates))
