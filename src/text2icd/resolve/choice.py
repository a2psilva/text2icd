"""L2's choice (task 7.6): for a phrase the coder escalates, the codes retrieval finds compete with
the coder's own.

Retrieval (`retrieve`) ranks Tabular nodes for the normalised phrase, and Tabular verification
completes them to billable codes with the phrase's evidence, as it completes the Index walk's
candidates. The pool is the coder's primary code and its alternates, then the first `POOL`
retrieved codes. Each code of the pool gets the features of `confidence.RETRIEVAL_FEATURES`, and
the retrieval stage of the calibration table scores it: the best score wins, and its
probability is the confidence of the code set (`Coder.code`).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.resolve.confidence import RETRIEVAL_FEATURES, Stage
from text2icd.resolve.retrieve import Hits
from text2icd.resolve.tabular import Completion

NODES = 50  # nodes retrieved per phrase
DEPTH = 20  # distinct billable codes completed from them, at most
POOL = 5  # retrieved codes that compete with the coder's
BOOK = "retrieval"  # the `Candidate.book` of a code retrieval found
RETRIEVED = "retrieved"  # its flag


class Pooled(BaseModel):
    """A code L2 weighed, kept on the code set for the UI and audits (task 9.11)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    description: str
    origin: Literal["coder", "alternate", "retrieved"]  # the coder's code, an alternate, or not
    probability: float  # that it is right, by the retrieval stage
    node: str | None = None  # the retrieved node it was completed from
    similarity: float | None = None


@dataclass(frozen=True, slots=True)
class Option:
    """One code of the pool, with where it comes from and its features."""

    code: str
    description: str
    coder: bool  # the coder's primary code
    index: bool  # the coder's code or one of its alternates (from the Index walk)
    node: str | None  # the retrieved node it was completed from, if retrieved
    similarity: float | None  # of that node's best text
    completion: Completion | None
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class _Retrieved:
    node: str
    similarity: float
    completion: Completion


def options(
    coder_codes: Sequence[str],
    confidence: float,
    hits: Hits,
    complete: Callable[[str], Completion | None],
    titles: Callable[[str], set[str]],
    describe: Callable[[str], str],
    words: frozenset[str],
) -> list[Option]:
    """The pool, the coder's codes first: `coder_codes`, its primary code then alternates;
    `hits`, the retrieved nodes, best first; `complete`, Tabular verification with the phrase's
    evidence; `titles`, the words of a code's titles; `words`, the phrase's content words."""
    retrieved: list[_Retrieved] = []
    seen: set[str] = set()
    for node, similarity in hits:
        completion = complete(node)
        if completion is None or completion.code in seen:
            continue
        seen.add(completion.code)
        retrieved.append(_Retrieved(node, similarity, completion))
        if len(retrieved) == DEPTH:
            break
    codes = [r.completion.code for r in retrieved]
    floor = hits[-1][1] if hits else 0.0
    first = retrieved[0].similarity if retrieved else 0.0
    margin = first - retrieved[1].similarity if len(retrieved) > 1 else 0.0
    agreement = float(bool(codes) and codes[0] in coder_codes)
    pool = list(dict.fromkeys([*coder_codes, *codes[:POOL]]))
    out: list[Option] = []
    for code in pool:
        found = retrieved[codes.index(code)] if code in codes else None
        similarity = found.similarity if found is not None else floor
        is_coder = bool(coder_codes) and code == coder_codes[0]
        description = found.completion.description if found is not None else describe(code)
        coverage = len(words & titles(code)) / len(words) if words else 0.0
        values = {
            "coder": float(is_coder),
            "coder_rank": 1 / (1 + coder_codes.index(code)) if code in coder_codes else 0.0,
            "retrieval_rank": 1 / (1 + codes.index(code)) if found is not None else 0.0,
            "retrieval_first": float(bool(codes) and code == codes[0]),
            "similarity": similarity,
            "similarity_gap": first - similarity,
            "top_similarity": first,
            "top_margin": margin,
            "confidence": confidence,
            "coder_confidence": confidence if is_coder else 0.0,
            "coverage": coverage,
            "coder_coverage": coverage if is_coder else 0.0,
            "coder_similarity": similarity if is_coder else 0.0,
            "not_retrieved": float(found is None),
            "unspecified": float("unspecified" in description.casefold()),
            "uncoded": float(not coder_codes),
            "words": float(len(words)),
            "agreement": agreement,
        }
        out.append(
            Option(
                code=code,
                description=description,
                coder=is_coder,
                index=code in coder_codes,
                node=found.node if found is not None else None,
                similarity=found.similarity if found is not None else None,
                completion=found.completion if found is not None else None,
                features={name: round(float(values[name]), 6) for name in RETRIEVAL_FEATURES},
            )
        )
    return out


def choose(pool: Sequence[Option], stage: Stage) -> tuple[Option, float] | None:
    """The option with the best score (the first of equals: the coder's before retrieval's),
    and its probability of being right."""
    if not pool:
        return None
    best = max(pool, key=lambda option: stage.score(option.features))  # max keeps the first
    return best, stage.probability(best.features)
