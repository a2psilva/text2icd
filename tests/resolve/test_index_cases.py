"""Index-walk cases written from the CMS Index (FY2027), and the worked examples of
docs/coder-workflow.md, as the Index walk codes them before Tabular verification."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import load
from text2icd.resolve.walk import DEFAULT_CODE, Candidate, IndexWalker

CASES = json.loads((Path(__file__).parent / "fy2027_index_walks.json").read_text())


@pytest.fixture(scope="module")
def walker(release_graph: Callable[[str], BuildReport]) -> Iterator[IndexWalker]:
    with load(path=release_graph(CASES["release"]).path) as graph:
        yield IndexWalker(graph)


def best(walker: IndexWalker, phrase: str) -> Candidate:
    result = walker.walk(phrase)
    assert result.diagnoses, phrase
    candidate = result.diagnoses[0].best
    assert candidate is not None, phrase
    return candidate


def test_enough_cases() -> None:
    cases = CASES["cases"]
    assert len(cases) >= 100
    assert len({case["code"][0] for case in cases}) >= 20  # spread over the chapters


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=pytest.mark.xfail(reason=case["xfail"], strict=True))
        if "xfail" in case
        else case
        for case in CASES["cases"]
    ],
    ids=[case["path"] for case in CASES["cases"]],
)
def test_index_case(walker: IndexWalker, case: dict[str, str]) -> None:
    assert best(walker, case["phrase"]).code == case["code"]


# The worked examples of docs/coder-workflow.md: the Index part of each.


@pytest.mark.slow
def test_default_code_example(walker: IndexWalker) -> None:
    found = best(walker, "Appendicitis")
    assert (found.code, found.path) == ("K37", "Appendicitis")
    assert DEFAULT_CODE in found.flags


@pytest.mark.slow
def test_subterm_descent_example(walker: IndexWalker) -> None:
    found = best(walker, "aspiration pneumonia due to food")
    assert (found.code, found.path) == ("J69.0", "Pneumonia > aspiration > due to > food")


@pytest.mark.slow
def test_combination_code_example(walker: IndexWalker) -> None:
    found = best(walker, "type 2 DM with diabetic CKD stage 3a")
    assert found.code == "E11.22"
    assert found.path == "Diabetes, diabetic > type 2 > with > chronic kidney disease"
    # The second Index walk, for the stage the Tabular note asks for (added in Phase 6).
    stage = walker.walk_text("chronic kidney disease stage 3a").best
    assert stage is not None
    assert stage.code == "N18.31"


@pytest.mark.slow
def test_etiology_and_manifestation_example(walker: IndexWalker) -> None:
    found = best(walker, "amyloid heart disease")
    assert (found.code, found.manifestation) == ("E85.4", "I43")


@pytest.mark.slow
def test_seventh_character_example(walker: IndexWalker) -> None:
    found = best(
        walker,
        "mild nonproliferative diabetic retinopathy with macular edema due to underlying "
        "condition, right eye",
    )
    # The Index gives the code up to the 6th character; the 7th (1 = right eye) comes from the
    # Tabular List's sevenChrDef, in Phase 6.
    assert "E08.3211".startswith(found.code)
    assert found.code.startswith("E08.32")


@pytest.mark.slow
def test_placeholder_example(walker: IndexWalker) -> None:
    found = best(walker, "accidental poisoning by 1-propanol, initial encounter")
    assert (found.code, found.book, found.column) == ("T51.3X1", "drug", 2)
