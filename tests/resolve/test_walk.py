"""The subterm walk of the Alphabetic Index: candidates, default codes and trace."""

from collections.abc import Callable, Iterator

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.resolve.walk import (
    DEFAULT_CODE,
    Candidate,
    IndexWalker,
    WalkResult,
    describe_walk,
    diagnosis_text,
)


def best(walker: IndexWalker, phrase: str) -> Candidate:
    result = walker.walk(phrase)
    assert result.diagnoses, phrase
    candidate = result.diagnoses[0].best
    assert candidate is not None, phrase
    return candidate


def test_walks_down_to_the_subterm(graph: Graph) -> None:
    walker = IndexWalker(graph)
    found = best(walker, "aspiration pneumonia due to food")
    assert found.code == "J69.0"
    assert found.path == "Pneumonia > aspiration > due to > food"
    assert found.coded_at == found.path
    assert found.score == 3
    assert found.used == ("aspiration", "pneumonia", "food")
    assert found.unused == ()
    assert DEFAULT_CODE not in found.flags
    actions = [step.action for step in found.trace]
    assert actions == ["main_term", "subterm", "subterm", "subterm", "code"]
    assert found.trace[2].detail == "due to"  # a connector, passed through


def test_default_code(graph: Graph) -> None:
    walker = IndexWalker(graph)
    found = best(walker, "amyloid heart")
    assert (found.code, found.flags) == ("E85.4", (DEFAULT_CODE,))
    default = next(step for step in found.trace if step.action == "default_code")
    assert default.guideline == "I.A.18"
    # A nonessential modifier does not change the code.
    disease = best(walker, "amyloid heart disease")
    assert (disease.code, disease.score) == ("E85.4", 2.8)
    assert DEFAULT_CODE in disease.flags
    # A code the Tabular List lacks is no code (the fixture has no J18.9).
    assert walker.walk("pneumonia").diagnoses[0].candidates == ()


def test_nonessential_modifiers(graph: Graph) -> None:
    walker = IndexWalker(graph)
    found = best(walker, "aspiration pneumonia due to regurgitated food")
    assert found.code == "J69.0"
    assert found.score == 3.8  # a small bonus for the nonessential modifier
    assert "regurgitated" in found.used


def test_connector_subterms(graph: Graph) -> None:
    walker = IndexWalker(graph)
    assert best(walker, "pyelitis with contracted kidney").code == "N11.9"
    assert best(walker, "pneumonia with lung abscess").code == "J85.1"
    hydronephrosis = best(walker, "chronic pyelitis with calculus with hydronephrosis")
    assert hydronephrosis.code == "N13.6"
    assert hydronephrosis.path == "Pyelitis > chronic > with calculus > with hydronephrosis"


def test_without_undoes_with(graph: Graph) -> None:
    walker = IndexWalker(graph)
    found = walker.walk("chronic pyelitis with calculus without hydronephrosis").diagnoses[0]
    assert found.best is not None
    assert found.best.code == "N11.9"
    assert "N13.6" not in {candidate.code for candidate in found.candidates}


def test_manifestation(graph: Graph) -> None:
    found = best(IndexWalker(graph), "amyloid heart")
    assert (found.code, found.manifestation) == ("E85.4", "I43")


def test_see_is_not_a_default_code(graph: Graph) -> None:
    walker = IndexWalker(graph)
    # `Pneumonia > with > influenza` says "see Influenza, with, pneumonia": the coder follows it
    # (task 5.6) rather than take the default code of Pneumonia for it.
    found = walker.walk("pneumonia with influenza").diagnoses[0]
    assert all("influenza" not in candidate.used for candidate in found.candidates)


def test_episode_wording_is_not_walked(graph: Graph) -> None:
    walker = IndexWalker(graph)
    result = walker.walk("aspiration pneumonia due to food, initial encounter")
    assert result.diagnoses[0].text == "aspiration pneumonia due to food"
    assert diagnosis_text(result.parsed, (0,)) == "aspiration pneumonia due to food"


def test_diagnoses(graph: Graph) -> None:
    walker = IndexWalker(graph)
    both = walker.walk("aspiration pneumonia and chronic pyelitis")
    assert [walk.best.code for walk in both.diagnoses if walk.best] == ["J69.0", "N11.9"]
    assert walker.walk("no pneumonia").diagnoses == ()  # negated: nothing to code


def test_joined_when_one_path_codes_both(graph: Graph) -> None:
    walker = IndexWalker(graph)
    first = walker.walk_text("pneumonia due to food", (0,))
    second = walker.walk_text("aspiration", (1,))
    (joined,) = walker.join([first, second])
    assert joined.conditions == (0, 1)
    assert joined.text == "pneumonia due to food, aspiration"
    assert joined.best is not None
    assert joined.best.path == "Pneumonia > aspiration > due to > food"
    assert joined.trace[0].action == "joined"
    apart = walker.join(
        [walker.walk_text("aspiration pneumonia", (0,)), walker.walk_text("chronic pyelitis", (1,))]
    )
    assert [walk.text for walk in apart] == ["aspiration pneumonia", "chronic pyelitis"]


def test_describe(graph: Graph) -> None:
    result = IndexWalker(graph).walk("aspiration pneumonia due to food")
    lines = describe_walk(result, top=2)
    assert lines[0] == "normalized: aspiration pneumonia due to food"
    assert any(line.startswith("  1. J69.0") for line in lines)
    assert "  trace of 1:" in lines
    assert describe_walk(WalkResult(parsed=result.parsed))[-1] == "no diagnosis to walk"


@pytest.fixture(scope="module")
def release_walker(release_graph: Callable[[str], BuildReport]) -> Iterator[IndexWalker]:
    with load(path=release_graph("fy2027").path) as graph:
        yield IndexWalker(graph)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "code", "path"),
    [
        ("aspiration pneumonia due to food", "J69.0", "Pneumonia > aspiration > due to > food"),
        ("femur fracture, initial encounter", "S72.9", "Fracture, traumatic > femur, femoral"),
        (
            "type 2 diabetes with chronic kidney disease",
            "E11.22",
            "Diabetes, diabetic > type 2 > with > chronic kidney disease",
        ),
        ("diabetic nephropathy", "E11.21", "Diabetes, diabetic > with > nephropathy"),
        (
            "hypertensive heart disease with heart failure",
            "I11.0",
            "Hypertension, hypertensive > heart > with > heart failure",
        ),
        ("acute appendicitis", "K35.80", "Appendicitis > acute"),
        ("pneumonia", "J18.9", "Pneumonia"),
        ("Kaposi's sarcoma of skin", "C46.0", "Sarcoma > Kaposi's > skin"),
        (
            "paroxysmal cough due to Bordetella pertussis without pneumonia",
            "A37.00",
            "Cough > paroxysmal, due to Bordetella pertussis",
        ),
        (
            "psychoactive drug abuse with psychosis, delusions",
            "F19.150",
            "Abuse > drug NEC > psychoactive NEC > with > psychosis > delusions",
        ),
        (
            "epilepsy not intractable, without status epilepticus",
            "G40.909",
            "Epilepsy, epileptic, epilepsia > not intractable > without status epilepticus",
        ),
    ],
)
def test_release_walks(release_walker: IndexWalker, phrase: str, code: str, path: str) -> None:
    found = best(release_walker, phrase)
    assert (found.code, found.path) == (code, path)


@pytest.mark.slow
def test_release_unspecified_subterm(release_walker: IndexWalker) -> None:
    found = best(release_walker, "Kaposi's sarcoma")
    assert found.code == "C46.9"  # Sarcoma > Kaposi's > unspecified site
    assert DEFAULT_CODE in found.flags
