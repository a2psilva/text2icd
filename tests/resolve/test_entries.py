"""Main-term selection: candidates and the coder's ranking."""

from collections.abc import Callable, Iterator

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.resolve.main_terms import Correction, MainTermFinder, index_speller


def titles(finder: MainTermFinder, text: str, book: str = "index") -> list[str]:
    return [entry.term.title for entry in finder.candidates(finder.phrase_words(text), book)]


def test_phrase_words(graph: Graph) -> None:
    finder = MainTermFinder(graph)
    found = finder.phrase_words("Aspirtion pneumonias due to the food")
    assert found.forms == ("aspiration", "pneumonia", "food")
    assert found.corrections == (Correction("aspirtion", "aspiration"),)
    assert finder.phrase_words("pneumonia, pneumonia").forms == ("pneumonia",)


def test_the_condition_comes_first(graph: Graph) -> None:
    finder = MainTermFinder(graph)
    (entry,) = finder.candidates(finder.phrase_words("aspiration pneumonia due to food"))
    assert entry.term.title == "Pneumonia"
    assert entry.match.used == {"pneumonia"}
    assert entry.reach == 2  # aspiration and food are below it
    assert entry.score == 3
    assert entry.found_by == "word"


def test_ranked_by_what_the_subterms_can_use(graph: Graph) -> None:
    finder = MainTermFinder(graph)
    assert titles(finder, "pyelitis with contracted kidney") == ["Pyelitis"]
    assert titles(finder, "acute abdomen") == ["Abdomen, abdominal"]  # see also condition
    assert titles(finder, "abandonment") == ["Abandonment"]


def test_search_when_no_main_term_holds_a_word(graph: Graph) -> None:
    finder = MainTermFinder(graph)
    entries = finder.candidates(finder.phrase_words("regurgitated milk"))
    assert [entry.term.title for entry in entries][:1] == ["Pneumonia"]
    assert entries[0].found_by == "search"
    assert finder.candidates(finder.phrase_words("of the")) == []


def test_other_books(graph: Graph) -> None:
    finder = MainTermFinder(graph)
    assert titles(finder, "1-propanol", "drug")[:1] == ["1-propanol"]


def test_speller_keeps_known_words(graph: Graph) -> None:
    finder = MainTermFinder(graph, speller=index_speller(graph, known={"aspirtion"}))
    assert finder.phrase_words("aspirtion").corrections == ()


@pytest.fixture(scope="module")
def release_finder(release_graph: Callable[[str], BuildReport]) -> Iterator[MainTermFinder]:
    with load(path=release_graph("fy2027").path) as graph:
        yield MainTermFinder(graph)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "main_term"),
    [
        ("femur fracture", "Fracture, traumatic"),
        ("kidney failure", "Failure, failed"),
        ("renal failure", "Failure, failed"),
        ("aspiration pneumonia due to food", "Pneumonia"),
        ("type 2 diabetes with chronic kidney disease", "Diabetes, diabetic"),
        ("congestive heart failure", "Failure, failed"),
        ("acute appendicitis", "Appendicitis"),
        ("hyperplasia of prostate", "Hyperplasia, hyperplastic"),
        ("systeimc candidiasis", "Candidiasis, candidal"),
    ],
)
def test_release_main_terms(release_finder: MainTermFinder, phrase: str, main_term: str) -> None:
    assert titles(release_finder, phrase)[0] == main_term
