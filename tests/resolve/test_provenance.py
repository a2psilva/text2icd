"""Task 9.11: each code links to the conditions (chunks) it codes, and its Index path is a list
of graph terms, so a UI can draw the path from the words to the code."""

import pytest

from text2icd.graph.model import Graph
from text2icd.resolve.coder import Coder


def test_path_steps_follow_the_index(graph: Graph) -> None:
    code_set = Coder.from_graph(graph).code("aspiration pneumonia due to food")
    (diagnosis,) = code_set.diagnoses
    candidate = diagnosis.candidate
    assert candidate.path == "Pneumonia > aspiration > due to > food"
    assert [step.title for step in candidate.steps] == candidate.path.split(" > ")
    for step in candidate.steps:
        assert graph.term(step.term).title == step.title
    assert [step.took for step in candidate.steps] == [
        ("pneumonia",),
        ("aspiration",),
        (),
        ("food",),
    ]
    assert candidate.coded_step == 3
    # The trace names the terms and the code it reached.
    index = [(s.action, s.term, s.code) for s in diagnosis.trace if s.term is not None]
    terms = [step.term for step in candidate.steps]
    assert index[:4] == [("main_term", terms[0], None), *(("subterm", t, None) for t in terms[1:])]
    assert index[-1] == ("code", terms[3], "J69.0")
    assert code_set.sequence[0].conditions == (0,)
    assert diagnosis.conditions == (0,)


def test_tabular_steps_name_their_code(graph: Graph) -> None:
    (diagnosis,) = Coder.from_graph(graph).code("poisoning by propanol").diagnoses
    tabular = [(s.action, s.code) for s in diagnosis.trace if s.layer == "tabular"]
    assert tabular[-1] == ("complete", "T51.3X1A")
    assert all(code is not None for _, code in tabular)


def test_xrefs_to_is_the_reverse_of_xrefs(graph: Graph) -> None:
    pairs = {(x.term, x.target) for t in graph.terms() for x in graph.xrefs(t.id) if x.target}
    assert pairs
    reverse = {(x.term, target) for target in {t for _, t in pairs} for x in graph.xrefs_to(target)}
    assert reverse == pairs


@pytest.mark.slow
def test_codes_link_to_the_chunks_they_code(release_coder: Coder) -> None:
    phrase = "DM2 w/ CKD 3a, possible UTI 2/2 E. coli"
    code_set = release_coder.code(phrase)
    parsed = code_set.parsed
    chunks = [phrase[slice(*parsed.source(c.start, c.end))] for c in parsed.conditions]
    assert chunks == ["DM2", "CKD 3a", "possible UTI 2/2 E. coli"]
    linked = {s.code: [chunks[i] for i in s.conditions] for s in code_set.sequence}
    assert linked["E11.22"] == ["DM2", "CKD 3a"]
    assert linked["N18.31"] == ["DM2", "CKD 3a"]  # brought in by E11.22's note
    assert all(s.conditions for s in code_set.sequence)


@pytest.mark.slow
def test_list_codes_link_one_to_one(release_coder: Coder) -> None:
    code_set = release_coder.code("COPD, CHF, AFib")
    assert [(s.code, s.conditions) for s in code_set.sequence] == [
        ("J44.9", (0,)),
        ("I50.9", (1,)),
        ("I48.91", (2,)),
    ]


@pytest.mark.slow
def test_a_see_step_is_marked(release_coder: Coder) -> None:
    (diagnosis,) = release_coder.code("DM2").diagnoses
    steps = diagnosis.candidate.steps
    assert [(s.title, s.via) for s in steps[:2]] == [
        ("Mellitus, diabetes", ""),
        ("Diabetes, diabetic", "see"),
    ]
    assert steps[1].note  # the cross-reference as written
