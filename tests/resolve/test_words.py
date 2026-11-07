"""Word forms and typo correction of the Index walk."""

import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.graph.build import BuildReport
from text2icd.graph.model import load
from text2icd.resolve.words import (
    Speller,
    WordForms,
    WordFormsError,
    default_word_forms,
    singular,
)
from text2icd.text.terms import words

FORMS = default_word_forms()


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("fractures", "fracture"),
        ("injuries", "injury"),
        ("ovaries", "ovary"),
        ("abscesses", "abscess"),
        ("abscess", "abscess"),
        ("headaches", "headache"),
        ("branches", "branch"),
        ("rashes", "rash"),
        ("boxes", "box"),
        ("vertebrae", "vertebra"),
        ("stenoses", "stenosis"),
        ("thromboses", "thrombosis"),
        ("metastases", "metastasis"),
        ("kidneys", "kidney"),
        ("ribs", "rib"),
        ("diseases", "disease"),
        ("arthritis", "arthritis"),
        ("pelvis", "pelvis"),
        ("fetus", "fetus"),
        ("nervous", "nervous"),
        ("pancreas", "pancreas"),
        ("pneumonias", "pneumonia"),
        ("hematomas", "hematoma"),
        ("gas", "gas"),
        ("legs", "leg"),
        ("eyes", "eye"),
        ("doses", "dose"),  # too short for -oses
        ("t2dm", "t2dm"),  # not all letters
        ("2", "2"),
    ],
)
def test_singular(word: str, expected: str) -> None:
    assert singular(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("renal", "kidney"),
        ("kidney", "kidney"),
        ("kidneys", "kidney"),
        ("hepatic", "liver"),
        ("cardiac", "heart"),
        ("pulmonary", "lung"),
        ("lungs", "lung"),
        ("calculi", "calculus"),
        ("calculus", "calculus"),
        ("teeth", "tooth"),
        ("dental", "tooth"),
        ("cervical", "cervical"),  # neck or cervix: left alone
        ("cystic", "cystic"),
    ],
)
def test_form(word: str, expected: str) -> None:
    assert FORMS.form(word) == expected


def test_content_words() -> None:
    assert FORMS.content("Renal failure, acute") == ["kidney", "failure", "acute"]
    assert FORMS.content("Fractures of the ribs") == ["fracture", "rib"]
    assert FORMS.content("aspiration pneumonia due to food") == ["aspiration", "pneumonia", "food"]
    # `without` decides between subterms, so it is kept; NEC and NOS are not.
    assert FORMS.content("pneumonia without complication NOS") == [
        "pneumonia",
        "without",
        "complication",
    ]
    assert FORMS.content("Schönlein's purpura") == ["schonlein", "purpura"]
    assert FORMS.content("type 2 diabetes") == ["type", "2", "diabete"]
    assert FORMS.content("of the") == []


def test_load_errors(tmp_path: Path) -> None:
    path = tmp_path / "forms.toml"
    path.write_text("schema = 2\n")
    with pytest.raises(WordFormsError, match="schema"):
        WordForms.load(path)
    path.write_text("schema = 1\nadjectives = 3\n")
    with pytest.raises(WordFormsError, match="adjectives"):
        WordForms.load(path)
    path.write_text("schema = 1\n[plurals]\nfeet = 1\n")
    with pytest.raises(WordFormsError, match="plurals"):
        WordForms.load(path)
    path.write_text('schema = 1\n[adjectives]\nRenal = "Kidney"\n')
    loaded = WordForms.load(path)
    assert loaded.form("renal") == "kidney"
    assert loaded.plurals == {}


def test_speller() -> None:
    speller = Speller({"systemic": 5, "system": 3, "cells": 2, "cecs": 9, "pneumonia": 1})
    assert speller.correct("systeimc") == "systemic"  # a swap is one edit
    assert speller.correct("sytsemic") == "systemic"
    assert speller.correct("celsl") == "cells"
    assert speller.correct("cels") is None  # under five letters
    assert speller.correct("systemic") is None  # an Index word
    assert speller.correct("cxllz") is None  # two edits in a short word
    assert speller.correct("pnumoniaa") == "pneumonia"  # two edits in a long word
    assert speller.correct("pnmoniaaa") is None  # three edits
    assert speller.correct("x-ray") is None
    assert Speller({"cells": 1}, known={"celss"}).correct("celss") is None


def test_speller_prefers_the_more_frequent_word() -> None:
    speller = Speller({"aaaab": 1, "aaaac": 5})
    assert speller.correct("aaaaa") == "aaaac"
    speller = Speller.from_texts(["Cells, cellss", "cellss"])
    assert speller.correct("cellsx") == "cellss"
    assert speller.known("cells")


@pytest.mark.parametrize(
    "text",
    ["Aspiration PNEUMONIA, due to food", "Schönlein's purpura", "Ⅳ stage", "ﬁbrosis", "", "C3-4"],
)
def test_words_fast_path(text: str) -> None:
    """Task 9.12: ASCII text skips the decomposition, with the same words."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    assert words(text) == re.findall(r"[a-z0-9]+", plain)


def test_speller_reads_only_words_of_a_near_length() -> None:
    """Task 9.12: a word more letters longer or shorter than the edits allowed is never the
    nearest, so the speller leaves it out; the answer and its ties are unchanged."""
    counts = {"pneumonia": 3, "pneumonias": 1, "pneumoniae": 1, "pneumon": 4, "anemia": 2}
    speller = Speller(counts)
    full = Speller(counts)
    full._within = lambda length, limit: full._choices  # type: ignore[method-assign]
    for word in ("pneumonai", "pneumoniaes", "pnemonia", "pneumoni", "anemai", "xyzzyx"):
        assert speller.correct(word) == full.correct(word), word
    assert speller.correct("pneumoniaes") == "pneumoniae"  # 1 edit; before pneumonias by name


@pytest.mark.slow
def test_lexicon_nouns_are_index_words(release_graph: Callable[[str], BuildReport]) -> None:
    with load(path=release_graph("fy2027").path) as graph:
        vocabulary: set[str] = set()
        for term in graph.terms():
            vocabulary.update(words(term.title))
            for modifier in term.nonessential:
                vocabulary.update(words(modifier))
    assert not {noun for noun in FORMS.adjectives.values() if noun not in vocabulary}
    assert not {word for word in FORMS.plurals.values() if word not in vocabulary}
    assert not {adjective for adjective in FORMS.adjectives if adjective not in vocabulary}
