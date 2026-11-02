"""Index term profiles and how a term matches the words of a phrase."""

import pytest

from text2icd.graph.model import Graph
from text2icd.resolve.profiles import (
    MISSING,
    NONESSENTIAL,
    OTHER_PART,
    Profile,
    Profiles,
    match,
    slots,
    split_connector,
)
from text2icd.resolve.words import default_word_forms

FORMS = default_word_forms()


def make(title: str, nonessential: str = "", connector: str | None = None) -> Profile:
    parts = tuple(found for part in title.split(", ") if (found := slots(part, FORMS)))
    return Profile(0, parts, frozenset(FORMS.content(nonessential)), connector, frozenset())


def test_slots() -> None:
    assert slots("inhalation of food or vomit", FORMS) == (
        frozenset({"inhalation"}),
        frozenset({"food", "vomit"}),
    )
    assert slots("Fractures, NEC", FORMS) == (frozenset({"fracture"}),)
    assert slots("or", FORMS) == ()


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("due to food", ("due to", "food")),
        ("with", ("with", "")),
        ("without complication", (None, "without complication")),
        ("in (due to)", ("in", "(due to)")),
        ("within normal limits", (None, "within normal limits")),
        ("aspiration", (None, "aspiration")),
        ("Associated with pregnancy", ("associated with", "pregnancy")),
    ],
)
def test_split_connector(title: str, expected: tuple[str | None, str]) -> None:
    assert split_connector(title) == expected


def test_match_counts_used_words() -> None:
    found = match(make("aspiration"), {"aspiration", "pneumonia"})
    assert found.used == {"aspiration"}
    assert found.score == 1


def test_match_penalises_missing_essential_words() -> None:
    found = match(make("amniotic fluid"), {"fluid", "pneumonia"})
    assert (found.used, found.missing) == ({"fluid"}, 1)
    assert found.score == pytest.approx(1 - MISSING)
    assert match(make("newborn"), {"pneumonia"}).score == pytest.approx(-MISSING)


def test_match_nonessential_modifiers_are_optional() -> None:
    term = make("food", "(regurgitated)")
    assert match(term, {"food", "regurgitated"}).score == pytest.approx(1 + NONESSENTIAL)
    assert match(term, {"food"}).score == 1  # no penalty when absent
    assert match(term, {"food"}).consumed == {"food"}


def test_match_comma_parts() -> None:
    # Variants: either one does, the other costs only a little.
    found = match(make("Diabetes, diabetic"), {"diabetic", "type", "2"})
    assert found.used == {"diabetic"}
    assert found.score == pytest.approx(1 - OTHER_PART)
    # Qualifiers: each one the phrase holds counts.
    both = match(make("syphilitic, congenital"), {"syphilitic", "congenital"})
    assert both.used == {"syphilitic", "congenital"}
    assert both.score == 2
    # The best part decides what is missing.
    assert match(make("solids, liquids NEC"), {"liquid"}).missing == 0


def test_match_or_slots() -> None:
    found = match(make("inhalation of food or vomit"), {"inhalation", "vomit"})
    assert found.used == {"inhalation", "vomit"}
    assert found.missing == 0


def test_match_connector_only_term() -> None:
    found = match(make("", connector="with"), {"pneumonia"})
    assert found.score == 0
    assert not found.used


def test_profiles_of_the_fixture(graph: Graph) -> None:
    profiles = Profiles(graph, FORMS)
    pneumonia = graph.term_at("Pneumonia")
    food = graph.term_at("Pneumonia > aspiration > due to > food")
    due_to = graph.term_at("Pneumonia > aspiration > due to")
    assert pneumonia is not None
    assert food is not None
    assert due_to is not None
    main = profiles[pneumonia.id]
    assert main.parts == ((frozenset({"pneumonia"}),),)
    assert {"acute", "septic"} <= main.nonessential
    assert {"aspiration", "food", "regurgitated", "newborn", "meconium"} <= main.below
    assert profiles[food.id].nonessential == {"regurgitated"}
    assert profiles[due_to.id].connector == "due to"
    assert profiles[due_to.id].parts == ()
    assert profiles[due_to.id].below >= {"food", "vomitus"}
    assert profiles.main_terms_with("pneumonia") == (pneumonia.id,)
    assert profiles.main_terms_with("food") == ()
    assert profiles.main_terms_with("propanol", "drug")
    assert sum(1 for _ in profiles) == sum(1 for _ in graph.terms())
