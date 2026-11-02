"""The Index router: book, column and what is looked up."""

import pytest

from text2icd.resolve.route import route


@pytest.mark.parametrize(
    ("phrase", "book", "column", "text", "flags"),
    [
        ("malignant neoplasm of upper lobe of lung", "neoplasm", 2, "upper lobe of lung", ()),
        ("cancer of omentum", "neoplasm", 2, "omentum", ()),
        ("secondary malignant neoplasm of liver", "neoplasm", 3, "liver", ()),
        ("metastatic carcinoma to bone", "neoplasm", 3, "bone", ()),
        ("carcinoma in situ of cervix", "neoplasm", 4, "cervix", ()),
        ("benign neoplasm of colon", "neoplasm", 5, "colon", ()),
        ("neoplasm of uncertain behavior of ovary", "neoplasm", 6, "ovary", ()),
        ("neoplasm of unspecified behavior of bladder", "neoplasm", 7, "bladder", ()),
        ("neoplasm of skin", "neoplasm", 7, "skin", ()),  # no behaviour written
        ("accidental poisoning by aspirin", "drug", 2, "aspirin", ()),
        ("aspirin overdose", "drug", 2, "aspirin", ("intent_unknown_accidental",)),
        ("intentional self-poisoning with heroin", "drug", 3, "heroin", ()),
        ("assault by poisoning with arsenic", "drug", 4, "arsenic", ()),
        ("poisoning by amidone, undetermined intent", "drug", 5, "amidone", ()),
        ("adverse effect of digoxin", "drug", 6, "digoxin", ()),
        ("underdosing of insulin", "drug", 7, "insulin", ()),
        ("digoxin toxicity", "drug", 6, "digoxin", ("toxicity_as_adverse_effect",)),
        ("toxic effect of lead", "drug", 2, "lead", ("intent_unknown_accidental",)),
    ],
)
def test_tables(phrase: str, book: str, column: int, text: str, flags: tuple[str, ...]) -> None:
    (found,) = route(phrase)
    assert (found.book, found.column, found.text, found.flags) == (book, column, text, flags)


def test_cue_words_are_kept_once() -> None:
    (found,) = route("assault by poisoning with arsenic")
    assert found.cue == "assault by poisoning with"
    assert found.guideline == "I.C.19.e"
    assert route("aspirin overdose")[0].guideline == "I.C.19.e.5(b)"


def test_external_cause_only_when_stated() -> None:
    (fall,) = route("hip fracture due to fall")
    assert (fall.book, fall.column, fall.cue, fall.guideline) == ("eindex", None, "fall", "I.C.20")
    assert route("hip fracture") == []
    # Poisoning by assault is a Drug Table column, not an external cause code.
    assert [found.book for found in route("assault by poisoning with arsenic")] == ["drug"]


def test_main_index_only() -> None:
    assert route("aspiration pneumonia due to food") == []
    assert route("adenocarcinoma of stomach") == []  # morphology: the Index sends it on
