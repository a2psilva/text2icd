from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.index import IndexBook, IndexSource, parse_index, split_modifiers

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cms"


def load(book: IndexSource) -> IndexBook:
    return parse_index(FIXTURES / f"{book}_sample.xml", book)


@pytest.fixture(scope="module")
def index() -> IndexBook:
    return load("index")


def child(book: IndexBook, term_id: int, title: str) -> int:
    return next(c for c in book[term_id].children if book[c].title == title)


def test_book_metadata(index: IndexBook) -> None:
    assert index.source == "index"
    assert index.version == "2027"
    assert index.title == "ICD-10-CM INDEX TO DISEASES and INJURIES"
    assert index.columns == {}
    titles = [index[m].title for m in index.main_terms]
    assert titles == [
        "Abandonment",
        "Abdomen, abdominal",
        "Amyloid heart",
        "Bonnier's syndrome",
        "Pneumonia",
        "Pyelitis",
    ]


def test_cross_references_and_codes(index: IndexBook) -> None:
    abandonment = index.find_main_term("abandonment")
    assert abandonment is not None
    assert abandonment.see == "Maltreatment"
    assert abandonment.code is None

    abdomen = index.find_main_term("Abdomen, abdominal")
    assert abdomen is not None
    assert abdomen.see_also == "condition"
    assert index[child(index, abdomen.id, "acute")].code == "R10.0"

    amyloid = index.find_main_term("Amyloid heart")
    assert amyloid is not None
    assert (amyloid.code, amyloid.manifestation, amyloid.nonessential) == (
        "E85.4",
        "I43",
        ("disease",),
    )
    assert amyloid.display_title == "Amyloid heart (disease)"

    bonnier = index.find_main_term("Bonnier's syndrome")
    assert bonnier is not None
    assert bonnier.subcategory == "H81.8"

    pyelitis = index.find_main_term("Pyelitis")
    assert pyelitis is not None
    assert any(t.see_category for t in index.descendants(pyelitis.id))


def test_subterms_levels_and_paths(index: IndexBook) -> None:
    pneumonia = index.find_main_term("Pneumonia")
    assert pneumonia is not None
    assert pneumonia.code == "J18.9"
    assert pneumonia.nonessential == (
        "acute",
        "double",
        "migratory",
        "purulent",
        "septic",
        "unresolved",
    )
    aspiration = child(index, pneumonia.id, "aspiration")
    due_to = child(index, aspiration, "due to")
    food = child(index, due_to, "food")
    assert index[food].level == 3
    assert index[food].nonessential == ("regurgitated",)
    assert index[food].code == "J69.0"
    assert index.path(food) == ("Pneumonia", "aspiration", "due to", "food")
    paths = index.paths()
    assert len(paths) == len(index)
    assert paths[food] == "Pneumonia > aspiration > due to > food"
    assert index.paths(" / ")[food] == "Pneumonia / aspiration / due to / food"
    assert all(paths[t.id] == " > ".join(index.path(t.id)) for t in index.terms)
    with_ = child(index, pneumonia.id, "with")
    influenza = child(index, with_, "influenza")
    assert index[influenza].see == "Influenza, with, pneumonia"
    assert index[influenza].letter == "P"


def test_terms_are_numbered_in_index_order(index: IndexBook) -> None:
    for term in index.terms:
        assert index[term.id] is term
        assert all(c > term.id for c in term.children)
        if term.parent is not None:
            assert index[term.parent].level == term.level - 1


def test_external_cause_title_keeps_text_after_modifiers() -> None:
    eindex = load("eindex")
    abandonment = eindex.find_main_term("Abandonment NEC")
    assert abandonment is not None
    assert abandonment.nonessential == (
        "causing exposure to weather conditions",
        "with intent to injure or kill",
    )
    assert abandonment.code == "X58"
    assert abandonment.display_title == (
        "Abandonment (causing exposure to weather conditions) (with intent to injure or kill) NEC"
    )


def test_neoplasm_table_cells() -> None:
    neoplasm = load("neoplasm")
    assert neoplasm.columns[2] == "Malignant Primary"
    assert neoplasm.columns[7] == "Unspecified Behavior"
    main = neoplasm[neoplasm.main_terms[0]]
    assert main.title == "Neoplasm, neoplastic"
    assert main.cell(2) == "C80.1"
    abdomen = neoplasm[child(neoplasm, main.id, "abdomen, abdominal")]
    assert abdomen.cell(3) == "C79.8-"
    assert neoplasm.path(child(neoplasm, abdomen.id, "cavity")) == (
        "Neoplasm, neoplastic",
        "abdomen, abdominal",
        "cavity",
    )


def test_drug_table_cells_and_not_applicable() -> None:
    drug = load("drug")
    assert drug.columns[6] == "Adverse effect"
    propanol = drug.find_main_term("1-propanol")
    assert propanol is not None
    assert propanol.letter == "#"
    assert [propanol.cell(c) for c in range(2, 8)] == [
        "T51.3X1",
        "T51.3X2",
        "T51.3X3",
        "T51.3X4",
        None,
        None,
    ]
    assert propanol.cell(9) is None
    assert drug.find_main_term("2,4-D") is not None
    assert drug.find_main_term("not a substance") is None


@pytest.mark.parametrize(
    ("text", "words", "groups"),
    [
        ("(acute) (double)", "", ["acute", "double"]),
        ("(on) chronic (congestive)", "chronic", ["on", "congestive"]),
        ("(except toe(s) alone)", "", ["except toe(s) alone"]),
        ("((primarily) for)", "", ["(primarily) for"]),
        ("(s) or cigarette(s)", "or cigarette(s)", ["s"]),
        ("(-ortho-)cresol (pesticide)", "cresol", ["-ortho-", "pesticide"]),
        ("(unbalanced", "(unbalanced", []),
        ("", "", []),
    ],
)
def test_split_modifiers(text: str, words: str, groups: list[str]) -> None:
    assert split_modifiers(text) == (words, groups)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "index.xml"
    path.write_text(f"<ICD10CM.index><version>2099</version>{body}</ICD10CM.index>")
    return path


def test_modifier_attached_to_a_word_keeps_the_printed_title(tmp_path: Path) -> None:
    body = (
        "<letter><title>G</title><mainTerm><title><nemod>(gastro)</nemod>enteritis</title>"
        "<code>K52.9</code></mainTerm></letter>"
    )
    term = parse_index(write(tmp_path, body), "index")[0]
    assert (term.title, term.nonessential) == ("enteritis", ("gastro",))
    assert term.display_title == "(gastro)enteritis"


def test_essential_words_inside_nemod_join_the_title(tmp_path: Path) -> None:
    body = (
        "<letter><title>F</title><mainTerm><title>Failure</title><term level='1'>"
        "<title>heart<nemod>(acute) (on) chronic (congestive)</nemod></title><code>I50.9</code>"
        "</term></mainTerm></letter>"
    )
    book = parse_index(write(tmp_path, body), "index")
    heart = book[book[book.main_terms[0]].children[0]]
    assert heart.title == "heart chronic"
    assert heart.nonessential == ("acute", "on", "congestive")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "<letter><title>A</title><mainTerm><code>A00</code></mainTerm></letter>",
            "without a title",
        ),
        (
            "<letter><title>A</title><mainTerm><title>A</title><term level='2'><title>b</title>"
            "</term></mainTerm></letter>",
            "term level 2 under a level 0 term",
        ),
        (
            "<letter><title>A</title><mainTerm><title>A</title><code>A00</code><code>A01</code>"
            "</mainTerm></letter>",
            "more than one <code>",
        ),
        (
            "<letter><title>A</title><mainTerm><title>A</title><extra/></mainTerm></letter>",
            "unexpected <extra>",
        ),
        (
            "<letter><title>A</title><mainTerm><title>A<b>x</b></title></mainTerm></letter>",
            "unexpected <b> in index title",
        ),
        ("<appendix/>", "unexpected <appendix> at the top level"),
    ],
)
def test_unexpected_structure_is_an_error(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_index(write(tmp_path, body), "index")


def test_wrong_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tabular.xml"
    path.write_text("<ICD10CM.tabular/>")
    with pytest.raises(ValueError, match=r"expected <ICD10CM\.index>"):
        parse_index(path, "index")


EXPECTED = {
    # release: {book: (main terms, all terms, codes, see, see also, manifestations, cells)}
    "fy2026": {
        "index": (8121, 81509, 63138, 12139, 2063, 769, 0),
        "eindex": (314, 8090, 6671, 791, 98, 0, 0),
        "neoplasm": (1, 1940, 0, 67, 120, 1, 10922),
        "drug": (5179, 7430, 0, 97, 35, 0, 43224),
    },
    "fy2026-apr": {
        "index": (8122, 81515, 63142, 12141, 2063, 769, 0),
        "eindex": (314, 8090, 6671, 791, 98, 0, 0),
        "neoplasm": (1, 1940, 0, 67, 120, 1, 10922),
        "drug": (5179, 7430, 0, 97, 35, 0, 43224),
    },
    "fy2027": {
        "index": (8130, 81642, 63260, 12147, 2062, 769, 0),
        "eindex": (314, 8090, 6671, 788, 98, 0, 0),
        "neoplasm": (1, 1940, 0, 67, 120, 1, 10922),
        "drug": (5184, 7436, 0, 97, 35, 0, 43260),
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("release_id", list(EXPECTED))
def test_full_release(cms_release: Callable[[str], ReleaseFiles], release_id: str) -> None:
    files = cms_release(release_id)
    paths: dict[IndexSource, Path] = {
        "index": files.index,
        "eindex": files.eindex,
        "neoplasm": files.neoplasm,
        "drug": files.drug,
    }
    for source, path in paths.items():
        book = parse_index(path, source)
        counts = (
            len(book.main_terms),
            len(book),
            sum(1 for t in book.terms if t.code),
            sum(1 for t in book.terms if t.see),
            sum(1 for t in book.terms if t.see_also),
            sum(1 for t in book.terms if t.manifestation),
            sum(len(t.cells) for t in book.terms),
        )
        assert counts == EXPECTED[release_id][source], source
        # One real term has only a modifier as its title: Shock > electric > (taser) -> T75.4.
        assert all(t.title or t.nonessential for t in book.terms), source
