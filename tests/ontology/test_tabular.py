from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.tabular import Tabular, parse_tabular

SAMPLE_CODES = [
    "E08", "E08.3", "E08.32", "E08.321", "E08.329", "E11", "E11.2", "E11.22",
    "T51", "T51.3", "T51.3X", "T51.3X1", "T51.3X2", "T51.3X3", "T51.3X4",
]  # fmt: skip


@pytest.fixture(scope="module")
def sample() -> Tabular:
    return parse_tabular(Path(__file__).parents[1] / "fixtures" / "cms" / "tabular_sample.xml")


def test_structure(sample: Tabular) -> None:
    assert sample.version == "2027"
    assert sample.chapters == ("4", "19")
    assert sample["4"].children == ("E08-E13",)
    assert sample["E08-E13"].kind == "section"
    assert sample["E08-E13"].description == "Diabetes mellitus (E08-E13)"
    assert sample["E08-E13"].children == ("E08", "E11")
    assert [node.key for node in sample.diags()] == SAMPLE_CODES


def test_parents_and_ancestors(sample: Tabular) -> None:
    assert sample["E11.22"].parent == "E11.2"
    assert [n.key for n in sample.ancestors("E11.22")] == ["E11.2", "E11", "E08-E13", "4"]
    assert sample["4"].parent is None


def test_descriptions_and_notes(sample: Tabular) -> None:
    e1122 = sample["E11.22"]
    assert e1122.description == "Type 2 diabetes mellitus with diabetic chronic kidney disease"
    assert e1122.notes_of("useAdditionalCode") == (
        "code to identify stage of chronic kidney disease (N18.1-N18.6)",
    )
    assert sample["E08"].notes_of("codeFirst")
    assert sample["T51-T65"].notes_of("notes")[0].startswith("When no intent is indicated")


def test_seventh_characters_are_inherited(sample: Tabular) -> None:
    assert sample["T51"].seventh_characters == (
        ("A", "initial encounter"),
        ("D", "subsequent encounter"),
        ("S", "sequela"),
    )
    assert sample["T51.3X1"].seventh_characters == ()
    assert sample.seventh_characters_for("T51.3X1") == sample["T51"].seventh_characters
    assert [c for c, _ in sample.seventh_characters_for("E08.321")] == ["1", "2", "3", "9"]
    assert sample.seventh_characters_for("E11.22") == ()
    assert sample.seventh_character_owner("T51.3X1") == "T51"
    assert sample.seventh_character_owner("E08.32") == "E08.32"
    assert sample.seventh_character_owner("E11.22") is None
    assert sample["E08.32"].notes_of("sevenChrNote")


def test_single_category_section_does_not_collide_with_its_diag(tmp_path: Path) -> None:
    body = (
        "<chapter><name>1</name><desc>C</desc><section id='B10'><desc>Herpes (B10)</desc>"
        "<diag><name>B10</name><desc>Other human herpesviruses</desc></diag></section></chapter>"
    )
    tabular = parse_tabular(write(tmp_path, body))
    assert tabular["B10-B10"].kind == "section"
    assert tabular["B10"].parent == "B10-B10"


def test_placeholder_flag(sample: Tabular) -> None:
    assert sample["T51.3X"].placeholder is True
    assert sample["T51.3X1"].placeholder is False


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tabular.xml"
    path.write_text(f"<ICD10CM.tabular><version>2099</version>{body}</ICD10CM.tabular>")
    return path


def test_addenda_markup_keeps_new_text_and_drops_old(tmp_path: Path) -> None:
    body = (
        "<chapter><name>1</name><desc>Chapter</desc><section id='A00-A09'><desc>S</desc>"
        "<diag><name>A00</name><desc>Cholera<old> (old wording)</old><new>, all types</new></desc>"
        "</diag></section></chapter>"
    )
    assert parse_tabular(write(tmp_path, body))["A00"].description == "Cholera, all types"


def test_visual_impairment_table_is_kept_as_text(tmp_path: Path) -> None:
    body = (
        "<chapter><name>7</name><desc>Eye</desc><section id='H53-H54'><desc>S</desc>"
        "<diag><name>H54</name><desc>Blindness</desc><visualImpairment><visCategory>"
        "<heading>Category</heading><value>1</value></visCategory></visualImpairment>"
        "</diag></section></chapter>"
    )
    assert parse_tabular(write(tmp_path, body))["H54"].notes_of("visualImpairment") == (
        "Category 1",
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<chapter><name>1</name><surprise/></chapter>", "unexpected <surprise> inside chapter 1"),
        ("<chapter><name>1</name><diag><name>A00</name></diag></chapter>", "unexpected <diag>"),
        ("<appendix/>", "unexpected <appendix> at the top level"),
        (
            "<chapter><name>1</name><section id='A00'><diag><name>A00</name></diag>"
            "<diag><name>A00</name></diag></section></chapter>",
            "duplicate Tabular key 'A00'",
        ),
        ("<chapter><desc>no name</desc></chapter>", "without a name or id"),
    ],
)
def test_unexpected_structure_is_an_error(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_tabular(write(tmp_path, body))


def test_wrong_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "index.xml"
    path.write_text("<ICD10CM.index/>")
    with pytest.raises(ValueError, match=r"expected <ICD10CM\.tabular>"):
        parse_tabular(path)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("release_id", "diags", "definitions", "placeholders"),
    [("fy2026", 46_881, 508, 246), ("fy2026-apr", 46_881, 508, 246), ("fy2027", 47_025, 508, 245)],
)
def test_full_release(
    cms_release: Callable[[str], ReleaseFiles],
    release_id: str,
    diags: int,
    definitions: int,
    placeholders: int,
) -> None:
    tabular = parse_tabular(cms_release(release_id).tabular)
    nodes = list(tabular.diags())
    assert len(nodes) == diags
    assert sum(1 for n in tabular.nodes.values() if n.seventh_characters) == definitions
    assert sum(1 for n in nodes if n.placeholder) == placeholders
    assert len(tabular.chapters) == 22
    for node in tabular.nodes.values():
        assert node.description, node.key
        assert node.parent is None or node.parent in tabular
        assert all(child in tabular for child in node.children)
    assert tabular["E11.22"].notes_of("useAdditionalCode")
