from pathlib import Path

import pytest

from text2icd.ontology.index import parse_index
from text2icd.ontology.tabular import parse_tabular
from text2icd.ontology.validate import (
    IndexReference,
    load_exceptions,
    references,
    unresolved_references,
)
from text2icd.releases import load_registry

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cms"


def index_with(tmp_path: Path, terms: str) -> Path:
    path = tmp_path / "index.xml"
    path.write_text(
        "<ICD10CM.index><version>2099</version><letter><title>X</title>"
        f"<mainTerm><title>Test</title>{terms}</mainTerm></letter></ICD10CM.index>"
    )
    return path


def test_references_cover_codes_manifestations_and_cells() -> None:
    drug = parse_index(FIXTURES / "drug_sample.xml", "drug")
    refs = list(references(drug))
    assert IndexReference("drug", "1-propanol", "cell 2", "T51.3X1") in refs
    assert all(ref.value not in {"-", "--"} for ref in refs)


def test_unresolved_references(tmp_path: Path) -> None:
    tabular = parse_tabular(FIXTURES / "tabular_sample.xml")
    terms = (
        "<term level='1'><title>known</title><code>E11.22</code></term>"
        "<term level='1'><title>partial</title><code>E11.2-</code></term>"
        "<term level='1'><title>billable only</title><code>T51.3X1A</code></term>"
        "<term level='1'><title>missing</title><code>E11.99</code></term>"
        "<term level='1'><title>text</title><manif>see note</manif></term>"
    )
    book = parse_index(index_with(tmp_path, terms), "index")
    unresolved = unresolved_references({"index": book}, tabular, frozenset({"T51.3X1A"}))
    assert unresolved == [
        IndexReference("index", "Test > missing", "code", "E11.99"),
        IndexReference("index", "Test > text", "manifestation", "see note"),
    ]


def test_shipped_exceptions_are_documented() -> None:
    known_releases = set(load_registry().releases)
    exceptions = load_exceptions()
    assert exceptions
    for exception in exceptions:
        assert exception.reason
        assert exception.releases <= known_releases
        assert exception.reference.field in {"code", "manifestation", "subcategory", "see_category"}
    corrected = [e for e in exceptions if e.code]
    assert {e.code for e in corrected} == {"D49.2", "E83.31", "D58.2", "T70.3"}


def test_exceptions_schema_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "exceptions.toml"
    path.write_text("schema = 2\n")
    with pytest.raises(ValueError, match="unsupported exceptions schema"):
        load_exceptions(path)
