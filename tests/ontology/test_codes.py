import pytest

from text2icd.ontology.codes import (
    CodeReference,
    category,
    dot,
    is_code,
    is_partial,
    parse_reference,
    strip_partial,
    undot,
    with_seventh_character,
)


@pytest.mark.parametrize(
    ("raw", "dotted"),
    [
        ("E1122", "E11.22"),
        ("E11.22", "E11.22"),
        (" e11.22 ", "E11.22"),
        ("I10", "I10"),
        ("Z87891", "Z87.891"),
        ("T513X1A", "T51.3X1A"),
        ("C4A0", "C4A.0"),
        ("U071", "U07.1"),
        ("QA00101", "QA0.0101"),
    ],
)
def test_dot_and_undot_round_trip(raw: str, dotted: str) -> None:
    assert dot(raw) == dotted
    assert undot(dotted) == dotted.replace(".", "")


@pytest.mark.parametrize(
    "bad", ["", "E1", "11.22", "1E1.22", "E11.22345", "E11-22", "E11.2/2", "HIV", "SCFE", "ABC.1"]
)
def test_invalid_codes_are_rejected(bad: str) -> None:
    assert not is_code(bad)
    with pytest.raises(ValueError, match="not an ICD-10-CM code"):
        dot(bad)


@pytest.mark.parametrize(
    "good", ["A00", "E11.22", "e1122", "M1A.0110", "Z3A.10", "T07.XXXA", "QA0.0101"]
)
def test_well_formed_codes(good: str) -> None:
    assert is_code(good)


def test_category() -> None:
    assert category("T51.3X1A") == "T51"
    assert category("C4A0") == "C4A"


@pytest.mark.parametrize(
    ("code", "seventh", "expected"),
    [
        ("T51.3X1", "A", "T51.3X1A"),
        ("S72.001", "D", "S72.001D"),
        ("T07", "A", "T07.XXXA"),
        ("S06.0", "S", "S06.0XXS"),
        ("E08.321", "1", "E08.3211"),
        ("O32.1", "0", "O32.1XX0"),
    ],
)
def test_seventh_character_is_padded_with_placeholders(
    code: str, seventh: str, expected: str
) -> None:
    assert with_seventh_character(code, seventh) == expected


def test_seventh_character_validation() -> None:
    with pytest.raises(ValueError, match="already has seven characters"):
        with_seventh_character("T51.3X1A", "A")
    with pytest.raises(ValueError, match="one letter or digit"):
        with_seventh_character("T51.3X1", "AB")


@pytest.mark.parametrize(
    ("code", "partial", "stripped"),
    [("C79.8-", True, "C79.8"), ("E10.-", True, "E10"), ("J69.0", False, "J69.0")],
)
def test_partial_codes(code: str, partial: bool, stripped: str) -> None:
    assert is_partial(code) is partial
    assert strip_partial(code) == stripped


@pytest.mark.parametrize(
    ("text", "inside", "outside"),
    [
        ("N18.1-N18.6", ["N18.1", "N18.31", "N18.6"], ["N18.9", "N18.0", "N19"]),
        ("J09-J18", ["J09.X1", "J12.82", "J18.9"], ["J19", "J06.9", "J20.0"]),
        ("A00-B99", ["A00.0", "B99.9", "B20"], ["C00.0", "Z99.89"]),
        ("E10.-", ["E10", "E10.9", "E10.3211"], ["E11.9", "E1"]),
        ("C79.8-", ["C79.81", "C79.89"], ["C79.9", "C79.7"]),
        ("K70.3", ["K70.30", "K70.31"], ["K70.4", "K70"]),
        ("S72.0-S72.9", ["S72.001A", "S72.92XA"], ["S73.0"]),
    ],
)
def test_references_contain_their_subdivisions(
    text: str, inside: list[str], outside: list[str]
) -> None:
    reference = parse_reference(text)
    for code in inside:
        assert code in reference, f"{code} should be in {text}"
    for code in outside:
        assert code not in reference, f"{code} should not be in {text}"


def test_reference_accepts_en_dash_and_undotted_codes() -> None:
    assert parse_reference("N18.1\N{EN DASH}N18.6") == CodeReference("N181", "N186")
    assert "N1831" in parse_reference("N181-N186")


def test_reference_str() -> None:
    assert str(parse_reference("J09-J18")) == "J09-J18"
    assert str(parse_reference("E10.-")) == "E10"


def test_non_codes_are_never_contained() -> None:
    reference = parse_reference("A00-Z99")
    assert "not a code" not in reference
    assert 42 not in reference


@pytest.mark.parametrize("bad", ["", "N18.6-N18.1", "diabetes", "E11.22-x", "--"])
def test_invalid_references_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match=r"code reference|comes before"):
        parse_reference(bad)
