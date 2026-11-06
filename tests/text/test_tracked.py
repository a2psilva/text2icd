"""Task 9.11: spans of the normalised phrase map back to the phrase as written."""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.text.abbreviations import default_lexicon
from text2icd.text.model import ParsedPhrase
from text2icd.text.normalize import SHORTHAND, clean, clean_tracked, normalize
from text2icd.text.parse import Parser
from text2icd.text.tracked import Tracked, source_span

SAMPLE = Path("benchmarks/cms-synthetic/fy2027-sample.jsonl")
ODD = (
    "",
    "   ",
    "  • CHF exacerbation; s/p CABG ",
    "Ostéomyélite of L femur — chronic",
    "ﬁbrosis of lung, ½ lobe",
    "Café au lait spots",
    "STRASSE ß Straße",
    "pain​ in­knee\t\tleft",
    "(1) pneumonia , right lower lobe ;",
    "1,000 mg overdose,acetaminophen",
)


def sample_phrases() -> list[str]:
    with SAMPLE.open(encoding="utf-8") as fh:
        return [json.loads(line)["phrase"] for line in fh]


def test_identity() -> None:
    text = Tracked.of("abc")
    assert (text.starts, text.ends) == ((0, 1, 2), (1, 2, 3))
    assert text.origin(1, 3) == (1, 3)
    assert text.origin(1, 1) == (1, 1)
    assert text.origin(3, 3) == (3, 3)
    assert text.origin(2, 99) == (2, 3)
    assert Tracked.of("").origin(0, 0) == (0, 0)


def test_sub_keeps_what_it_does_not_change() -> None:
    text = Tracked.of("DM2 w/ CKD").sub(re.compile(r"w/"), "with")
    assert text.text == "DM2 with CKD"
    # "w" maps to "w"; the inserted "ith" to the "/" it replaces; "CKD" moved by two.
    assert text.origin(4, 5) == (4, 5)
    assert text.origin(5, 8) == (5, 6)
    assert text.origin(9, 12) == (7, 10)
    unchanged = Tracked.of("no match here")
    assert unchanged.sub(re.compile(r"zzz"), "y") is unchanged


def test_sub_maps_a_replacement_to_what_it_replaces() -> None:
    text = Tracked.of("type 2 DM").sub(re.compile(r"DM"), lambda _: "diabetes mellitus")
    assert text.text == "type 2 diabetes mellitus"
    assert text.origin(7, len(text.text)) == (7, 9)
    deleted = Tracked.of("a, b").sub(re.compile(r","), "")
    assert deleted.text == "a b"
    assert deleted.origin(1, 1) == (2, 2)  # an empty span sits where the comma was


def test_sub_is_re_sub() -> None:
    """The same text, and each callback called once per match, as `re.sub` would."""
    for phrase in [*ODD, *sample_phrases()]:
        for pattern, replace in SHORTHAND:
            calls: list[str] = []

            def counted(
                match: re.Match[str],
                replace: Callable[[re.Match[str]], str] = replace,
                calls: list[str] = calls,
            ) -> str:
                calls.append(match[0])
                return replace(match)

            assert Tracked.of(phrase).sub(pattern, counted).text == pattern.sub(replace, phrase)
            assert len(calls) == len(list(pattern.finditer(phrase)))


def test_casefold_and_nfkc() -> None:
    folded = Tracked.of("Straße").map_chars(str.casefold)
    assert folded.text == "strasse"
    assert folded.origin(4, 6) == (4, 5)  # "ss" comes from "ß"
    ligature = Tracked.of("ﬁbrosis").nfkc()
    assert ligature.text == "fibrosis"
    assert ligature.origin(0, 2) == (0, 1)
    assert ligature.origin(2, 3) == (1, 2)
    composed = Tracked.of("Café!").nfkc()
    assert composed.text == "Café!"
    assert composed.origin(3, 4) == (3, 5)  # é from e and its accent
    assert composed.origin(4, 5) == (5, 6)


def test_clean_tracked_is_clean() -> None:
    for phrase in [*ODD, *sample_phrases()]:
        assert clean_tracked(Tracked.of(phrase)).text == clean(phrase), phrase


def test_source_span_edges() -> None:
    assert source_span((0, 1), (1, 3), 0, 2) == (0, 3)
    assert source_span((0, 1), (1, 3), 2, 2) == (3, 3)
    assert source_span((), (), 0, 0) == (0, 0)
    bare = ParsedPhrase(original="abc", normalized="abc", setting="inpatient", conditions=())
    assert bare.source(1, 2) == (1, 2)  # no map: spans as they are
    assert bare.source(1, 9) == (1, 3)


@pytest.mark.parametrize(
    ("phrase", "chunks"),
    [
        ("DM2 w/ CKD 3a, possible UTI 2/2 E. coli", ["DM2", "CKD 3a", "possible UTI 2/2 E. coli"]),
        ("  • CHF exacerbation; s/p CABG", ["CHF exacerbation", "s/p CABG"]),
        ("Type II DM w/o complications", ["Type II DM w/o complications"]),
    ],
)
def test_chunks_map_back_to_what_was_written(phrase: str, chunks: list[str]) -> None:
    parsed = Parser().parse(phrase)
    found = [phrase[slice(*parsed.source(c.start, c.end))] for c in parsed.conditions]
    assert found == chunks


def test_expansions_and_relations_have_spans() -> None:
    phrase = "DM2 w/ CKD 3a"
    parsed = Parser().parse(phrase)
    assert [(e.written, phrase[e.start : e.end]) for e in parsed.expansions] == [
        ("DM2", "DM2"),
        ("CKD", "CKD"),
    ]
    (relation,) = parsed.relations
    assert relation.start is not None
    assert relation.end is not None
    assert phrase[slice(*parsed.source(relation.start, relation.end))] == "w/"
    modifiers = {
        m.kind: phrase[slice(*parsed.source(m.start, m.end))]
        for m in parsed.conditions[1].modifiers
    }
    assert modifiers["stage"] == "3a"


def test_spans_round_trip_on_the_sample() -> None:
    """Every expansion's span holds what was written; every condition's span is inside the
    phrase, in order, and not empty."""
    parser = Parser()
    for phrase in sample_phrases():
        parsed = parser.parse(phrase)
        assert len(parsed.starts) == len(parsed.normalized) == len(parsed.ends)
        for expansion in parsed.expansions:
            assert expansion.start is not None
            assert phrase[expansion.start : expansion.end] == expansion.written
        previous = 0
        for condition in parsed.conditions:
            start, end = parsed.source(condition.start, condition.end)
            assert 0 <= start < end <= len(phrase), phrase
            assert start >= previous, phrase
            previous = start


def test_normalize_records_the_map() -> None:
    result = normalize("CHF w/ AF", default_lexicon())
    assert result.text == "congestive heart failure with atrial fibrillation"
    assert len(result.starts) == len(result.text)
    assert source_span(result.starts, result.ends, 0, len("congestive heart failure")) == (0, 3)
