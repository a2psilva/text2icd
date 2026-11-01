"""Parser for the ICD-10-CM Alphabetic Index XML files.

One schema covers four books: the Index to Diseases and Injuries (`icd10cm_index_<year>.xml`), the
External Cause of Injuries Index (`eindex`), the Neoplasm Table (`neoplasm`) and the Table of Drugs
and Chemicals (`drug`). The two tables add numbered columns (`<cell col="2">C76.2</cell>`).

A term's title is split the way a coder reads it: the essential words, and the nonessential
modifiers in parentheses, which may be present or absent without changing the code (I.A.7).
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IndexSource = Literal["index", "eindex", "neoplasm", "drug"]

NOT_APPLICABLE = frozenset({"-", "--"})
_SINGLE_VALUED = ("code", "see", "seeAlso", "seecat", "subcat", "manif")


@dataclass(frozen=True, slots=True)
class IndexTerm:
    """A main term (level 0) or a subterm. `id` is its position in the book."""

    id: int
    level: int
    title: str
    nonessential: tuple[str, ...]
    parent: int | None
    children: tuple[int, ...]
    letter: str
    printed_title: str = ""
    code: str | None = None
    see: str | None = None
    see_also: str | None = None
    see_category: str | None = None
    subcategory: str | None = None
    manifestation: str | None = None
    cells: tuple[tuple[int, str], ...] = ()

    def cell(self, column: int) -> str | None:
        """The code in a table column, or None when the cell is absent or marked not applicable.

        The Table of Drugs and Chemicals writes `--` for an empty cell, the Neoplasm Table `-`.
        """
        for col, value in self.cells:
            if col == column:
                return None if value in NOT_APPLICABLE else value
        return None

    @property
    def display_title(self) -> str:
        """The title as printed in the Index, modifiers in place: `(gastro)enteritis`."""
        return self.printed_title or self.title


@dataclass(frozen=True, slots=True)
class IndexBook:
    source: IndexSource
    version: str
    title: str
    columns: dict[int, str]
    terms: tuple[IndexTerm, ...]
    main_terms: tuple[int, ...]

    def __getitem__(self, term_id: int) -> IndexTerm:
        return self.terms[term_id]

    def __len__(self) -> int:
        return len(self.terms)

    def path(self, term_id: int) -> tuple[str, ...]:
        """Titles from the main term down to `term_id`: ("Pneumonia", "aspiration", "due to")."""
        titles = []
        current: int | None = term_id
        while current is not None:
            term = self.terms[current]
            titles.append(term.title)
            current = term.parent
        return tuple(reversed(titles))

    def descendants(self, term_id: int) -> Iterator[IndexTerm]:
        """All subterms below `term_id`, depth first, in Index order."""
        for child in self.terms[term_id].children:
            yield self.terms[child]
            yield from self.descendants(child)

    def find_main_term(self, title: str) -> IndexTerm | None:
        """The main term whose essential title matches `title` exactly (case-insensitive)."""
        wanted = title.casefold()
        for term_id in self.main_terms:
            if self.terms[term_id].title.casefold() == wanted:
                return self.terms[term_id]
        return None


def parse_index(path: Path, source: IndexSource) -> IndexBook:
    root = ET.parse(path).getroot()
    if root.tag != "ICD10CM.index":
        raise ValueError(f"{path}: expected <ICD10CM.index>, found <{root.tag}>")
    columns = {int(h.get("col", "0")): _clean(h.text) for h in root.iterfind("indexHeading/head")}
    records: list[IndexTerm | None] = []
    main_terms: list[int] = []
    for element in root:
        if element.tag == "letter":
            letter = _clean(element.findtext("title"))
            for main in element.iterfind("mainTerm"):
                main_terms.append(_add_term(main, 0, None, letter, records))
        elif element.tag not in {"version", "title", "indexHeading"}:
            raise ValueError(f"{path}: unexpected <{element.tag}> at the top level")
    terms = tuple(term for term in records if term is not None)
    assert len(terms) == len(records)
    return IndexBook(
        source=source,
        version=_clean(root.findtext("version")),
        title=_clean(root.findtext("title")),
        columns=columns,
        terms=terms,
        main_terms=tuple(main_terms),
    )


def _add_term(
    element: ET.Element,
    level: int,
    parent: int | None,
    letter: str,
    records: list[IndexTerm | None],
) -> int:
    if element.tag == "term":
        declared = int(element.get("level", "-1"))
        if declared != level:
            raise ValueError(f"term level {declared} under a level {level - 1} term ({parent})")
    term_id = len(records)
    records.append(None)
    title_element = element.find("title")
    if title_element is None:
        raise ValueError(f"index term without a title under {parent}")
    title, nonessential = _split_title(title_element)
    printed_title = _printed_title(title_element)

    fields: dict[str, str] = {}
    cells: list[tuple[int, str]] = []
    children: list[int] = []
    for child in element:
        tag = child.tag
        if tag == "title":
            continue
        if tag in _SINGLE_VALUED:
            if tag in fields:
                raise ValueError(f"more than one <{tag}> in index term {title!r}")
            fields[tag] = _clean(child.text)
        elif tag == "cell":
            cells.append((int(child.get("col", "0")), _clean(child.text)))
        elif tag == "term":
            children.append(_add_term(child, level + 1, term_id, letter, records))
        else:
            raise ValueError(f"unexpected <{tag}> in index term {title!r}")

    records[term_id] = IndexTerm(
        id=term_id,
        level=level,
        title=title,
        nonessential=nonessential,
        parent=parent,
        children=tuple(children),
        letter=letter,
        printed_title=printed_title,
        code=fields.get("code"),
        see=fields.get("see"),
        see_also=fields.get("seeAlso"),
        see_category=fields.get("seecat"),
        subcategory=fields.get("subcat"),
        manifestation=fields.get("manif"),
        cells=tuple(cells),
    )
    return term_id


def _split_title(title: ET.Element) -> tuple[str, tuple[str, ...]]:
    """`Abandonment<nemod>(a) (b)</nemod> NEC` -> ("Abandonment NEC", ("a", "b"))."""
    essential = [title.text or ""]
    modifiers: list[str] = []
    for child in title:
        if child.tag != "nemod":
            raise ValueError(f"unexpected <{child.tag}> in index title {_clean(title.text)!r}")
        words, groups = split_modifiers(child.text or "")
        essential.append(words)
        modifiers.extend(groups)
        essential.append(child.tail or "")
    return _clean(" ".join(essential)), tuple(modifiers)


def _printed_title(title: ET.Element) -> str:
    """The title as the printed Index shows it: a modifier after text is set off by a space."""
    parts = [title.text or ""]
    for child in title:
        if "".join(parts).strip() and not "".join(parts).endswith(" "):
            parts.append(" ")
        parts.extend((child.text or "", child.tail or ""))
    return _clean("".join(parts))


def split_modifiers(text: str) -> tuple[str, list[str]]:
    """Separate parenthesized modifiers from essential words inside a `<nemod>` element.

    `(on) chronic (congestive)` -> ("chronic", ["on", "congestive"]). A group starts only at the
    beginning or after a space or another group, so `cigarette(s)` stays one word; nested
    parentheses stay inside their group: `(except toe(s) alone)` -> ["except toe(s) alone"].
    """
    words: list[str] = []
    groups: list[str] = []
    i = 0
    while i < len(text):
        starts_group = text[i] == "(" and (i == 0 or text[i - 1].isspace() or text[i - 1] == ")")
        if not starts_group:
            words.append(text[i])
            i += 1
            continue
        depth, j = 0, i
        while j < len(text):
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        if depth:
            words.append(text[i:])
            break
        groups.append(_clean(text[i + 1 : j]))
        words.append(" ")
        i = j + 1
    return _clean("".join(words)), [g for g in groups if g]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())
