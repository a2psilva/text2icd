"""Parser for the ICD-10-CM Tabular List XML (`icd10cm_tabular_<year>.xml`).

The Tabular List is a tree of chapters, sections (blocks such as `E08-E13`) and `diag` elements
(categories, subcategories and codes). Every node keeps its instructional notes in document order.
Nothing is interpreted here: billable status and 7th-character expansion are derived elsewhere.
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

NoteKind = Literal[
    "inclusionTerm",
    "includes",
    "excludes1",
    "excludes2",
    "codeFirst",
    "useAdditionalCode",
    "codeAlso",
    "notes",
    "sevenChrNote",
    "visualImpairment",
]
NodeKind = Literal["chapter", "section", "diag"]

NOTE_KINDS: frozenset[str] = frozenset(get_args(NoteKind)) - {"visualImpairment"}
_IGNORED = frozenset({"sectionIndex", "deactivated"})
_REMOVED_IN_ADDENDA = "old"


@dataclass(frozen=True, slots=True)
class Note:
    kind: NoteKind
    text: str


@dataclass(frozen=True, slots=True)
class TabularNode:
    """A chapter (key `"4"`), a section or a diag (key `"E11.22"`).

    Sections are keyed by their code range (`"E08-E13"`); a single-category block such as `B10` is
    keyed `"B10-B10"`, so it never collides with the diag of the same name.
    """

    key: str
    kind: NodeKind
    description: str
    parent: str | None
    children: tuple[str, ...]
    notes: tuple[Note, ...]
    seventh_characters: tuple[tuple[str, str], ...] = ()
    placeholder: bool = False

    def notes_of(self, kind: NoteKind) -> tuple[str, ...]:
        return tuple(note.text for note in self.notes if note.kind == kind)


@dataclass(frozen=True, slots=True)
class Tabular:
    version: str
    nodes: dict[str, TabularNode]
    chapters: tuple[str, ...]

    def __getitem__(self, key: str) -> TabularNode:
        return self.nodes[key]

    def __contains__(self, key: object) -> bool:
        return key in self.nodes

    def diags(self) -> Iterator[TabularNode]:
        """All diag nodes in Tabular List order."""
        return (node for node in self.nodes.values() if node.kind == "diag")

    def ancestors(self, key: str) -> Iterator[TabularNode]:
        """The parents of `key`, nearest first, up to its chapter."""
        parent = self.nodes[key].parent
        while parent is not None:
            node = self.nodes[parent]
            yield node
            parent = node.parent

    def seventh_characters_for(self, key: str) -> tuple[tuple[str, str], ...]:
        """The 7th-character definition that applies to `key`: its own or the nearest ancestor's."""
        owner = self.seventh_character_owner(key)
        return self.nodes[owner].seventh_characters if owner is not None else ()

    def seventh_character_owner(self, key: str) -> str | None:
        """The node whose 7th-character definition applies to `key`, or None."""
        if self.nodes[key].seventh_characters:
            return key
        for ancestor in self.ancestors(key):
            if ancestor.seventh_characters:
                return ancestor.key
        return None


def parse_tabular(path: Path) -> Tabular:
    root = ET.parse(path).getroot()
    if root.tag != "ICD10CM.tabular":
        raise ValueError(f"{path}: expected <ICD10CM.tabular>, found <{root.tag}>")
    nodes: dict[str, TabularNode] = {}
    chapters = []
    for element in root:
        if element.tag == "chapter":
            chapters.append(_build(element, "chapter", None, nodes))
        elif element.tag not in {"version", "title", "introduction"}:
            raise ValueError(f"{path}: unexpected <{element.tag}> at the top level")
    return Tabular(
        version=root.findtext("version", "").strip(), nodes=nodes, chapters=tuple(chapters)
    )


def _build(
    element: ET.Element, kind: NodeKind, parent: str | None, nodes: dict[str, TabularNode]
) -> str:
    key = _section_key(element.get("id", "")) if kind == "section" else _text(element.find("name"))
    if not key:
        raise ValueError(f"<{element.tag}> without a name or id under {parent}")
    if key in nodes:
        raise ValueError(f"duplicate Tabular key {key!r}")
    nodes[key] = TabularNode(key, kind, "", parent, (), ())

    notes: list[Note] = []
    sevenths: list[tuple[str, str]] = []
    children: list[str] = []
    for child in element:
        tag = child.tag
        if tag in {"name", "desc"} or tag in _IGNORED:
            continue
        if tag in NOTE_KINDS:
            kind_of_note = cast(NoteKind, tag)
            notes.extend(Note(kind_of_note, _text(n)) for n in child.findall("note"))
        elif tag == "sevenChrDef":
            sevenths.extend((ext.get("char", ""), _text(ext)) for ext in child.findall("extension"))
            notes.extend(Note("sevenChrNote", _text(n)) for n in child.findall("note"))
        elif tag == "visualImpairment":
            notes.append(Note("visualImpairment", " ".join(" ".join(child.itertext()).split())))
        elif tag == "section" and kind == "chapter":
            children.append(_build(child, "section", key, nodes))
        elif tag == "diag" and kind in {"section", "diag"}:
            children.append(_build(child, "diag", key, nodes))
        else:
            raise ValueError(f"unexpected <{tag}> inside {kind} {key}")

    nodes[key] = TabularNode(
        key=key,
        kind=kind,
        description=_text(element.find("desc")),
        parent=parent,
        children=tuple(children),
        notes=tuple(notes),
        seventh_characters=tuple(sevenths),
        placeholder=element.get("placeholder") == "true",
    )
    return key


def _section_key(section_id: str) -> str:
    if not section_id or "-" in section_id:
        return section_id
    return f"{section_id}-{section_id}"


def _text(element: ET.Element | None) -> str:
    """Text of a mixed-content element, whitespace collapsed, without addenda `<old>` text."""
    if element is None:
        return ""
    return " ".join(_raw_text(element).split())


def _raw_text(element: ET.Element) -> str:
    parts = [element.text or ""]
    for child in element:
        if child.tag != _REMOVED_IN_ADDENDA:
            parts.append(_raw_text(child))
        parts.append(child.tail or "")
    return "".join(parts)
