"""What the graph routes return (task 10.3): a code, an Index term and a search, as JSON objects
with named fields (the graph's own rows are tuples). The UI's explorer reads these, and so does
the static demo, which saves them as files."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from text2icd.graph.model import Code, CrossRef, Edge, Graph, Note, Pointer, Term

Book = Literal["index", "eindex", "neoplasm", "drug"]
BOOKS: tuple[Book, ...] = ("index", "eindex", "neoplasm", "drug")


class _View(BaseModel):
    model_config = ConfigDict(frozen=True)


class CodeRow(_View):
    key: str
    kind: str  # chapter, section, category, subcategory, code...
    description: str
    billable: bool
    parent: str | None = None
    depth: int = 0
    placeholder: bool = False


class NoteRow(_View):
    id: int
    node: str
    kind: str  # includes, excludes1, codeFirst, useAdditionalCode...
    text: str


class EdgeRow(_View):
    source: str
    kind: str
    written: str  # the reference as the note writes it ("E11.2-", "N18.1-N18.6")
    target: str | None  # the code it resolves to, when one
    status: str
    note: str
    note_id: int


class TermRow(_View):
    id: int
    book: str
    title: str
    path: str  # "Pneumonia > aspiration > due to > food"
    level: int
    parent: int | None = None
    nonessential: tuple[str, ...] = ()


class PointerRow(_View):
    term: int
    column: int
    written: str  # the code as the Index writes it ("E11.2-")
    code: str | None  # the code it resolves to
    status: str
    path: str = ""  # the term's path, when the row is listed under a code


class XrefRow(_View):
    term: int  # the term the cross-reference is written under
    kind: str  # see, see_also, see_condition
    written: str
    status: str
    target: int | None
    target_path: str | None = None
    source_path: str | None = None
    column: int | None = None


class CodeView(_View):
    release: str
    code: CodeRow
    ancestors: tuple[CodeRow, ...]  # from the parent up to the chapter
    children: tuple[CodeRow, ...]
    seventh: tuple[tuple[str, str], ...]  # the 7th characters that apply: (character, meaning)
    notes: tuple[NoteRow, ...]  # its own, in document order
    inherited: tuple[NoteRow, ...]  # its ancestors'
    edges: tuple[EdgeRow, ...]  # what its notes name
    named_by: tuple[EdgeRow, ...]  # notes elsewhere that name it
    terms: tuple[PointerRow, ...]  # Index terms that give it


class TermView(_View):
    release: str
    term: TermRow
    ancestors: tuple[TermRow, ...]  # from the parent up to the main term
    subterms: tuple[TermRow, ...]
    pointers: tuple[PointerRow, ...]  # the codes it gives
    xrefs: tuple[XrefRow, ...]  # where it sends the coder
    xrefs_to: tuple[XrefRow, ...]  # terms that send the coder here


class IndexHitRow(_View):
    term: TermRow
    score: float  # BM25: lower is better


class TabularHitRow(_View):
    code: CodeRow
    kind: str  # the text matched: description, inclusion, includes
    text: str
    score: float


class SearchView(_View):
    release: str
    query: str
    index: tuple[IndexHitRow, ...]
    tabular: tuple[TabularHitRow, ...]


def code_row(code: Code) -> CodeRow:
    return CodeRow(
        key=code.key,
        kind=code.kind,
        description=code.description,
        billable=code.billable,
        parent=code.parent,
        depth=code.depth,
        placeholder=code.placeholder,
    )


def term_row(term: Term) -> TermRow:
    return TermRow(
        id=term.id,
        book=term.book,
        title=term.title,
        path=term.path,
        level=term.level,
        parent=term.parent,
        nonessential=term.nonessential,
    )


def _note(note: Note) -> NoteRow:
    return NoteRow(id=note.id, node=note.node, kind=note.kind, text=note.text)


def _edge(edge: Edge) -> EdgeRow:
    return EdgeRow(
        source=edge.source,
        kind=edge.kind,
        written=edge.written,
        target=edge.target,
        status=edge.status,
        note=edge.note,
        note_id=edge.note_id,
    )


def _pointer(pointer: Pointer, path: str = "") -> PointerRow:
    return PointerRow(
        term=pointer.term,
        column=pointer.column,
        written=pointer.written,
        code=pointer.target,
        status=pointer.status,
        path=path,
    )


def _xref(graph: Graph, xref: CrossRef) -> XrefRow:
    target = graph.term(xref.target) if xref.target is not None else None
    return XrefRow(
        term=xref.term,
        kind=xref.kind,
        written=xref.written,
        status=xref.status,
        target=xref.target,
        target_path=target.path if target is not None else None,
        source_path=graph.term(xref.term).path,
        column=xref.column,
    )


def code_view(graph: Graph, key: str) -> CodeView:
    """The code `key`; KeyError when the release lacks it."""
    code = graph.code(key)
    own = graph.notes(code.key)
    return CodeView(
        release=graph.release,
        code=code_row(code),
        ancestors=tuple(code_row(c) for c in graph.ancestors(code.key)),
        children=tuple(code_row(c) for c in graph.children(code.key)),
        seventh=graph.seven_char_def(code.key),
        notes=tuple(_note(n) for n in own),
        inherited=tuple(_note(n) for n in graph.notes(code.key, inherited=True)[len(own) :]),
        edges=tuple(_edge(e) for e in graph.edges(code.key)),
        named_by=tuple(_edge(e) for e in graph.edges_to(code.key)),
        terms=tuple(_pointer(p, graph.term(p.term).path) for p in graph.pointers_to(code.key)),
    )


def term_view(graph: Graph, term_id: int) -> TermView:
    """The Index term `term_id`; KeyError when the release lacks it."""
    term = graph.term(term_id)
    ancestors = []
    parent = graph.term_parent(term.id)
    while parent is not None:
        ancestors.append(term_row(parent))
        parent = graph.term_parent(parent.id)
    return TermView(
        release=graph.release,
        term=term_row(term),
        ancestors=tuple(ancestors),
        subterms=tuple(term_row(t) for t in graph.subterms(term.id)),
        pointers=tuple(_pointer(p) for p in graph.pointers(term.id)),
        xrefs=tuple(_xref(graph, x) for x in graph.xrefs(term.id)),
        xrefs_to=tuple(_xref(graph, x) for x in graph.xrefs_to(term.id)),
    )


def search_view(graph: Graph, query: str, book: Book | None = None, k: int = 20) -> SearchView:
    """The Index terms (of `book`, or every book) and Tabular texts that match `query`."""
    index = graph.search_index(query, k, book=book)
    tabular = graph.search_tabular(query, k) if book is None else []
    return SearchView(
        release=graph.release,
        query=query,
        index=tuple(IndexHitRow(term=term_row(graph.term(h.term)), score=h.score) for h in index),
        tabular=tuple(
            TabularHitRow(
                code=code_row(graph.code(h.node)), kind=h.kind, text=h.text, score=h.score
            )
            for h in tabular
        ),
    )
