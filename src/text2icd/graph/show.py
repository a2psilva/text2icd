"""Plain-text views of a graph node, for `text2icd graph show` and `text2icd graph code`."""

from text2icd.graph.model import Code, CrossRef, Edge, Graph, Pointer, Term

# Long lists (subterms, Index terms giving a code) are cut to this many lines.
LIST_LIMIT = 25


def show_term(graph: Graph, term: Term) -> list[str]:
    """An Index term: its modifiers, codes, cross-references and subterms."""
    lines = [f"{term.book}: {term.path}"]
    lines.append(f"  term {term.id}, level {term.level}, printed {term.printed_title!r}")
    if term.nonessential:
        lines.append("  nonessential: " + " ".join(f"({m})" for m in term.nonessential))
    for pointer in graph.pointers(term.id):
        lines.append("  " + _pointer(graph, term.book, pointer))
    for xref in graph.xrefs(term.id):
        lines.append("  " + _xref(graph, xref))
    subterms = graph.subterms(term.id)
    if subterms:
        lines.append(f"  subterms ({len(subterms)}):")
        lines += _limited([f"    {sub.printed_title}" for sub in subterms])
    return lines


def show_code(graph: Graph, code: Code) -> list[str]:
    """A Tabular code: where it sits, its 7th characters, notes with their edges, and who names
    it (Index terms and other notes)."""
    flags = [code.kind, "billable" if code.billable else "not billable"]
    if code.placeholder:
        flags.append("placeholder X")
    lines = [f"{code.key}  {code.description}", f"  {', '.join(flags)}, depth {code.depth}"]
    for ancestor in graph.ancestors(code.key):
        lines.append(f"  in {ancestor.key}  {ancestor.description}")
    seventh = graph.seven_char_def(code.key)
    if seventh:
        lines.append(f"  7th characters (from {code.seventh_from}):")
        lines += [f"    {char}  {meaning}" for char, meaning in seventh]
    lines += _notes(graph, code.key)
    for ancestor in graph.ancestors(code.key):
        inherited = _notes(graph, ancestor.key, indent="    ")
        if inherited:
            lines.append(f"  notes inherited from {ancestor.key}:")
            lines += inherited[1:]
    children = graph.children(code.key)
    if children:
        lines.append(f"  children ({len(children)}):")
        lines += _limited([f"    {child.key}  {child.description}" for child in children])
    terms = graph.pointers_to(code.key)
    if terms:
        lines.append(f"  given by Index terms ({len(terms)}):")
        lines += _limited([f"    {_term_label(graph, pointer)}" for pointer in terms])
    named = graph.edges_to(code.key)
    if named:
        lines.append(f"  named in notes ({len(named)}):")
        lines += _limited([f"    {edge.source} {edge.kind}: {edge.note}" for edge in named])
    return lines


def _notes(graph: Graph, key: str, indent: str = "    ") -> list[str]:
    notes = graph.notes(key)
    if not notes:
        return []
    edges: dict[int, list[Edge]] = {}
    for edge in graph.edges(key):
        edges.setdefault(edge.note_id, []).append(edge)
    lines = ["  notes:"]
    for note in notes:
        lines.append(f"{indent}{note.kind}: {note.text}")
        for edge in edges.get(note.id, []):
            target = edge.target or (edge.start if edge.start == edge.end else None)
            span = target or f"{edge.start}-{edge.end}"
            lines.append(f"{indent}  -> {span} [{edge.status}]")
    return lines


def _pointer(graph: Graph, book: str, pointer: Pointer) -> str:
    field = pointer.field
    if field == "cell":
        field = f"cell {pointer.column} ({graph.column_heading(book, pointer.column)})"
    if pointer.target is None:
        return f"{field}: {pointer.written!r} [{pointer.status}]"
    partial = " (incomplete)" if pointer.partial else ""
    description = graph.code(pointer.target).description
    return f"{field}: {pointer.written}{partial} -> {pointer.target}  {description}"


def _xref(graph: Graph, xref: CrossRef) -> str:
    kind = xref.kind.replace("_", " ")
    out = f"{kind} {xref.written!r} [{xref.status}]"
    if xref.target is not None:
        target = graph.term(xref.target)
        out += f" -> {target.book}: {target.path}"
    elif xref.book is not None:
        out += f" -> {xref.book}"
    if xref.column is not None and xref.book is not None:
        out += f", column {xref.column} ({graph.column_heading(xref.book, xref.column)})"
    if xref.instruction:
        out += f", instruction {xref.instruction!r}"
    if xref.remainder and xref.status == "partial":
        out += f", then {xref.remainder!r}"
    if xref.reason:
        out += f": {xref.reason}"
    return out


def _term_label(graph: Graph, pointer: Pointer) -> str:
    term = graph.term(pointer.term)
    field = f"cell {pointer.column}" if pointer.field == "cell" else pointer.field
    return f"{term.book}: {term.path} [{field}]"


def _limited(lines: list[str]) -> list[str]:
    if len(lines) <= LIST_LIMIT:
        return lines
    return [*lines[:LIST_LIMIT], f"    ... and {len(lines) - LIST_LIMIT} more"]
