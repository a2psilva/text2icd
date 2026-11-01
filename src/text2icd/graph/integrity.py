"""Integrity checks of a built graph, run on the SQLite file: `text2icd graph check`.

Each check passes, warns (a known kind of error in the CMS data that the graph records, such as a
note naming a code that does not exist) or fails. A graph with a failing check must not be used.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from text2icd.graph.build import content_hash

Outcome = Literal["pass", "warn", "fail"]
# Problems listed per check; the count is always complete.
EXAMPLES = 5
# Unresolved cross-references above this share fail the check, even when each has a reason.
MAX_UNRESOLVED_XREFS = 0.01


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    outcome: Outcome
    summary: str
    problems: tuple[str, ...] = field(default=())

    def lines(self) -> list[str]:
        out = [f"{self.outcome.upper():<4}  {self.name}: {self.summary}"]
        out += [f"        {problem}" for problem in self.problems[:EXAMPLES]]
        if len(self.problems) > EXAMPLES:
            out.append(f"        ... and {len(self.problems) - EXAMPLES} more")
        return out


Findings = tuple[list[str], str]  # the problems found, and a one-line summary


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    run: Callable[[sqlite3.Connection], Findings]
    warn: bool = False  # the problems are known errors in the CMS data: warn, do not fail


def check_graph(conn: sqlite3.Connection) -> list[CheckResult]:
    """Run every check on an open graph file. A check the file cannot answer (a missing table, a
    file built by another version) fails with the database error."""
    results = []
    for check in CHECKS:
        try:
            problems, summary = check.run(conn)
        except sqlite3.Error as exc:
            results.append(CheckResult(check.name, "fail", f"database error: {exc}"))
            continue
        outcome: Outcome = "pass" if not problems else "warn" if check.warn else "fail"
        results.append(CheckResult(check.name, outcome, summary, tuple(problems)))
    return results


def failed(results: list[CheckResult]) -> bool:
    return any(result.outcome == "fail" for result in results)


def _rows(conn: sqlite3.Connection, sql: str) -> list[str]:
    return [" ".join(str(value) for value in row) for row in conn.execute(sql)]


def _count(conn: sqlite3.Connection, sql: str) -> int:
    (count,) = conn.execute(sql).fetchone()
    return int(count)


def check_content_hash(conn: sqlite3.Connection) -> Findings:
    """The rows still hash to `meta.content_sha256`: the file was not changed after the build."""
    stored = conn.execute("SELECT value FROM meta WHERE key = 'content_sha256'").fetchone()
    actual = content_hash(conn)
    problems = [] if stored and stored[0] == actual else [f"meta {stored}, rows {actual}"]
    return problems, actual[:16]


def check_tabular_tree(conn: sqlite3.Connection) -> Findings:
    """Every parent exists, depths follow the parents, and every node is reached from a chapter
    (a node on a cycle would not be)."""
    problems = _rows(
        conn,
        "SELECT c.key, 'parent', c.parent FROM tabular_node c LEFT JOIN tabular_node p "
        "ON p.key = c.parent WHERE c.parent IS NOT NULL AND p.key IS NULL "
        "UNION ALL SELECT c.key, 'depth', c.depth FROM tabular_node c JOIN tabular_node p "
        "ON p.key = c.parent WHERE c.depth != p.depth + 1 "
        "UNION ALL SELECT key, 'root', kind FROM tabular_node "
        "WHERE (parent IS NULL) != (kind = 'chapter')",
    )
    total = _count(conn, "SELECT count(*) FROM tabular_node")
    reached = _count(
        conn,
        "WITH RECURSIVE tree(key) AS (SELECT key FROM tabular_node WHERE parent IS NULL "
        "UNION ALL SELECT c.key FROM tabular_node c JOIN tree ON c.parent = tree.key) "
        "SELECT count(*) FROM tree",
    )
    if reached != total:
        problems.append(f"{total - reached} nodes not reached from a chapter")
    return problems, f"{total:,} nodes"


def check_billable(conn: sqlite3.Connection) -> Findings:
    """The billable Tabular nodes are exactly the billable codes of the CMS order file."""
    problems = _rows(
        conn,
        "SELECT 'not in the order file:', key FROM tabular_node WHERE billable = 1 "
        "AND key NOT IN (SELECT code FROM order_code WHERE billable = 1) "
        "UNION ALL SELECT 'not in the Tabular List:', code FROM order_code WHERE billable = 1 "
        "AND code NOT IN (SELECT key FROM tabular_node WHERE billable = 1)",
    )
    count = _count(conn, "SELECT count(*) FROM tabular_node WHERE billable = 1")
    return problems, f"{count:,} equal to the order file"


def check_index_tree(conn: sqlite3.Connection) -> Findings:
    """Every subterm's parent exists in the same book one level up, and every term is reached
    from a main term."""
    problems = _rows(
        conn,
        "SELECT c.id, c.path FROM index_term c LEFT JOIN index_term p ON p.id = c.parent "
        "WHERE (c.parent IS NULL) != (c.level = 0) OR (c.parent IS NOT NULL AND "
        "(p.id IS NULL OR p.book != c.book OR p.level != c.level - 1))",
    )
    total = _count(conn, "SELECT count(*) FROM index_term")
    reached = _count(
        conn,
        "WITH RECURSIVE tree(id) AS (SELECT id FROM index_term WHERE parent IS NULL "
        "UNION ALL SELECT c.id FROM index_term c JOIN tree ON c.parent = tree.id) "
        "SELECT count(*) FROM tree",
    )
    if reached != total:
        problems.append(f"{total - reached} terms not reached from a main term")
    return problems, f"{total:,} terms"


def check_pointers(conn: sqlite3.Connection) -> Findings:
    """Every code an Index term gives resolves to a Tabular node, apart from the known errors
    that index_exceptions.toml corrects or excludes."""
    problems = _rows(
        conn,
        "SELECT t.book || ':', t.path, '[' || p.field || ']', p.written FROM index_pointer p "
        "JOIN index_term t ON t.id = p.term WHERE p.status = 'unresolved' OR (p.target IS NOT "
        "NULL AND p.target NOT IN (SELECT key FROM tabular_node))",
    )
    counts = dict(conn.execute("SELECT status, count(*) FROM index_pointer GROUP BY status"))
    summary = ", ".join(f"{status} {counts.get(status, 0):,}" for status in ("ok", "corrected"))
    summary += f", excluded {counts.get('excluded', 0)}"
    return problems, summary


def check_xrefs(conn: sqlite3.Connection) -> Findings:
    """Every `see` and `see also` resolves, is an instruction, or is a listed Index error."""
    problems = _rows(
        conn,
        "SELECT t.book || ':', t.path, '[' || x.kind || ']', x.written FROM index_xref x "
        "JOIN index_term t ON t.id = x.term WHERE x.status = 'unresolved' AND x.reason IS NULL",
    )
    counts = dict(conn.execute("SELECT status, count(*) FROM index_xref GROUP BY status"))
    total = sum(counts.values())
    unresolved = counts.get("unresolved", 0)
    if total and unresolved / total > MAX_UNRESOLVED_XREFS:
        problems.append(f"{unresolved} unresolved of {total}, over {MAX_UNRESOLVED_XREFS:.0%}")
    summary = (
        f"{total:,}; resolved {counts.get('resolved', 0):,}, partial {counts.get('partial', 0):,}, "
        f"instruction {counts.get('instruction', 0):,}, unresolved {unresolved} (all listed)"
        if not problems
        else f"{total:,}"
    )
    return problems, summary


def check_edges(conn: sqlite3.Connection) -> Findings:
    """Every code or range named in a note lies in the release. A miss is an error in the notes:
    the edge is kept as unresolved, and the check warns."""
    problems = _rows(
        conn,
        "SELECT source, kind || ':', written FROM tabular_edge WHERE status = 'unresolved' "
        "ORDER BY note, ordinal",
    )
    total = _count(conn, "SELECT count(*) FROM tabular_edge")
    return problems, f"{total:,}; unresolved {len(problems)}"


def check_reachability(conn: sqlite3.Connection) -> Findings:
    """The billable codes an Index term leads to: the code itself, or a code above it from which
    the coder continues in the Tabular List. A code no term leads to is a gap in the Index, which
    a coder fills by browsing the Tabular List; the check warns and lists them."""
    unreached = _rows(
        conn,
        "WITH RECURSIVE below(key) AS (SELECT DISTINCT target FROM index_pointer "
        "WHERE target IS NOT NULL UNION SELECT c.key FROM tabular_node c "
        "JOIN below ON c.parent = below.key) "
        "SELECT key, description FROM tabular_node WHERE billable = 1 "
        "AND key NOT IN (SELECT key FROM below) ORDER BY ordinal",
    )
    # Given directly: the code itself, or for a code generated from a 7th character, its leaf.
    direct = _count(
        conn,
        "SELECT count(*) FROM tabular_node t WHERE t.billable = 1 AND EXISTS (SELECT 1 FROM "
        "index_pointer p WHERE p.target = t.key OR (t.kind = 'generated' AND p.target = t.parent))",
    )
    billable = _count(conn, "SELECT count(*) FROM tabular_node WHERE billable = 1")
    reached = billable - len(unreached)
    summary = (
        f"{reached / billable:.2%} of billable codes, {direct / billable:.1%} given directly; "
        f"{len(unreached)} not reached"
        if billable
        else "no billable codes"
    )
    return unreached, summary


def check_search(conn: sqlite3.Connection) -> Findings:
    """The search tables hold one row per Index term and per Tabular text."""
    problems = []
    for fts, table in (("fts_index_term", "index_term"), ("fts_tabular", "tabular_term")):
        indexed = _count(conn, f"SELECT count(*) FROM {fts}")
        rows = _count(conn, f"SELECT count(*) FROM {table}")
        if indexed != rows:
            problems.append(f"{fts} has {indexed:,} rows, {table} {rows:,}")
    tokenizer = conn.execute("SELECT value FROM meta WHERE key = 'fts_tokenizer'").fetchone()
    summary = f"tokenizer {tokenizer[0] if tokenizer else 'unknown'}"
    return problems, summary


CHECKS = (
    Check("content hash", check_content_hash),
    Check("tabular tree", check_tabular_tree),
    Check("billable codes", check_billable),
    Check("index tree", check_index_tree),
    Check("index pointers", check_pointers),
    Check("cross-references", check_xrefs),
    Check("note edges", check_edges, warn=True),
    Check("search tables", check_search),
    Check("reachable from the Index", check_reachability, warn=True),
)
