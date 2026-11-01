# ADR 0001: SQLite with FTS5 as the knowledge-graph store

- Status: Accepted
- Date: 2026-11-01

## Context

The resolver navigates a graph built from the CMS ICD-10-CM files: Tabular List codes (46,881
`<diag>` elements in the FY2026 April update, plus the codes generated from 7th-character
definitions), Alphabetic Index terms with their subterms and cross-references (63,142 code
references, 12,141 "see" and 2,063 "see also" entries), and the Neoplasm and Drug tables. It also needs ranked full-text search over
Index paths, Tabular descriptions and inclusion terms to find main-term candidates.

Constraints:

- **No infrastructure.** The tool must run as a library, a CLI or a small container with no database
  server, so that it stays cheap to operate.
- **One artifact per release.** Each fiscal-year release is built once and shipped as a file. CMS
  content is public domain, so the built artifact can be published.
- **Fast start-up.** Loading the graph should take under a second, so the CLI is usable for one-off
  queries.
- **Deterministic builds.** The same input files must give the same graph content.

## Decision

Store each release's graph in **one SQLite file** (`text2icd-graph-<release>.sqlite`) with:

- ordinary tables for nodes, notes, edges, cross-references and table cells, defined by versioned SQL;
- **FTS5** virtual tables for ranked (BM25) search over Index paths, titles with nonessential
  modifiers, Tabular descriptions and inclusion terms;
- a `meta` table holding the release id, source-file hashes, builder version and a content hash over
  the sorted rows.

At run time the tables are loaded into plain Python dicts and tuples, so traversal never touches the
disk. Searches go to FTS5.

## Alternatives considered

- **networkx graph, pickled.** Convenient graph algorithms, but high memory use per node, no
  full-text search, and pickles are tied to library versions and unsafe to load from untrusted
  sources. The resolver needs only simple traversal (children, parents, cross-references), not
  general graph algorithms.
- **Graph database (for example Neo4j).** Expressive queries, but it is a server to run, secure and
  pay for, which conflicts with the zero-infrastructure goal.
- **JSON or Parquet files.** Easy to inspect, but they need a separate search index and slower
  parsing at start-up.
- **DuckDB.** Excellent for analytics, but its full-text search is an extension, and SQLite ships with
  Python's standard library.

## Consequences

- No runtime dependency beyond the standard library's `sqlite3`; BM25 ranking comes for free.
- The graph can be attached to a GitHub Release and downloaded instead of built.
- Traversal helpers are written by hand. This is acceptable because the access patterns are few.
- FTS5 must be compiled into the interpreter's SQLite. It is available on CPython 3.11 to 3.13 as
  tested; `tests/test_environment.py` checks it on every CI platform.
- Determinism is checked by the content hash, not by comparing database files byte for byte, since
  SQLite page layout can vary between builds.
