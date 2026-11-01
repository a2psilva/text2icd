-- text2icd knowledge graph, one SQLite file per ICD-10-CM release.
-- Bump SCHEMA_VERSION in text2icd/graph/build.py whenever this file changes.
--
-- Codes are stored dotted (E11.22). Index term ids are global across the four Index books, in the
-- order index, eindex, neoplasm, drug. `ordinal` columns keep the order of the source files.

-- Release id, schema and builder versions, source file hashes, content hash.
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

-- The CMS code order file as published: every code with its billable flag and descriptions.
CREATE TABLE order_code (
    code              TEXT PRIMARY KEY,
    ordinal           INTEGER NOT NULL UNIQUE,
    billable          INTEGER NOT NULL CHECK (billable IN (0, 1)),
    short_description TEXT NOT NULL,
    long_description  TEXT NOT NULL
) WITHOUT ROWID;

-- Tabular List nodes: chapters (key '4'), sections (key 'E08-E13'), codes from the XML (kind
-- 'diag') and the codes generated from 7th-character definitions (kind 'generated', for example
-- T51.3X1A, whose parent is the leaf T51.3X1).
CREATE TABLE tabular_node (
    key          TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('chapter', 'section', 'diag', 'generated')),
    parent       TEXT REFERENCES tabular_node (key),
    ordinal      INTEGER NOT NULL UNIQUE,
    depth        INTEGER NOT NULL,
    description  TEXT NOT NULL,
    billable     INTEGER NOT NULL CHECK (billable IN (0, 1)),
    -- XML placeholder="true", or for a generated code: X padding was added before the 7th character.
    placeholder  INTEGER NOT NULL CHECK (placeholder IN (0, 1)),
    -- The node whose 7th-character definition applies (its own or the nearest ancestor's).
    seventh_from TEXT REFERENCES tabular_node (key),
    -- For generated codes: the 7th character that was added.
    seventh      TEXT
) WITHOUT ROWID;
CREATE INDEX tabular_node_parent ON tabular_node (parent, ordinal);

-- Instructional notes in document order. `kind` uses the XML element names: inclusionTerm,
-- includes, excludes1, excludes2, codeFirst, useAdditionalCode, codeAlso, notes, sevenChrNote,
-- visualImpairment.
CREATE TABLE tabular_note (
    id      INTEGER PRIMARY KEY,
    node    TEXT NOT NULL REFERENCES tabular_node (key),
    kind    TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text    TEXT NOT NULL,
    UNIQUE (node, ordinal)
);
CREATE INDEX tabular_note_kind ON tabular_note (kind);

-- 7th-character definitions (sevenChrDef), on the node that declares them.
CREATE TABLE seventh_char (
    owner   TEXT NOT NULL REFERENCES tabular_node (key),
    char    TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    meaning TEXT NOT NULL,
    PRIMARY KEY (owner, char)
) WITHOUT ROWID;

-- Alphabetic Index terms of the four books. `title` holds the essential words, `nonessential` the
-- modifiers in parentheses as a JSON array, and `path` the essential titles from the main term
-- down, joined with ' > '.
CREATE TABLE index_term (
    id            INTEGER PRIMARY KEY,
    book          TEXT NOT NULL CHECK (book IN ('index', 'eindex', 'neoplasm', 'drug')),
    parent        INTEGER REFERENCES index_term (id),
    ordinal       INTEGER NOT NULL,
    level         INTEGER NOT NULL,
    letter        TEXT NOT NULL,
    title         TEXT NOT NULL,
    printed_title TEXT NOT NULL,
    nonessential  TEXT NOT NULL,
    path          TEXT NOT NULL
);
CREATE INDEX index_term_parent ON index_term (parent, ordinal);
CREATE INDEX index_term_book_level ON index_term (book, level);

-- Column headings of the Neoplasm Table and the Table of Drugs and Chemicals.
CREATE TABLE table_column (
    book    TEXT NOT NULL CHECK (book IN ('neoplasm', 'drug')),
    col     INTEGER NOT NULL,
    heading TEXT NOT NULL,
    PRIMARY KEY (book, col)
) WITHOUT ROWID;

-- Code references found in note text, one row per reference in the order written. These are the
-- excludes1, excludes2, codeFirst, useAdditionalCode and codeAlso edges (other note kinds give
-- references too). `range_start` and `range_end` are dotted code prefixes, equal for a single code;
-- `target` is the Tabular node a single reference names. status: ok (at least one code of the
-- release lies in the range) or unresolved (no code inside: an error in the notes).
CREATE TABLE tabular_edge (
    note        INTEGER NOT NULL REFERENCES tabular_note (id),
    ordinal     INTEGER NOT NULL,
    source      TEXT NOT NULL REFERENCES tabular_node (key),
    kind        TEXT NOT NULL,
    written     TEXT NOT NULL,
    range_start TEXT NOT NULL,
    range_end   TEXT NOT NULL,
    target      TEXT REFERENCES tabular_node (key),
    status      TEXT NOT NULL CHECK (status IN ('ok', 'unresolved')),
    PRIMARY KEY (note, ordinal)
) WITHOUT ROWID;
CREATE INDEX tabular_edge_source ON tabular_edge (source, kind);
CREATE INDEX tabular_edge_target ON tabular_edge (target);

-- Codes the Index gives: <code>, <manif>, <subcat>, <seecat> and table cells (`col` is the table
-- column of a cell, 0 otherwise). `code` is the dotted code without a trailing '-', `partial` says
-- the Index wrote it as incomplete, and `target` is the Tabular node it resolves to.
-- status: ok; corrected (replaced by index_exceptions.toml); excluded (a known data error with no
-- correction, kept out of the graph); unresolved (unknown, rejected by the integrity checks).
CREATE TABLE index_pointer (
    term    INTEGER NOT NULL REFERENCES index_term (id),
    field   TEXT NOT NULL
            CHECK (field IN ('code', 'manifestation', 'subcategory', 'see_category', 'cell')),
    col     INTEGER NOT NULL,
    written TEXT NOT NULL,
    code    TEXT,
    partial INTEGER NOT NULL CHECK (partial IN (0, 1)),
    target  TEXT REFERENCES tabular_node (key),
    status  TEXT NOT NULL CHECK (status IN ('ok', 'corrected', 'excluded', 'unresolved')),
    PRIMARY KEY (term, field, col),
    CHECK ((status IN ('ok', 'corrected')) = (target IS NOT NULL))
) WITHOUT ROWID;
CREATE INDEX index_pointer_target ON index_pointer (target);
