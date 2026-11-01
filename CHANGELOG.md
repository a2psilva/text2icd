# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: `src/` package layout, Typer CLI with a `version` command, pytest smoke tests.
- Tooling: uv, ruff, mypy (strict), pre-commit hooks, GitHub Actions CI on Python 3.11 to 3.13.
- Data-safety guardrails: git-ignored `data/` tree, a pre-commit hook and a CI job that block
  restricted files from being committed.
- Documentation: architecture, coder workflow, data and compliance, evaluation protocol.
- Architecture decision records 0001 to 0004 in `docs/adr/`: SQLite + FTS5 graph store, the
  deterministic cascade with a constrained LLM navigator, the LLM provider and MIMIC-IV endpoint
  policy, and release naming.
- `data/README.md` records the local MIMIC-IV (v3.x) and MIMIC-IV-Note (v2.2) versions and how the
  local `diagnoses_icd10.csv` extract was made.
- Test that the interpreter's SQLite includes FTS5 with BM25 ranking.
- `text2icd.config.Settings`: every `TEXT2ICD_*` setting (data and cache directories, release id,
  LLM provider and limits), read from the environment or a `.env` file. `.env.example` now uses
  `TEXT2ICD_RELEASE` instead of a fiscal-year number.
- Release registry (`src/text2icd/resources/releases.toml`) listing the CMS files and SHA-256 hashes
  of FY2027, the FY2026 April update, FY2026, and the FY2016 to FY2025 code lists.
- `text2icd data fetch` downloads, verifies and unpacks release files (retries, hash and size
  checks, safe zip extraction, no overwrites); `text2icd data releases` shows what is present.
- `text2icd.ontology.codes`: add and remove the dot, validate code format, append 7th characters
  with `X` placeholders, and parse code references and ranges such as `N18.1-N18.6` or `E10.-`.
- `text2icd.ontology.tabular`: parser for the Tabular List XML (chapters, sections, codes, all
  instructional notes, 7th-character definitions with inheritance, placeholders); and
  `text2icd.ontology.files` to locate the files of a fetched release.
- `text2icd.ontology.index`: parser for the Alphabetic Index, the External Cause Index, the
  Neoplasm Table and the Table of Drugs and Chemicals. Titles are split into essential words and
  nonessential modifiers (including nested and mixed cases), with see, see also, subcat, seecat,
  manifestation codes and table cells.
- `text2icd.ontology.order`: strict parser for the fixed-width code order files, giving the billable
  code set of every release from FY2016 (69,823 codes) to FY2027 (74,879).
- `text2icd.ontology.billable`: generates the billable code set from the Tabular List (leaf codes and
  7th-character expansions). It equals the order file exactly for FY2026, the FY2026 April update
  and FY2027, including the S06 exception stated in the Tabular notes.
- `text2icd.ontology.validate` and `resources/index_exceptions.toml`: every code the Index points to
  must exist in the Tabular List; the few CMS data errors are listed with reasons and, where clear,
  the intended code.
- `text2icd data stats`: parses a fetched release, prints its counts and runs both checks; it exits
  non-zero if a check fails.
- `text2icd build-graph`: builds the knowledge-graph SQLite file of a release
  (`data/processed/text2icd-graph-<release>.sqlite`) with the code order file, every Tabular node
  including the codes generated from 7th-character definitions, notes, 7th-character definitions,
  and every Index term with its full path. The build is atomic and deterministic: a content hash
  over the sorted rows is stored in `meta` with the hash of each source file.
- `text2icd.ontology.billable.iter_billable` tells which leaf, 7th character and definition each
  billable code comes from; `Tabular.seventh_character_owner` finds the definition that applies.
- Index-to-Tabular pointers (`index_pointer`): every code an Index term gives, including partial
  codes such as `C79.8-` and table cells, resolved to its Tabular node. The corrections in
  `index_exceptions.toml` are applied (the four "pubic ramus" entries); the other listed exceptions
  are kept out with a warning in the build log. FY2026 and FY2027 have no unresolved pointer.
- `IndexBook.paths()` returns the joined path of every term in one pass.
- `text2icd.graph.model.load(release) -> Graph`: the built graph in memory, with accessors for codes
  (`code`, `parent`, `children`, `ancestors`, `seven_char_def`, `notes` and `edges`, optionally
  inherited from ancestors), Index terms (`term_at`, `subterms`, `pointers`, `xrefs`), reverse
  lookups and search. FY2027 loads in about 0.85 s within 255 MB.
- `text2icd graph show "<Index path>"` and `text2icd graph code <code>` print a term or a code with
  its notes, edges, cross-references and the Index terms that give it.
- Full-text search in the graph: FTS5 tables over every Index term (title, modifiers and path) and
  over Tabular descriptions and inclusion terms (`tabular_term`), with `graph.search.search_index`
  and `search_tabular` (BM25, free-text queries made safe for FTS5). `build-graph --tokenizer`
  chooses `porter` (default) or `unicode61`; the choice is recorded in `meta`.
- Index cross-references (`index_xref`): every `see` and `see also` resolved to the Index term it
  names, by longest match on titles that contain commas, with table columns taken from the column
  headings (`Neoplasm, benign, by site`), books, `by site` and code instructions kept as such, and
  the words a partial match leaves over. About 88% resolve fully and 0.1% do not; those 16 are
  Index errors listed with a reason in `resources/xref_exceptions.toml`, and the build warns about
  any other.
- Tabular note edges (`tabular_edge`): every code or range named in an instructional note
  (`(N18.1-N18.6)`, `(E10.-)`, `category T51`), with the stray dots of the CMS files handled. About
  11,600 per release; the build log gives the share of notes with a reference by kind, the words
  that look like codes (`B12`, `GM2`) and any reference with no code in the release.

### Fixed

- Code validation accepts a letter in the second position, as in the `QA0`-`QA1` genetic disorder
  codes added in FY2026.
- Neoplasm Table cells marked `-` are read as not applicable, like `--` in the Table of Drugs.
- Code validation requires a digit in the second or third character, so words such as `HIV` or
  `SCFE` are no longer taken for codes.
