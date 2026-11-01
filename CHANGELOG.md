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

### Fixed

- Code validation accepts a letter in the second position, as in the `QA0`-`QA1` genetic disorder
  codes added in FY2026.
