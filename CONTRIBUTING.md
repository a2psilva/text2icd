# Contributing to text2icd

Thanks for helping. This guide covers setup, the workflow, and the data rules, which are not optional.

## Data rules (read first)

- **Never commit, paste or upload MIMIC-IV data**: no notes, no rows, no phrases copied from a note,
  no screenshots of them, no per-note derived files. This applies to code, tests, fixtures, issues,
  pull requests, commit messages and LLM chats. The PhysioNet data use agreement forbids it.
- Tests use **CMS excerpts** (public domain) and **hand-written synthetic phrases** only.
- Only **aggregate** metrics computed on MIMIC-IV may be shared, for example accuracy on a test split.
- Details: [docs/data-and-compliance.md](docs/data-and-compliance.md).

The pre-commit hook `forbid-restricted-data` and the CI job `data-guard` block files under `data/` and
`notebooks/local/`. Do not bypass them.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # Python 3.12 virtual environment with the dev tools
uv run pre-commit install   # run the hooks on every commit
make check                  # what CI runs
```

Place the CMS FY2026 release files in `data/raw/cms/fy2026/` (see [data/README.md](data/README.md)).

## Workflow

1. Pick an unchecked item in [TASKS.md](TASKS.md), or open an issue first for anything larger.
2. Branch from `main` (`feat/...`, `fix/...`, `docs/...`).
3. Keep changes focused. Add or update tests with the change.
4. Run `make check`. CI runs the same steps on Python 3.11, 3.12 and 3.13.
5. Tick the task in `TASKS.md` and add a line under `Unreleased` in `CHANGELOG.md`.
6. Open a pull request and complete its checklist.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

## Code style

- Ruff formats and lints (`make format`). Line length is 100.
- mypy runs in strict mode on `src/`. Public functions are fully typed.
- Prefer small pure functions and plain data classes. The graph and parsers must stay deterministic:
  the same input files must produce the same graph content hash.
- Every coding decision the resolver makes must appear in the result trace. If a rule comes from the
  ICD-10-CM Official Guidelines, cite its section (for example `I.A.15`) in the code that applies it.

## Tests

- Tests live under `tests/`, mirroring the layout of `src/text2icd/`.
- Tests that need local MIMIC-IV files are marked `@pytest.mark.mimic`. They are skipped automatically
  when the files are absent and never run in CI.
- Mark long-running tests `@pytest.mark.slow`.
