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
uv sync --all-extras        # Python 3.12 virtual environment, dev tools and local embeddings
uv run pre-commit install   # run the hooks on every commit
make check                  # what CI runs
```

Download the CMS release files with `uv run text2icd data fetch` (see [data/README.md](data/README.md)).

## Workflow

1. Open an issue first for anything larger than a small fix. Changes to a recorded design decision
   (storage, the cascade, LLM providers, release handling) also need a new record in
   [docs/adr/](docs/adr/README.md).
2. Branch from `main` (`feat/...`, `fix/...`, `docs/...`).
3. Keep changes focused. Add or update tests with the change.
4. Run `make check`. CI runs the same steps on Python 3.11, 3.12 and 3.13.
5. Add a line under `Unreleased` in `CHANGELOG.md`.
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
- Mark long-running tests `@pytest.mark.slow`. Tests on the real CMS files are slow tests that skip
  when the release is not fetched; the CI integration job fetches FY2026 and FY2027 and runs them.
- Mark tests of a time or memory budget `@pytest.mark.benchmark` as well. CI skips them, because
  shared runners vary too much; run them locally with `uv run pytest -m benchmark`.
