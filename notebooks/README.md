# notebooks/

Exploration notebooks.

- `notebooks/local/` is git-ignored and is the only place for notebooks that read MIMIC-IV. Their
  outputs contain restricted text, so they must never be committed.
- Notebooks committed at this level may use CMS data only. The `nbstripout` pre-commit hook removes
  outputs before commit.
- Code that proves useful in a notebook moves into `src/text2icd/` with tests.

Notebooks read data from `data/raw/` (see [data/README.md](../data/README.md)), for example
`data/raw/mimic-iv/diagnoses_icd.csv`.
