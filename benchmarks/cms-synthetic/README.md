# cms-synthetic

Diagnosis phrases generated from the CMS ICD-10-CM files, each with its exact gold code set. The
CMS files are US government works in the public domain, so the set can be shared and reproduced by
anyone. It contains no patient data.

`fy2027-sample.jsonl` is a stratified sample of 1,067 examples from the full FY2027 set (142,757
examples), used for regression tests in CI. The format is described in
[docs/evaluation.md](../../docs/evaluation.md#file-format); `fy2027-sample.manifest.json` records
the seed, the graph it was generated from and the hash of the full set.

## How the phrases are made

| Generator | Source | Example |
| --- | --- | --- |
| `index` | An Alphabetic Index path, read as a phrase, with the code it gives and its manifestation code | `Pneumonia > aspiration > due to > food` -> "aspiration pneumonia due to food" (J69.0) |
| `inclusion` | A Tabular inclusion term of a billable code, without its parenthesised words | "Granular conjunctivitis (trachomatous)" -> "granular conjunctivitis" (A71.1) |
| `neoplasm` | A Neoplasm Table row and one of its columns | "benign neoplasm of abdomen" |
| `drug` | A Table of Drugs and Chemicals row and one of its columns | "adverse effect of digoxin, initial encounter" |

A code that needs a 7th character gets one at random, and its meaning is added to the phrase, so the
label stays exact. Each phrase appears as generated (`base`) and in one perturbed form:

| Perturbation | Change |
| --- | --- |
| `nonessential` | One of the term's nonessential modifiers added; the code must not change |
| `word-order` | The Index order ("pneumonia, aspiration, due to food") or a reordered inclusion term |
| `abbreviation` | A common clinical abbreviation (CHF, T2DM, w/) |
| `typo` | One edit in one word of five letters or more |

Splits are by ICD-10-CM category, so all phrases of a category are in the same split. Phrases that
two sources label differently are dropped.

The phrases follow the classification's own wording, so this set measures regressions and gives a
reproducible public benchmark. It does not measure accuracy on clinical text.

## Regenerate

```bash
uv run text2icd data fetch -r fy2027
uv run text2icd build-graph -r fy2027
uv run text2icd eval synth -r fy2027 --sample-output benchmarks/cms-synthetic/fy2027-sample.jsonl
```

The full set is written to `data/processed/eval/cms-synthetic-fy2027.jsonl` (git-ignored, about
40 MB). Generation is deterministic for a given release and seed; a slow test checks that the
committed sample matches the generator.
