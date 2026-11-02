# Evaluation

The evaluation harness is built before the resolver, so that every change is measured against
fixed data. This page defines the datasets, metrics, baselines and protocol.

## Datasets

| Dataset | Built from | Size (target) | Labels | Committed? |
| --- | --- | --- | --- | --- |
| `cms-synthetic` | CMS inclusion terms and Index paths turned into phrases, with perturbations | 5,000+ | Exact by construction | Yes (public domain) |
| `mimic-silver` | "Discharge Diagnosis" sections of MIMIC-IV-Note, aligned to the admission's ICD-10 codes | 10,000+ phrases | Automatic, precision estimated by manual review | No |
| `mimic-gold` | Stratified sample of `mimic-silver` | 300–500 phrases | Hand-annotated, ideally reviewed by a certified coder | No |

### File format

Every dataset is a JSONL file, one example per line, read and checked by
`text2icd.eval.datasets`:

```json
{"id": "cms-synthetic:000042", "phrase": "aspiration pneumonia due to food",
 "gold": [{"code": "J69.0", "role": "primary"}], "release": "fy2027",
 "source": "cms-synthetic", "split": "dev", "meta": {"generator": "index-path"}}
```

| Field | Content |
| --- | --- |
| `id` | Unique within the dataset |
| `phrase` | The diagnosis phrase, on one line |
| `gold` | The full code set: one `primary` code first, then any `companion` codes. Each code has a `status` against `release` (`valid` by default, `mapped` with `mapped_to`, or `unmappable`; see [Code-set versions](#code-set-versions)) |
| `release` | The release the gold codes are checked against; one per dataset |
| `source` | Where the example came from, such as `cms-synthetic` or `mimic-silver` |
| `split` | `dev` or `test` |
| `meta` | Anything else: how the example was made, local identifiers for MIMIC-IV sets |

`text2icd eval validate FILE...` checks the format, repeated ids and mixed releases, checks each gold
code's status against the billable codes of the release (when it is fetched), and prints the dataset
hash. The hash is a SHA-256 over the examples in id order, so it identifies the content whatever the
line order or formatting; every result file records it.

### `cms-synthetic`

- Positive examples: each Tabular inclusion term with its code; each Index path read as a phrase
  ("Pneumonia > aspiration > due to > food" becomes "aspiration pneumonia due to food") with its code
  after Tabular verification.
- Perturbations: word-order changes, common abbreviations ("DM", "CKD", "HTN"), typos, dropped
  nonessential modifiers.
- Purpose: CI regression tests and a public benchmark anyone can reproduce. It is not used to report
  real-world accuracy, because its phrases follow the classification's own wording.

### `mimic-silver`

1. Extract the "Discharge Diagnosis" section from each discharge summary and split it into items
   (bullets, lines, "Primary:" and "Secondary:" headers removed).
2. Join the admission's ICD-10 codes from `diagnoses_icd` (`icd_version = 10`; codes are stored without
   the dot).
3. Align each item to at most two of that admission's codes with an aligner that does **not** use the
   text2icd resolver (for example embedding similarity between the item and each code's description),
   keeping only alignments with a clear margin over the runner-up. Using the resolver would bias the
   labels toward its own mistakes.
4. Review 200 random alignments by hand and report the estimated label precision with the results.

### `mimic-gold`

Stratified by ICD-10-CM chapter and by code frequency (head, torso, tail), so rare codes are
represented. Each phrase gets its full code set, including companion codes.

### Splits

`dev` and `test` are split by `subject_id`, so no patient appears in both. `test` is frozen once
created and is used only for reported numbers, never for tuning. The calibration model for confidence
is fitted on `dev`.

### Code-set versions

MIMIC-IV codes were assigned under the release in force at each admission, roughly FY2016 to FY2022.
Some of those codes were later deleted or split (for example N18.3 became N18.30, N18.31 and N18.32 in
FY2021). `text2icd.eval.versions` checks each gold code against the evaluated release, using the
order files of every fetched release from FY2016 and the CMS conversion tables:

- `valid`: billable in the release, scored by exact match;
- `mapped`: billable in an earlier release and replaced since, scored by **mapped match** (a
  prediction counts if it is one of the codes in `mapped_to`) and reported separately. The
  replacements come from the conversion tables, read backwards (old code to the new codes that list
  it as their previous assignment) and followed across releases until codes billable in the target
  remain. Only references that name codes exactly (`N18.3`, `H35.20-H35.23`) are read backwards;
  broad ones (`I63.-`, "Codes in Categories T15-T19") name codes that still exist. A code that
  became a header with no conversion row maps to its billable subdivisions;
- `unmappable`: never billable, or deleted with no replacement; left out of the scores and counted.

`text2icd eval reconcile N18.3 -r fy2027` shows one code; `--dataset IN -o OUT` rewrites the
statuses of a whole dataset against the release.

Against FY2027, the 3,455,747 ICD-10 code assignments in the local MIMIC-IV `diagnoses_icd` (19,440
distinct codes) are 96.2% valid, 3.7% mapped (406 codes through the conversion tables, 4 through
their subdivisions) and 83 assignments (38 distinct codes) unmappable.

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy@1 | Primary predicted code equals the gold primary code |
| Accuracy@5 | Gold primary code is among the primary code and the first four alternatives |
| Category accuracy | First three characters match |
| Code-set F1 | Precision and recall of the full predicted code set against the gold set, including companions |
| Invalid-code rate | Share of predicted codes not billable in the release. Must be 0 |
| Selective accuracy | Accuracy at each coverage level when phrases below a confidence threshold are abstained |
| Calibration | Expected calibration error of the reported confidence |
| LLM call rate | Share of phrases that reached L3, and mean calls per phrase |
| Cost | US dollars per 1,000 phrases, from the provider `usage` fields |
| Latency | p50 and p95 per phrase, with and without the LLM |

All metrics except latency are reported with 95% bootstrap confidence intervals (1,000 resamples
of the examples, fixed seed), overall, for the `valid` and `mapped` examples separately, and per
chapter. They are computed by `text2icd.eval.metrics.score`:

- A system returns, per example, a code set (primary code first), ranked alternates, an optional
  confidence, and its LLM calls, cost and latency (`text2icd.eval.predictions.Prediction`). An
  example with no prediction counts as wrong.
- Accuracy@5 reads the primary code then the alternates. Code-set precision, recall and F1 are
  pooled over all examples (micro), each predicted code matching at most one gold code.
- Examples with an `unmappable` gold code are counted and left out of the accuracy and code-set
  metrics; the invalid-code rate counts every predicted code, alternates included.
- Selective accuracy is accuracy@1 on the most confident 100%, 90%, ..., 50% of the examples; once
  a system reports any confidence, an example without one counts as 0. Calibration uses ten
  equal-width bins.

## Baselines

| Id | Method | Purpose |
| --- | --- | --- |
| B1 | BM25 over Tabular descriptions and inclusion terms | Lexical floor |
| B2 | Embedding nearest neighbour over the same text | Semantic floor at zero API cost |
| B3 | A small, low-cost LLM given the top 20 codes from B2 | The "cheap LLM plus retrieval" approach text2icd must beat |
| B4 | A large, high-end LLM given the top 20 codes from B2, on a small sample with a hard budget | Reference ceiling for an expensive model |

Success criterion for v1.0: text2icd beats B3 on accuracy@1 and code-set F1 on `mimic-gold`, at less
than one tenth of B3's cost per 1,000 phrases.

## Protocol

- Every benchmark run records the git commit, release id, settings, model ids and dataset hashes in
  its output file under `data/processed/eval/` (git-ignored).
- Ablations switch off one component at a time: no Index walk, no Tabular verification, no
  abbreviation lexicon, no L2, no L3.
- Error analysis groups failures into buckets (wrong main term, wrong subterm, wrong Tabular child,
  missing companion code, missing 7th character, label noise) and tracks bucket sizes across
  iterations. Only bucket counts are published, never examples from MIMIC-IV.
