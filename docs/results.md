# Results

Aggregate numbers for text2icd v0.1.0, the deterministic coder: the Index walk, Tabular
verification, the instructional notes and the guideline rule pack (Phases 1 to 6). The coder makes
no model or API call; it costs $0 per phrase. How each dataset is built and scored is in
[evaluation.md](evaluation.md). Only aggregate results are published: no example from MIMIC-IV
appears here or anywhere in the repository ([data-and-compliance.md](data-and-compliance.md)).

All runs use the FY2027 release, `text2icd eval run` with its defaults (inpatient setting, initial
encounter when the phrase states none), and 1,000 bootstrap resamples for the 95% intervals.
Latency is per phrase on one CPU core of an Apple laptop, after the graph is loaded.

## `cms-synthetic` test

The public set built from the CMS files themselves: Index paths turned into phrases, Tabular
inclusion terms, Neoplasm Table and Table of Drugs rows, with perturbations (typos,
abbreviations, word order). One gold code per phrase. The test split was not used for tuning.

| System | Acc@1 | Acc@5 | Category | Code-set F1 | Invalid | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 BM25 | 37.7% (37.4-38.1) | 50.6% | 49.4% | 40.4% (40.1-40.8) | 0% | 2.5 / 22.8 ms |
| B2 embedding | 39.4% (39.0-39.7) | 53.3% | 51.7% | 40.9% (40.6-41.3) | 0% | 0.5 / 0.5 ms |
| `coder` | 94.2% (94.1-94.4) | 95.2% | 97.7% | 93.9% (93.8-94.1) | 0% | 1.0 / 3.9 ms |

74,945 phrases. On the dev split the coder scores 95.5% Acc@1 and 95.3% F1. A code set the
guidelines require counts against a single gold code: companions the notes require lower
precision, and a code sequenced first by a rule lowers Acc@1 ("severe sepsis": A41.9, R65.20, where
the gold is R65.20).

By chapter, the coder is weakest on external causes (chapter 20, 18.8% of 277 phrases): it looks up
the External Cause Index only when a phrase states a mechanism of injury, and these phrases are
the Index's own wording of the event. Pregnancy (chapter 15, 89.8%) and injuries (chapter 19,
89.5%) follow; every other chapter is above 92%.

## `mimic-silver` dev

Diagnosis items from the "Discharge Diagnosis" sections of MIMIC-IV-Note, each aligned to one code
of its admission by an aligner that uses no part of text2icd. These are clinical phrases as
written, abbreviations and all. Two caveats:

- The labels are automatic. Their precision has not yet been estimated by the planned hand review
  of 200 alignments, so these numbers bound the coder's accuracy only roughly.
- This is the `dev` split. The `test` split is reported once the set is final and frozen.

88,590 phrases. Gold codes were assigned under the releases of FY2016 to FY2022: 4,697 of them
(5.3%) have been replaced since and are scored by mapped match, on which the coder scores 65.8%
Acc@1.

| System | Acc@1 | Acc@5 | Category | Code-set F1 | Invalid | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 BM25 | 36.7% (36.3-37.0) | 70.7% | 60.6% | 37.6% (37.3-38.0) | 0% | 0.2 / 3.7 ms |
| B2 embedding | 37.5% (37.2-37.8) | 74.0% | 70.7% | 37.5% (37.3-37.9) | 0% | 0.3 / 0.3 ms |
| `coder` | 73.1% (72.8-73.4) | 79.6% | 87.7% | 73.0% (72.7-73.3) | 0% | 0.5 / 3.5 ms |

## Confidence (after v0.1.0)

Each code set now carries a calibrated confidence that its primary code is right, and is marked
for escalation below a threshold ([architecture.md](architecture.md#confidence-and-escalation)).
The shipped table is fitted on `cms-synthetic` dev. Its threshold, 0.39, is the lowest confidence
at which the code sets kept reach 98% precision on dev. Accuracy and the code sets do not change.

| Dataset | Table | ECE | Acc@1 of the most confident 90% / 70% / 50% | Kept at the threshold |
| --- | --- | --- | --- | --- |
| `cms-synthetic` dev, 5 held-out folds | shipped | 0.001 | 99.6% / 99.8% / 99.9% | 96.2%, at 98.2% |
| `cms-synthetic` test | shipped | 0.003 (0.002-0.004) | 99.3% / 99.8% / 99.8% | 95.2%, at 97.6% |
| `mimic-silver` dev | shipped | 0.151 (0.148-0.153) | 77.5% / 82.0% / 84.2% | 90.1%, at 77.5% |
| `mimic-silver` dev, 5 held-out folds | local, fitted on `mimic-silver` dev | 0.041 | 76.9% / 83.5% / 85.0% | no threshold reaches 98% |

The held-out folds are grouped by gold code. On `cms-synthetic` dev the AUROC is 0.969.

- **Clinical phrases need their own table.** On `mimic-silver` the shipped table is overconfident,
  because it learned how often the coder is right on the CMS's own wording. A table fitted on the
  clinical phrases is calibrated (0.041), but it is local only: it is fitted on MIMIC-IV and
  never shipped. `TEXT2ICD_CALIBRATION` points the coder at it.
- **The ranking is weaker on clinical phrases** (AUROC 0.715). Part of what counts as wrong there
  is the silver label, which is not yet audited.
- **Phrases where a rule puts another code first** get a low confidence ("severe sepsis": A41.9
  before R65.20). Single-code gold rarely agrees with the guideline order.

## Phase 6 exit criteria

- **Invalid-code rate exactly 0 on every dataset.** Met: 0 invalid codes on `cms-synthetic` dev
  and test, on the committed `cms-synthetic-sample` (all splits) and on `mimic-silver` dev. The
  coder returns no code, flagged `needs_review`, rather than an invalid one.
- **On `mimic-gold` dev, accuracy@1 and code-set F1 beat B1 and B2.** Waiting for the hand
  annotation of `mimic-gold`. On `mimic-silver` dev the coder roughly doubles both baselines
  (73.1% Acc@1 against 36.7% and 37.5%; 73.0% F1 against 37.6% and 37.5%), and the intervals
  are far apart.
- **Numbers published here, aggregates only.** This page.

## Reproduce

```bash
uv run text2icd data fetch -r fy2027 && uv run text2icd build-graph -r fy2027
uv run text2icd eval synth -r fy2027                   # builds the public set
uv run text2icd eval run -d cms-synthetic -s coder --split test
uv run text2icd eval run -d cms-synthetic -s bm25 --split test
uv run text2icd eval run -d cms-synthetic -s embedding --split test
uv run text2icd eval mimic silver -r fy2027             # needs MIMIC-IV access (PhysioNet)
uv run text2icd eval run -d mimic-silver -s coder
uv run text2icd eval calibrate -d cms-synthetic -o src/text2icd/resources/calibration.json
uv run text2icd eval calibrate -d mimic-silver          # a local table, under data/
```

Each run writes its result file, with the commit, release, graph and dataset hashes, under
`data/processed/eval/results/`; `text2icd eval report` renders them as a table.
