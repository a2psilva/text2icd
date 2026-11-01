# data/

Everything in this directory is git-ignored except README files and `.gitkeep` placeholders.
A pre-commit hook and a CI job enforce this. Read
[docs/data-and-compliance.md](../docs/data-and-compliance.md) before adding anything here.

## Layout

```
data/
├── raw/                    files exactly as downloaded, never edited
│   ├── cms/
│   │   └── fy2026/         one directory per ICD-10-CM release
│   └── mimic-iv/           PhysioNet credentialed data (local only)
├── interim/                intermediate outputs (parsed tables, alignments)
└── processed/              build outputs (knowledge graph SQLite, evaluation sets)
```

## CMS ICD-10-CM (public domain)

Source: <https://www.cms.gov/medicare/coding-billing/icd-10-codes>.

Currently present in `raw/cms/fy2026/` (from "2026 Code Tables, Tabular and Index"):

| File | Content |
| --- | --- |
| `icd10cm_tabular_2026.xml` + `icd10cm_tabular.xsd` | Tabular List: chapters, blocks, codes, instructional notes, 7th characters |
| `icd10cm_index_2026.xml` + `icd10cm_index.xsd` | Alphabetic Index to Diseases and Injuries |
| `icd10cm_eindex_2026.xml` | External Cause of Injuries Index |
| `icd10cm_neoplasm_2026.xml` + `icd10cm_drug_neoplasm.xsd` | Neoplasm Table |
| `icd10cm_drug_2026.xml` | Table of Drugs and Chemicals |
| `*.pdf` | PDF renderings of the same books, for human reference |

Still to download: the code descriptions in tabular order (billable flags), the
addendum and conversion table, the Official Coding Guidelines PDF, the April 1, 2026 update, the
FY2027 release, and the order files for FY2016 to FY2025 (for MIMIC-IV version reconciliation).
`text2icd data fetch` will automate this.

CMS publishes a mid-year update for some fiscal years (for example April 1, 2026). Releases are
therefore named `fy<YEAR>` for October and `fy<YEAR>-apr` for April.

## MIMIC-IV (PhysioNet credentialed, never committed)

Requires a PhysioNet credentialed account and a signed data use agreement for
[MIMIC-IV](https://physionet.org/content/mimiciv/) and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/).

### Versions

| Dataset | Version | How it was identified |
| --- | --- | --- |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | **v3.x** (3.0 or 3.1) | `diagnoses_icd` covers 545,497 admissions; v2.2 has only 431,231 admissions in total, v3.0 has 546,028 |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | **v2.2** | `discharge.csv` has 331,793 rows, matching other projects that count the v2.2 file (the PhysioNet page states 331,794) |

v3.0 and v3.1 differ in `diagnoses_icd` only by the removal of two patients who were missing from
`patients.csv`, so the exact MIMIC-IV release cannot be read from this table. Record it from the
download page if known. Evaluation results must state both versions.

MIMIC-IV-Note v2.2 was published alongside MIMIC-IV v2.2 (January 2023), so the admissions that
MIMIC-IV v3.0 added (2020 to 2022) are not expected to have a discharge summary.

### Files in `raw/mimic-iv/`

| File | Origin |
| --- | --- |
| `discharge.csv` | MIMIC-IV-Note v2.2, `note/discharge` (discharge summaries) |
| `diagnoses_icd.csv` | MIMIC-IV v3.x, `hosp/diagnoses_icd` (ICD-9 and ICD-10 codes per admission) |
| `diagnoses_icd10.csv` | Local working extract, **not an official file and not for evaluation** (see below) |
| `MIMIC-IV-notes.txt` | Notes on the discharge summary sections |

Aggregate facts about the local copy: 254,377 admissions have ICD-10 diagnosis codes, and 122,288 of
those also have a discharge summary. Those 122,288 admissions are the population the MIMIC-IV
evaluation sets are drawn from.

### How `diagnoses_icd10.csv` was made

It is exactly the ICD-10 rows among the **first 1,048,575 data rows** of `diagnoses_icd.csv`, sorted
by `icd_code`. That row count is the limit of an Excel worksheet (1,048,576 rows including the
header), so the file was most likely produced by opening `diagnoses_icd.csv` in Excel, filtering
`icd_version = 10` and sorting by code. Consequences:

- It covers only the patients at the start of the file (`subject_id` 10000084 to 11663007), which is
  16.5% of the ICD-10 rows (570,764 of 3,455,747) and 41,723 admissions. It is not a random sample.
- The last admission before the cut-off is incomplete: 2 of its ICD-10 codes are missing.
- Only 20,115 of its 41,723 admissions have a discharge summary.

Evaluation code reads `diagnoses_icd.csv` directly and never uses this file. To regenerate it for
exploration:

```python
import pandas as pd

head = pd.read_csv("data/raw/mimic-iv/diagnoses_icd.csv", nrows=1_048_575, dtype={"icd_code": str})
icd10 = head[head["icd_version"] == 10].sort_values("icd_code", kind="stable")
```
