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

Currently present in `raw/mimic-iv/`:

| File | Origin |
| --- | --- |
| `discharge.csv` | MIMIC-IV-Note, discharge summaries |
| `diagnoses_icd.csv` | MIMIC-IV hosp module, ICD-9 and ICD-10 diagnoses per admission |
| `diagnoses_icd10.csv` | Local subset of `diagnoses_icd.csv` (ICD-10 rows for 41,723 admissions). Document how it was made |
| `MIMIC-IV-notes.txt` | Notes on the discharge summary sections |

Record the exact MIMIC-IV and MIMIC-IV-Note versions you downloaded here, because evaluation results
depend on them.
