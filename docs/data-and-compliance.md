# Data and compliance

text2icd uses two kinds of data with very different rules. Code and contributors must keep them apart.

| | CMS ICD-10-CM files | MIMIC-IV and MIMIC-IV-Note |
| --- | --- | --- |
| What | Tabular List, Alphabetic Index, Neoplasm and Drug tables, code order files, addenda, guidelines | De-identified hospital records: discharge summaries and the ICD codes billed per admission |
| License | US government work, public domain | [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/) (see "Files" on the project page) |
| Used for | Building the knowledge graph; committed test fixtures; public synthetic benchmark | Local evaluation only |
| May be committed | Small excerpts in `tests/fixtures/` | **Never** |
| May be redistributed | Yes, including the built graph (GitHub Release asset) | **No**, in any form |

## MIMIC-IV rules

These follow from the data use agreement. When in doubt, do not share, and ask PhysioNet.

1. **Never commit MIMIC-IV data or anything derived per note or per patient.** That includes raw rows,
   discharge-diagnosis phrases, silver or gold evaluation sets built from notes, abbreviation lists
   mined from notes, result caches of MIMIC phrases, and notebooks with outputs.
2. **Never paste MIMIC-IV text** into issues, pull requests, commit messages, documentation,
   screenshots or general-purpose chat assistants.
3. **Only aggregate results may be published**, for example "accuracy@1 on the MIMIC test split".
   Do not publish individual examples, even in error analysis.
4. **Keep restricted files in git-ignored locations**: `data/` and `notebooks/local/`. Result caches go
   to `.text2icd/`, which is also git-ignored. The L3 decision cache there
   (`decisions.sqlite`) holds only hashes of the decisions and the option ids chosen, never a
   phrase. Message Batches (`-O llm=batch`) keep a batch's results on the provider for up to 29
   days; text2icd deletes each batch as soon as its results are read.
5. **LLM calls on MIMIC-IV text must go only to an approved endpoint** (next section).
6. Each person who runs MIMIC-IV experiments needs their own PhysioNet credentialing and signed DUA.

### Guardrails in this repository

- `.gitignore` excludes `data/**` (except READMEs and `.gitkeep`), `notebooks/local/`, `.text2icd/`,
  `.env` and all CSV files outside `tests/fixtures/`.
- The pre-commit hook `forbid-restricted-data` rejects any staged file under `data/` or
  `notebooks/local/`, and `check-added-large-files` rejects files over 500 KB.
- The CI job `data-guard` fails if a restricted path is tracked, in case hooks were skipped.
- `nbstripout` removes notebook outputs before commit.
- Tests that need MIMIC-IV are marked `mimic` and skipped when the files are absent, so they never
  run in CI.

### If restricted data was committed by mistake

1. If it has not been pushed, remove it from the commit (`git reset` or `git commit --amend`) and
   check `git log -p` for other occurrences.
2. If it was pushed, treat it as an incident: make the repository private if possible, rewrite
   history to remove the data (for example with `git filter-repo`), and contact PhysioNet.

## LLM endpoints and MIMIC-IV

PhysioNet's guidance
["Responsible use of MIMIC data with online services like GPT"](https://physionet.org/news/post/gpt-responsible-use)
(published 2023-04-18) says the DUA prohibits sharing the data with third parties, including through
APIs such as OpenAI's. It then names a short list of hosted services it considers acceptable, each
with conditions, such as opting out of human review or a guarantee that prompts and responses are
not used for training. Check the post for the current list and conditions.

text2icd therefore:

- defaults to **no LLM** (`TEXT2ICD_LLM_PROVIDER=none`);
- reaches LLMs only through a provider interface, so each deployment can use a service on that list;
- enforces `TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT=true` for MIMIC-IV runs, which allows only the
  providers in an explicit allow-list kept in code and reviewed when PhysioNet updates its guidance.

The allow-list is `ALLOW_LIST` in `src/text2icd/llm/providers.py`; each entry quotes its basis in
the PhysioNet guidance and the date it was checked. It holds the providers that text2icd implements
and that the guidance names (checked 2026-11-04):

- `anthropic`: the guidance says Claude does not train on prompts or responses by default and that
  routine human review is not performed;
- `bedrock`: the guidance names Amazon Bedrock and says data is not shared back to the base model
  for training. Claude in Amazon Bedrock runs on AWS-managed infrastructure under Amazon Bedrock's
  data protection terms.

`vertex` (Claude on Google Cloud Vertex AI) is implemented but **not** on the list: the guidance
names Vertex AI for Google's Gemini models only. Check Google Cloud's data governance terms and your
own data use agreement before adding it. `text2icd eval run` refuses an LLM system on a MIMIC-IV
dataset when the flag is off or the provider is not on the list.

Re-read the PhysioNet page before running MIMIC-IV experiments with an LLM. It may have changed since
this document was written.

## CMS files

- Source: <https://www.cms.gov/medicare/coding-billing/icd-10-codes>.
- Each fiscal year (October 1 to September 30) has its own release, and some years have an April 1
  update. Releases are named `fy2026`, `fy2026-apr`, `fy2027`, and so on.
- Files are downloaded by `text2icd data fetch` and checked against the SHA-256 hashes recorded in
  `src/text2icd/resources/releases.toml`. Raw files are never edited; all processing writes to `data/interim/` or
  `data/processed/`.
