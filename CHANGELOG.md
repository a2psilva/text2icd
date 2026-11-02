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
- `text2icd.ontology.billable`: generates the billable code set from the Tabular List (leaf codes and
  7th-character expansions). It equals the order file exactly for FY2026, the FY2026 April update
  and FY2027, including the S06 exception stated in the Tabular notes.
- `text2icd.ontology.validate` and `resources/index_exceptions.toml`: every code the Index points to
  must exist in the Tabular List; the few CMS data errors are listed with reasons and, where clear,
  the intended code.
- `text2icd data stats`: parses a fetched release, prints its counts and runs both checks; it exits
  non-zero if a check fails.
- `text2icd build-graph`: builds the knowledge-graph SQLite file of a release
  (`data/processed/text2icd-graph-<release>.sqlite`) with the code order file, every Tabular node
  including the codes generated from 7th-character definitions, notes, 7th-character definitions,
  and every Index term with its full path. The build is atomic and deterministic: a content hash
  over the sorted rows is stored in `meta` with the hash of each source file.
- `text2icd.ontology.billable.iter_billable` tells which leaf, 7th character and definition each
  billable code comes from; `Tabular.seventh_character_owner` finds the definition that applies.
- Index-to-Tabular pointers (`index_pointer`): every code an Index term gives, including partial
  codes such as `C79.8-` and table cells, resolved to its Tabular node. The corrections in
  `index_exceptions.toml` are applied (the four "pubic ramus" entries); the other listed exceptions
  are kept out with a warning in the build log. FY2026 and FY2027 have no unresolved pointer.
- `IndexBook.paths()` returns the joined path of every term in one pass.
- `text2icd graph check`: integrity checks of a built graph (content hash, both trees, billable
  codes, Index pointers, cross-references, note edges, search tables, and the share of billable
  codes the Index leads to); it exits non-zero when a check fails.
- CI integration job (not required): fetches FY2026 and FY2027 with a cache, runs the slow tests,
  builds both graphs and checks them. Timing tests get a `benchmark` marker and run locally only.
- `text2icd.graph.model.load(release) -> Graph`: the built graph in memory, with accessors for codes
  (`code`, `parent`, `children`, `ancestors`, `seven_char_def`, `notes` and `edges`, optionally
  inherited from ancestors), Index terms (`term_at`, `subterms`, `pointers`, `xrefs`), reverse
  lookups and search. FY2027 loads in about 0.85 s within 255 MB.
- `text2icd graph show "<Index path>"` and `text2icd graph code <code>` print a term or a code with
  its notes, edges, cross-references and the Index terms that give it.
- Full-text search in the graph: FTS5 tables over every Index term (title, modifiers and path) and
  over Tabular descriptions and inclusion terms (`tabular_term`), with `graph.search.search_index`
  and `search_tabular` (BM25, free-text queries made safe for FTS5). `build-graph --tokenizer`
  chooses `porter` (default) or `unicode61`; the choice is recorded in `meta`.
- Index cross-references (`index_xref`): every `see` and `see also` resolved to the Index term it
  names, by longest match on titles that contain commas, with table columns taken from the column
  headings (`Neoplasm, benign, by site`), books, `by site` and code instructions kept as such, and
  the words a partial match leaves over. About 88% resolve fully and 0.1% do not; those 16 are
  Index errors listed with a reason in `resources/xref_exceptions.toml`, and the build warns about
  any other.
- Tabular note edges (`tabular_edge`): every code or range named in an instructional note
  (`(N18.1-N18.6)`, `(E10.-)`, `category T51`), with the stray dots of the CMS files handled. About
  11,600 per release; the build log gives the share of notes with a reference by kind, the words
  that look like codes (`B12`, `GM2`) and any reference with no code in the release.
- Evaluation dataset format (`text2icd.eval.datasets`): JSONL examples with a phrase, the gold code
  set (one primary code, then companion codes, each with its status against the release), the
  release, source, split and free metadata; a content hash independent of line order; and
  `text2icd eval validate` to check files against the release's billable codes.
- Version reconciliation (`text2icd.eval.versions`): each code assigned under an older release is
  `valid` in the evaluated release, `mapped` to the codes that replaced it (from the CMS conversion
  tables followed across releases, or a header's subdivisions) or `unmappable`.
  `text2icd eval reconcile` looks up codes or rewrites a dataset against a release.
- Evaluation metrics (`text2icd.eval.metrics`): accuracy@1 and @5, category accuracy, code-set
  precision, recall and F1, invalid-code rate, calibration error, selective accuracy by coverage,
  LLM call rate, cost per 1,000 phrases and latency percentiles, with 95% bootstrap intervals,
  overall, for valid and mapped gold codes, and per group (such as chapter).
- `cms-synthetic` (`text2icd eval synth`): diagnosis phrases generated from a release's graph (Index
  paths read as phrases with their manifestation codes, inclusion terms, Neoplasm Table and Table
  of Drugs rows, 7th characters stated in the phrase), each also perturbed once (nonessential
  modifier, word order, abbreviation or typo). FY2027 gives 142,757 examples; a stratified sample
  of 1,067 is committed in `benchmarks/cms-synthetic/`. Datasets get a manifest recording how they
  were made. `Graph.terms()` and `Graph.tabular_texts()` iterate over Index terms and Tabular texts.
- Evaluation runner: `text2icd eval run -d <dataset> -s <system>` scores a system on a dataset split
  and writes a result file with its provenance (git commit, release, graph and dataset hashes,
  system settings) and its metrics; `text2icd eval report` renders a markdown comparison, with
  per-chapter tables on request. First system: the B1 BM25 baseline (`-s bm25`). The CI
  integration job runs it on the committed `cms-synthetic` sample and now also fetches the FY2016
  to FY2025 code lists.
- LLM cost meter (`text2icd.llm.cost`): token usage read from each response (input, output, cache
  reads, 5-minute and 1-hour cache writes), priced from one table (`resources/prices.toml`, with
  long-context and batch prices), and a per-run meter that refuses further calls once the budget
  is spent.
- B2 embedding baseline (`-s embedding`): nearest Tabular descriptions and inclusion terms by
  cosine similarity of a local static embedding model (`text2icd.embeddings`; optional extra
  `embeddings`, model2vec; default `NeuML/pubmedbert-base-embeddings-8M`). CI installs all extras.
  On the FY2027 `cms-synthetic` dev split, B1 reaches 38.8% accuracy@1 and B2 40.6%.
- MIMIC-IV discharge diagnosis extractor (`text2icd.eval.mimic.sections`): the section of each
  discharge summary split into items, with primary or secondary group labels, bullets, numbering,
  underlines and leading labels handled; `text2icd eval mimic sections` prints aggregate counts
  only. Tests use hand-written synthetic notes.
- `mimic-silver` (`text2icd eval mimic silver`, local only): discharge diagnosis items aligned to
  their admission's billed ICD-10 codes by SapBERT similarity (run through ONNX, optional extra
  `embeddings`), one-to-one per admission above a score and a margin, with each code's status
  against the release and a patient-level dev/test split. `text2icd eval mimic freeze` records a
  set's test split, and rebuilds or test runs that would change it are refused. The reconciler now
  keeps each code's description.
- Silver-label audit (`text2icd eval mimic audit sample | review | score`): a seeded sample of
  `mimic-silver` dev alignments reviewed in the terminal, and the label precision with a 95%
  Wilson interval, overall and by alignment score.
- `mimic-gold` tooling (`text2icd eval mimic gold sample | annotate | build`) and the annotation guide
  (`docs/annotation-guide.md`): phrases drawn from `mimic-silver` by code frequency band and chapter,
  coded by hand in a terminal session with Index lookups and code views, every code checked against
  the release, and the answers built into a dataset with its agreement with the silver labels.
- B3/B4 LLM baselines (`-s llm`): the model chooses among B2's top 20 codes through a JSON schema
  of candidate numbers, one call per phrase, priced per call and stopped at the run's budget.
  Providers live behind one interface (`text2icd.llm.providers`, optional extra `llm`; Anthropic
  API first), and `eval run` refuses LLM systems on MIMIC-IV datasets unless the endpoint is on
  the cited allow-list and `TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT` is on.
- Main-term recall (`text2icd eval main-terms`): the share of phrases for which Index search finds a
  main term leading to the gold code, at 1, 5 and 10, searching main terms or all terms. It
  confirms the `porter` tokenizer as the default: 49.6% against 45.9% for `unicode61` at 1 on
  `mimic-silver` dev, and 88.3% against 83.6% at 10 searching all terms.
- Phrase normaliser (`text2icd.text.normalize`): Unicode NFKC, dashes and quotes, list markers and
  spacing, clinical shorthand (`w/`, `w/o`, `s/p`, `c/b`, `2/2`, `h/o`, `r/o`, `hx`, ...), case
  folding, and Roman numerals of stages and diabetes types, with every rewrite recorded as a trace
  step (`text2icd.trace.TraceStep`).
- Abbreviation lexicon (`src/text2icd/resources/abbreviations.toml`, `text2icd.text.abbreviations`):
  440 clinical abbreviations written from medical knowledge, expanded before case folding.
  Ambiguous ones (MS, ARF, PD) are resolved by hint words of the phrase or take their first meaning
  with a flag, entries that are also English words (ALL, AS, CAP) match only in capitals, and
  abbreviation-like tokens with no entry are listed. `normalize` takes the lexicon.
- Abbreviation mining (`text2icd eval mimic abbrev mine`, local only): counts the words of dev-split
  MIMIC-IV discharge diagnoses that neither the lexicon nor the CMS files know, writes them to the
  git-ignored `data/interim/`, and prints aggregate counts and only words used in 50 or more notes.
  34 lexicon entries were written from its counts. `text2icd.text.terms.vocabulary` gives the words
  of the CMS files.
- Modifier extractors (`text2icd.text.modifiers`): laterality, acuity, stage, severity, episode of
  care, fracture type, organism (named as the Index does), drug resistance, diabetes type, drugs
  and control, with or without complication, pregnancy, trimester (also derived from weeks of
  gestation), and remission, each with a canonical value and its span in the phrase.
- Status, negation and uncertainty (`text2icd.text.cues`): history, family history, status post and
  resolved, negation, and the uncertainty terms of Guideline II.H, before or after a condition. A
  `setting` (`inpatient` by default, or `outpatient`) applies the uncertain-diagnosis rule: coded
  as if established inpatient (II.H, III.C), not coded outpatient (IV.H). Resolved is a personal
  history, never remission.
- Condition splitter and parsed phrases (`text2icd.text.split`, `text2icd.text.parse`,
  `text2icd parse "<phrase>"`): list items, comma fragments, linking connectors (with, without, due
  to, secondary to, complicated by, in the setting of, on) and lists (and, or, vs, /) give the
  conditions of a phrase, their relations and their diagnoses. Index terms from the graph
  (`text2icd.text.terms.Terms`) decide which parts name a condition, keep CMS terms and Index
  "with" combinations whole ("nausea and vomiting"), and let a comma continue an Index path
  ("anemia, hemolytic"). Each condition gets the cues that cover it, its status, certainty and
  whether it is coded, in a `ParsedPhrase` with the trace of every step.
- Modifiers: the Tabular List's 7th-character wording ("initial encounter for closed fracture") is
  an episode of care, and may hold a fracture type.

### Fixed

- Code validation accepts a letter in the second position, as in the `QA0`-`QA1` genetic disorder
  codes added in FY2026.
- Neoplasm Table cells marked `-` are read as not applicable, like `--` in the Table of Drugs.
- Code validation requires a digit in the second or third character, so words such as `HIV` or
  `SCFE` are no longer taken for codes.
