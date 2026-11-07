# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `text2icd.resolve.resolver.Resolver` (tasks 9.2 and 9.3). It is the one place that builds
  the cascade from the settings:
  - one Index walker for both settings, since the setting is now per phrase;
    `IndexWalker.walk`, `Coder.code` and `Coder.deterministic` take `setting`;
  - a Tabular verifier per episode of care, built when first asked for;
  - the calibration;
  - L2's retriever, loaded on the first escalated phrase (`retrieve.LazyRetriever`; the vectors
    are memory-mapped);
  - L3 with `llm=True`.

  It keeps the last 4,096 code sets in memory, marked `cached`; nothing derived from a phrase
  is written to disk, in place of the planned SQLite L0. It refuses phrases over 1,000
  characters and is safe to share between threads. The CLI's `code` and the `coder` eval system
  are built through it.
- `Resolver.code_many` (task 9.5): any number of phrases, results in input order, streamed 256
  at a time. A phrase repeated within a batch is coded once, and its repeats are marked
  `cached`. Each phrase is still coded alone: searching a batch's L2 queries together was about
  24% faster on mimic-silver dev, but moved L2's confidence in the fourth decimal for 0.2% of
  phrases, depending on the other phrases in the batch.
- Private structured logs (task 9.7, `text2icd.logs`). Each coded phrase logs its length, a hash
  salted per process, the number of codes, the layer, escalation, cost and time; each batch
  logs its size. The phrase and its codes are logged only with `TEXT2ICD_LOG_INPUTS=true`.
  Nothing is printed until a level is set; `logs.configure()` writes JSON lines to stderr.
- Reproducibility tests (task 9.6): fresh interpreters with different hash seeds give the same
  result JSON, byte for byte, on the fixture graph and on the committed sample. With L2 as well,
  the cms-synthetic and mimic-silver dev sets gave identical code sets under two seeds.
- `text2icd.result.CodingResult` (task 9.1), the public result:
  - chunks with their spans in the input;
  - codes in coding order, with role, `primary`, source and the chunks each one codes;
  - one calibrated confidence, alternatives, a flat trace, flags, cost;
  - with `explain`, the per-diagnosis Index paths, notes, Excludes1, L2's pool and L3's
    decisions.

  Its JSON Schema is `docs/schema/coding-result.json` (`python -m text2icd.result`).
- `text2icd code --explain` and `--llm` (L3, opt-in; refused without a provider and a model).
- Spans in the phrase as written (task 9.11, for the UI's chunk view).
  - Normalisation runs on a `text.tracked.Tracked` text, which records where each character came
    from, so `ParsedPhrase.source(start, end)` maps a condition, modifier, cue or relation span
    back to the input: "chronic kidney disease stage 3a" was written "CKD 3a".
  - `Expansion` gains `start`/`end` in the input, and `Relation` gains its connector's span.
  - L3's abbreviation rewrite keeps the spans on the user's text (`ParsedPhrase.rebased`); the
    code set's `parsed.original` is now what the user wrote, not the rewritten phrase.
  - Codes and the rest of the JSON are unchanged on the cms-synthetic and mimic-silver dev sets.
- Paths and links for the UI (task 9.11, part 2).
  - Each Index candidate carries its path as graph terms (`Candidate.steps`: term id, title,
    the words it uses, see or see also, the cross-reference as written, the table column) and
    `coded_step`, the step that gives the code.
  - Each coded diagnosis and each code of the sequence lists the conditions (chunks) it codes
    (`conditions`): in "DM2 w/ CKD 3a, possible UTI 2/2 E. coli", E11.22 and N18.31 code "DM2"
    and "CKD 3a".
  - Trace steps name the Index term or Tabular code they are about (`term`, `code`).
  - The code set keeps the codes L2 weighed, with their probabilities (`pool`), and what L3 was
    asked and answered (`decisions`, without the phrase).
  - `Graph.xrefs_to` gives the cross-references that lead to a term.
  - Codes and the rest of the JSON are unchanged on the cms-synthetic and mimic-silver dev sets.
- `text2icd eval bench` (task 9.8) times a local system on one thread. It reports the import,
  graph load and setup times; peak memory; the first phrase; p50, p95, p99 and max latency;
  phrases per second; and the slowest example ids (ids only, never phrases). A `benchmark` test
  holds the deterministic path to p50 under 20 ms and at least 200 phrases per second.
- `eval run` and `eval report` show p99 and max latency next to p50 and p95
  (`eval.metrics.latency_summary`).
- `graph.model.read_meta` reads a graph file's `meta` table without loading the graph, and
  reports a file that is not a graph as a `GraphError`.
- Offline evaluation through Message Batches (task 8.8): `coder -O llm=batch` answers L3's
  decisions in rounds of batches (`llm.batch`: `BatchModel`, `Collector`, `Replay`,
  `answer_in_rounds`) at the batch price, keyed by their decision-cache key, then codes every
  phrase from the cache. The budget holds before each batch (an upper bound of each request's
  cost), answers are checked against the options, and each batch is deleted from the provider
  once its results are read. Anthropic API only; Bedrock and Vertex AI offer no batches.
- L3 in the coder (tasks 8.3 and 8.9): `resolve.navigate` asks the navigator, for a phrase L1
  and L2 still escalate, an abbreviation's meaning, the main term (when the Index entries are
  many), the Index entry (with retrieval's codes after the entries) and the Tabular child (when
  the phrase does not settle it, the 7th character's episode of care included), and codes the phrase again with each choice. No option keeps the
  coder's code set, flagged `needs_review`. `TabularVerifier.open_choice` finds the node to ask
  about. `CodeSet` records `llm_calls`, `cost_usd` and `usage`, and `resolver` is `llm` when L3
  changed the primary code. The `coder` eval system takes `-O llm=on` (and `provider`, `model`,
  `effort`, `budget`, `cache`), reports the provider as its endpoint for the MIMIC-IV policy, and
  is labelled `coder + L2 + L3 (<model>)`. `llm.cache.DecisionCache` stores answers by hash in
  `.text2icd/decisions.sqlite`, never the phrase.
- `eval run` stops at the first error no retry can fix (bad credentials, no access, unknown
  model) instead of recording it on every phrase.
- The L3 navigator interface (tasks 8.1, 8.4 to 8.7, 8.10): `text2icd.llm.navigator`
  (`Navigator.choose(decision) -> Choice`, `LLMNavigator`, `NoNavigator` for provider `none`) with
  decisions of four kinds (main term, Index entry, Tabular child, abbreviation sense). One cached
  system prompt with the coding rules (about 850 tokens, above the 512-token minimum Claude
  caches), then the fenced phrase, the path and the numbered options; the answer schema is an enum
  of the option ids and `none`, and any other answer counts as `none`. Limits: calls per phrase,
  the run's budget, `TEXT2ICD_LLM_TIMEOUT_S` per attempt and `TEXT2ICD_LLM_MAX_RETRIES` retries of
  retryable errors only. Prompt-injection tests check that a phrase cannot get an unlisted answer
  through.
- Amazon Bedrock (`bedrock`, `AnthropicBedrockMantle`) and Google Cloud Vertex AI (`vertex`,
  `AnthropicVertex`) providers, with the extras `bedrock` and `vertex` and the settings
  `TEXT2ICD_LLM_REGION` and `TEXT2ICD_LLM_PROJECT`. Neither has a price in `prices.toml` yet, so
  runs on them are refused until one is added.
- `eval run` warns when an LLM system's prompt cache was never read after its first calls.
- `text2icd eval check` (task 3.14): checks result files against an expectations file of
  bounds per system (`accuracy@1`, a subset's `kept/accuracy@1`, counts such as `errors`) and
  exits 1 when any fails or when a run is of another dataset, split, release or sample size.
  `benchmarks/cms-synthetic/fy2027-sample.expected.toml` holds the bounds for the committed sample,
  and the CI integration job now runs the coder on it and fails on a regression.
- Confidence for the deterministic coder (tasks 7.1 to 7.3). Each code set carries its `features`
  (path coverage, score margin, default or unspecified codes, Tabular ties, typo corrections,
  cross-references, rule changes, text flags), a calibrated `confidence` that the primary code is
  right, and `escalate` when that confidence is below the threshold chosen for 98% precision.
  `text2icd code` prints it, and the `coder` eval system reports it, so runs now show calibration
  error and selective accuracy.
- `text2icd eval calibrate`: fits the calibration (logistic score, isotonic map) on a split, with
  held-out results from folds grouped by gold code, and writes it as a small JSON table. The shipped
  table, `resources/calibration.json`, is fitted on `cms-synthetic` dev; `TEXT2ICD_CALIBRATION`
  selects another one. Tables fitted on restricted data are refused inside the package.
- An `eval` dependency group with scikit-learn, installed by default with `dev`; the runtime does
  not need it.
- `eval run` and `eval report` print a selective-accuracy table for runs that report a confidence.
- The Tabular step flags `tabular_tie` when children tie and the first is taken.
- `text2icd.resolve.retrieve`, the retrieval layer of L2: ranks the graph's Tabular texts and
  Index paths against a phrase, by local embeddings or BM25, and returns distinct codes for
  Tabular verification to complete.
- `text2icd eval retrieval` (task 7.4): recall@1, 5 and 20 of retrievers (`bm25`,
  `model2vec:<id>`, `onnx:<id>`) over chosen texts, with their codes completed to billable ones,
  on all examples, on those the coder escalates and on those it gets wrong; with latency and
  vector and model sizes. `embeddings.load_embedder` reads the model names.
- The vectors file of local retrieval (task 7.5). `text2icd graph embed`, or `text2icd build-graph
  --embeddings`, embeds the graph's texts with L2's model: bge-small-en-v1.5 by default, with its
  full-precision weights, whose vectors do not depend on the batch. It writes
  a float16 matrix next to the graph file, with a JSON file naming the model, the texts' sources,
  the graph and the code of each row: 94.8 MB for FY2027. `resolve.retrieve.load_vectors` reads it
  and refuses vectors built for another graph. `eval retrieval -m vectors` measures it.
- L2 in the coder (task 7.6). When a code set is escalated and the coder has a retriever, the
  retrieved codes compete with the coder's code and alternates. The calibration table's new
  retrieval stage scores each one (`resolve.choice`); the best wins, and its probability becomes
  the confidence. A retrieved code is coded again as the first candidate of its diagnosis, so
  that its notes and 7th character are read as usual. Code sets now say which layers escalated
  them (`escalated`), where the primary code came from (`resolver`) and what L2 chose it by
  (`choice`).
- `text2icd eval calibrate` fits the retrieval stage on the escalated examples when the graph's
  vectors file exists (`--retrieval/--no-retrieval`). The shipped table has one, fitted on
  `cms-synthetic` dev with bge-small-en-v1.5.
- `text2icd code --retrieval/--no-retrieval`, on by default when the vectors file and a staged
  table are both there; the `coder` eval system takes `-O retrieval=on`.
- Predictions record the layers that escalated them. `eval run` and `eval report` show accuracy
  on the phrases the coder keeps and on those it escalates; a run with L2 is labelled
  `coder + L2`.

### Changed

- Extras (task 10.6). `embeddings` now declares what L2 imports (onnxruntime, tokenizers,
  huggingface_hub). `model2vec`, used only by the B2 baseline and the MIMIC-IV aligner, moves to
  a new `eval` extra with scikit-learn (calibration fitting), which replaces the `eval` dependency
  group. `all` is `embeddings` and `llm`. A core install is 17 packages, `all` 40.
- Install hints name a checkout (`uv sync --extra ...`) or GitHub
  (`pip install "text2icd[...] @ git+https://github.com/a2psilva/text2icd"`), never a bare
  `pip install text2icd`, since text2icd is not on PyPI (`text2icd.extras`).
- L2's model is read at a pinned Hub revision (`embeddings.PINNED`; bge-small at `ea104dac`), or
  at the revision its vectors file records (`VectorsMeta.revision`, new). A model already
  downloaded loads with no network call: 0.17 s with the Hub unreachable, where each load used to
  ask the Hub first. When the model is not downloaded, L2 stays off with a warning unless asked
  for (`--retrieval`), rather than stall or fail the first escalated phrase.
- Faster coding, with identical output (task 9.12). On the committed sample the coder starts in
  1.89 s instead of 2.78 s and codes 814 phrases/s instead of 612, and its slowest phrase takes
  6.9 ms instead of 272 ms. On mimic-silver dev it codes 1,333 phrases/s instead of 1,007.
  - The resolver calls `gc.freeze()` once built, so a full garbage collection no longer walks
    its 1.2 million objects in the middle of a phrase.
  - `MainTermFinder.candidates(..., limit=)` scores a main term only if an exact upper bound on
    its score can still reach the top `limit`: 86% fewer are scored.
  - The speller compares a word only with Index words of a near length.
  - Start-up: an ASCII fast path in `text.terms.words`, and each Index profile built once.
  - `eval bench` loads L2 during setup; before, the model load was timed as a phrase.
  - The full code-set JSON of every cms-synthetic and mimic-silver dev phrase is unchanged, with
    and without L2.
- The slow tests share one FY2027 coder (`release_coder` in `tests/conftest.py`) instead of
  building one per module or per case.
- `text2icd code --json` prints a `CodingResult` instead of the internal `CodeSet`.
- The `Resolver` literal of `resolve.coder` (the cascade's layers) is now `Layer`.
- Tests no longer read the developer's `.env` file or `TEXT2ICD_*` and `ANTHROPIC_*` variables
  (an autouse fixture in `tests/conftest.py`). Before, a local `.env` that named a provider
  failed three tests, and an API key in it could have reached a paid endpoint.
- The endpoint allow-list for restricted data now holds `bedrock` as well as `anthropic`, both
  re-checked against the PhysioNet guidance on 2026-11-04; `vertex` is not on it, since the
  guidance names Vertex AI for Gemini models only.
- The B3/B4 baseline (`llm` eval system) builds its model with the new provider settings, so it
  runs on Bedrock and Vertex AI too, with the same timeout and retry limits.
- `.env.example`: the `TEXT2ICD_LLM_REGION`/`PROJECT`/`TIMEOUT_S`/`MAX_RETRIES` settings replace
  the `GOOGLE_CLOUD_*` lines, which the SDK does not read.
- Retrieval texts with the same words for the same code are kept once (129,283 to 123,486 for
  FY2027).
- `VectorIndex` keeps float16 vectors as they are and widens them block by block while searching.
- ONNX model names take an optional weights file: `onnx:<id>:<file>`. The default stays the
  quantized weights.
- `EmbeddingRetriever` takes the codes and vectors of its texts; `EmbeddingRetriever.embed`
  embeds texts.

### Fixed

- Thread safety for a server (task 9.10). Coding a phrase from any thread but the one that
  loaded the graph failed with `sqlite3.ProgrammingError` when it reached the Index search.
  - A `Graph` now opens one read-only connection per thread (and per process after a fork), so
    searches in different threads also run in parallel.
  - `DecisionCache` shares its connection under a lock, in WAL mode with a busy timeout, so
    another process can use the same file.
  - `CostMeter` updates its totals under a lock.
  - The word-form and spelling memos, keyed by words from the input, are bounded.
  - A file that is not a graph raises `GraphError` instead of `sqlite3.DatabaseError`.
- Index search within one book (task 9.9): SQLite's planner ran one full-text MATCH per term
  of the book (340 ms for "of" in the Table of Drugs). The query now scans the search table
  first: 2 ms, the same rows. The coder codes the sample about 1.8 times as fast (644 against
  361 phrases/s), its slowest phrase drops from 489 to 268 ms, and with L2 from 491 to 21 ms. A
  test checks the query plan, since SQLite versions differ.

## [0.1.0] - 2026-11-03

The deterministic coder (Phases 0 to 6): Index walk, Tabular verification, instructional notes and
the guideline rule pack, with the evaluation harness. Results: `docs/results.md`.

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
- `text2icd eval parse -d <dataset>`: how text understanding reads a dataset split, in aggregate
  counts, and the share of phrases split into the expected number of diagnoses (one per
  `cms-synthetic` phrase, or `meta.diagnoses`). The `mimic-gold` annotation now records it: commas
  separate the codes of different diagnoses (annotation guide rule 12).
- Abbreviation review (`text2icd eval mimic abbrev sample | review | score`, local only): 200
  expansions from dev discharge diagnoses, one per note, judged by hand, and the expansion
  accuracy with a 95% Wilson interval. The audit's review loop takes the lines to show.
- Main-term selection for the Index walk (`text2icd.resolve.main_terms`): phrase and Index words
  compared in shared forms (`text2icd.resolve.words`: singular, anatomical adjective to noun from
  `resources/word_forms.toml`), typos corrected to the nearest Index word (`rapidfuzz`, a new
  dependency), main terms that only say "see condition" dropped, and candidates ranked by what
  they and their subterms can use, then by the head noun of the phrase. `text2icd eval main-terms`
  measures it as the `coder` mode.
- Subterm walk of the Alphabetic Index (`text2icd.resolve.walk`): a beam search from the best main
  terms, per diagnosis, with nonessential modifiers as a small bonus (the subentry wins, I.A.7),
  connector subterms passed through, `without` honoured, and the default code (I.A.18) flagged
  `default_code_used`. It returns ranked candidates with their paths, flags and trace.
  `text2icd walk "<phrase>"` shows them, `text2icd eval index` measures the walk alone (prefix
  match of the gold code, by generator), and `eval run -s index` scores it like the baselines.
- The Index walk follows cross-references (I.A.16): `see` jumps to its target and `see also` adds
  it as another branch, continuing with the words not used yet (and those a partial reference
  leaves), with loop protection; references into the Neoplasm Table carry their column. Connector
  subterms follow I.A.15: "with" and "in" link listed conditions (`presumed_link` in the trace),
  "due to" needs the phrase to state the cause (otherwise `undocumented_link`, at a cost), and
  listed diagnoses are joined only through a presumed link.
- Index router (`text2icd.resolve.route`): neoplasm behaviour wording is looked up by site in the
  Neoplasm Table at that column; poisoning, overdose, adverse effect, underdosing and toxicity
  wording by substance in the Table of Drugs and Chemicals at the intent column (unknown intent
  coded as accidental, I.C.19.e.5(b); toxicity as an adverse effect, flagged); a stated
  mechanism of injury adds External Cause Index codes with role `external_cause`; history,
  family history and status-post conditions are looked up under History and Status. The
  `index` evaluation system adds the external cause code after the diagnoses.
- Tabular verification (`text2icd.resolve.tabular`) and the deterministic coder
  (`text2icd.resolve.coder`): Index candidates are completed to billable codes by matching the
  words of child descriptions and inclusion terms, with the "unspecified" child when the phrase
  says no more (`unspecified_chosen`), fractures displaced and closed unless stated otherwise,
  the 7th character from the stated episode of care or `TEXT2ICD_DEFAULT_EPISODE`
  (`seventh_character_assumed`) and placeholder `X`s; a diagnosis with no completable candidate
  gets no code (`needs_review`), never an invalid one. `text2icd code "<phrase>"` prints the code
  set and its trace, and `eval run -s coder` scores it.
- Instructional notes in the code set (`text2icd.resolve.instructions`): `code first`, `use
  additional code` and `code also` companions are taken from the set or found among the Index
  paths the phrase documents in full, with a second walk of each condition ("type 2 DM with
  diabetic CKD stage 3a" gives E11.22 and N18.31); optional instructions need words of their
  own, and a mandatory one found nowhere is listed as `required_unresolved`. Excludes1 conflicts
  keep the more specific code (`excludes1_conflict`, the other listed under `excluded`). The
  codes are sequenced by the notes: `code first` before, `use additional code` after,
  manifestations after their etiology, and a code "in diseases classified elsewhere" never
  first. `CodeSet.sequence` gives each code's role and the code that brought it in.
- Guideline rule pack (`text2icd.resolve.rules`), each rule traced with its section in a new
  `rules` trace layer: a condition its diagnosis's code leaves unused is coded on its own
  (I.B.9); acute and chronic get both codes, acute first (I.B.8); hypertension with heart
  disease, CKD or both becomes I11, I12 or I13 with its I50 and N18 codes (I.C.9.a.1-3);
  severe sepsis and septic shock get R65.20 or R65.21 after the systemic infection, A41.9 when
  none is documented (I.C.1.d.4); diabetes drugs, by name too, add Z79.4, Z79.84 and Z79.85
  (I.C.4.a.3); poisonings come before their manifestations and adverse effects and underdosing
  after the condition (I.C.19.e.5); the residual before the sequela (I.B.10).
- Tests for every guideline section the coder cites and every worked example of
  `docs/coder-workflow.md` (`tests/resolve/test_guidelines.py`); a new rule cannot be added
  without a case, since the test reads the cited sections from the source.
- `docs/results.md`: the deterministic coder against the baselines, aggregates only.
- Index-walk tests on FY2027: the worked examples of `docs/coder-workflow.md` and 110 cases drawn
  from the Index across all code chapters (`tests/resolve/fy2027_index_walks.json`).

### Fixed

- A family history is answered only by a family history code, and a personal history never by
  the condition itself: "family history of colon cancer" was Z85.038 (personal history), and
  "family history of lung cancer" is Z80.1. Under History, "cancer" and "malignancy" are read as
  "malignant neoplasm", where the Index's "Cancer, see Neoplasm, malignant" leads.
- Diagnoses listed side by side join into the combination code the Index presumes (I.A.15) even
  when the second states a stage the combination leaves to its `use additional code` note:
  "type 2 diabetes, CKD stage 3a" gives E11.22 and N18.31, not E11.9 and N18.31.
- Several `use additional code` notes that each say "code to identify ..." are separate
  instructions, not a list of alternatives: I13.0 and I13.2 now require both the heart failure
  and the CKD stage code. A mandatory `use additional code` note left unanswered takes the
  unspecified code it names when the title of its code states the condition (N18.9, I50.9).
- "Type 2 diabetes with oral hypoglycemics" no longer gives E11.638 (oral complications): the
  words of a diabetes drug are not walked.
- The Index walk no longer starts from a main term the phrase says is absent: "sepsis without
  septic shock" is no R65.21, and "osteoporosis without current pathological fracture" is M81.0,
  not M80.00XA.
- A bare `not` after a condition's first words ("epilepsy not intractable", "hernia not specified
  as recurrent") qualifies what follows it: the condition is no longer judged negated, and "not"
  stays in its text.

- Code validation accepts a letter in the second position, as in the `QA0`-`QA1` genetic disorder
  codes added in FY2026.
- Neoplasm Table cells marked `-` are read as not applicable, like `--` in the Table of Drugs.
- Code validation requires a digit in the second or third character, so words such as `HIV` or
  `SCFE` are no longer taken for codes.
