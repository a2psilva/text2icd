# Architecture

Status: design. The sections below describe the target system. Names of modules, tables and
fields are the planned ones and may be refined during implementation. The main design decisions,
with the alternatives that were considered, are recorded in [docs/adr/](adr/README.md).

## Goals and non-goals

Goals:

- Turn a diagnosis phrase into a valid, billable ICD-10-CM code set for a given fiscal-year release.
- Reproduce the coder's procedure (Index, then Tabular, then notes), so every code comes with a
  trace a human can audit.
- Keep the marginal cost close to zero: the deterministic path does most of the work, and an LLM
  is consulted only at ambiguous decision points, with hard limits.
- Never return a code that does not exist in the selected release.

Non-goals (for now): reading full notes, ICD-10-PCS procedure codes, principal-diagnosis sequencing
for an encounter, and reimbursement logic (MS-DRG, HCC).

## Component overview

```mermaid
flowchart LR
    subgraph build["Build time (once per release)"]
        CMS[(CMS XML + order file)] --> ONT[ontology parsers]
        ONT --> GB[graph builder]
        GB --> DB[(graph SQLite + FTS5)]
    end
    subgraph run["Run time (per phrase)"]
        Q[phrase] --> TXT[text understanding]
        TXT --> CAS[resolve: cascade]
        DB --> MEM[in-memory graph] --> CAS
        CAS <--> LLM[llm navigator]
        CAS --> RES[CodingResult]
    end
```

| Package | Responsibility |
| --- | --- |
| `text2icd.ontology` | Stream-parse the CMS XML (Tabular, Index, External Cause Index, Neoplasm Table, Drug Table) and the code order file into typed records. No interpretation beyond what the files state. |
| `text2icd.graph` | Build nodes and edges from those records, resolve cross-references, persist to SQLite with FTS5, load into memory, and expose a traversal API. |
| `text2icd.text` | Normalize phrases, expand abbreviations, split a phrase into conditions, and extract modifiers and flags. |
| `text2icd.resolve` | The cascade: result cache (L0), Index walk (L1), Tabular verification, confidence, local retrieval (L2), and the orchestration that calls L3. |
| `text2icd.llm` | Provider clients, constrained-choice prompts, budget and call caps, and the endpoint policy. |
| `text2icd.api` | FastAPI service around `resolve`. |
| `text2icd.eval` | Datasets, metrics, baselines and the benchmark runner. |
| `text2icd.cli` | Typer CLI: `data fetch`, `build-graph`, `code`, `batch`, `eval`, `serve`. |

## Knowledge graph

### Node types

| Node | Key | Main attributes |
| --- | --- | --- |
| `Chapter` | chapter number | title, code range |
| `Block` (section) | section id, e.g. `E08-E13` | title |
| `TabularCode` | code with dot, e.g. `E11.22` | description, billable flag, placeholder flag, depth, inclusion terms, inherited 7th-character definition |
| `SeventhCharacter` | (owner code, char) | meaning, e.g. `A` = initial encounter |
| `IndexTerm` | stable id | source (`index`, `eindex`, `neoplasm`, `drug`), level, title, nonessential modifiers, full path, code, manifestation code, `subcat`, `seecat` |
| `TableCell` | (IndexTerm id, column) | code for one column of the Neoplasm or Drug table, e.g. *Malignant Primary* |

### Edge types

| Edge | From → To | Source in the XML |
| --- | --- | --- |
| `parent_of` | Tabular → Tabular | `<chapter>` / `<section>` / `<diag>` nesting |
| `has_subterm` | IndexTerm → IndexTerm | `<mainTerm>` / `<term level>` nesting |
| `points_to` | IndexTerm → TabularCode | `<code>`, `<subcat>`, `<seecat>` (a trailing `-` means "incomplete") |
| `manifestation` | IndexTerm → TabularCode | `<manif>` (sequenced after the etiology code) |
| `see`, `see_also` | IndexTerm → IndexTerm | `<see>`, `<seeAlso>` text resolved to an Index path |
| `excludes1`, `excludes2` | TabularCode → TabularCode or range | `<excludes1>`, `<excludes2>` note text, code references parsed out |
| `code_first`, `use_additional`, `code_also` | TabularCode → TabularCode or range | the matching note elements |

Cross-reference text such as `Pneumonia, in (due to)` is read as a path of comma-separated segments
and resolved by longest match, first against main-term titles, because titles themselves contain
commas (`Abdomen, abdominal`), then against subterms. Titles are compared in lenient forms (case,
parenthesized words, NEC and NOS, hyphens, apostrophes, plurals, the parts of comma and "or"
titles). Parts of a reference that are instructions are kept as such: `by site`, a table column
(`Neoplasm, benign, by site` is the Benign column of the Neoplasm Table, matched against the column
headings of the release), a book (`Table of Drugs and Chemicals`), a code instruction
(`category M01`) or a generic one (`condition`). A reference is `resolved`, `partial` (a term is
reached and the words left over are kept for the coder), `instruction`, or `unresolved`. In FY2026
and FY2027 about 88% resolve, 6% are partial, 6% are instructions, and 16 (0.1%) are unresolved;
those are errors in the Index, listed with a reason in `src/text2icd/resources/xref_exceptions.toml`.

### Billable code set

The XML lists most codes explicitly, but codes that need a 7th character exist only through an
inherited `sevenChrDef` (for example E08.3211). The builder generates them, padding with the `X`
placeholder where the code is shorter than six characters, and the result must match the CMS code
order file exactly.

### Storage

One SQLite file per release, for example `data/processed/text2icd-graph-fy2026.sqlite`, written by
`text2icd build-graph --release fy2026`:

- tables for tabular nodes, notes, edges, index terms, table cells and cross-references;
- FTS5 virtual tables over Index full paths, titles plus nonessential modifiers, Tabular descriptions
  and inclusion terms, which gives BM25 ranking without extra dependencies;
- a `meta` table with the release id, source file hashes and the builder version.

The schema is versioned SQL in `src/text2icd/graph/schema.sql`. Tables so far:

| Table | Content |
| --- | --- |
| `meta` | release id, schema and builder versions, SHA-256 of each source file, content hash |
| `order_code` | the CMS code order file as published: code, billable flag, short and long descriptions |
| `tabular_node` | chapters, sections, XML codes and the codes generated from 7th-character definitions, with parent, depth, billable flag, placeholder flag and the node whose 7th-character definition applies |
| `tabular_note` | every instructional note in document order, by XML element name (`excludes1`, `useAdditionalCode`, ...) |
| `seventh_char` | 7th-character definitions on the node that declares them |
| `index_term` | terms of the four Index books with global ids, essential title, printed title, nonessential modifiers and full path (`Pneumonia > aspiration > due to > food`) |
| `table_column` | column headings of the Neoplasm Table and the Table of Drugs and Chemicals |
| `index_pointer` | the `points_to` and `manifestation` edges: every code a term gives (`code`, `manif`, `subcat`, `seecat`, table cells), as written and resolved to a Tabular node, with a status (`ok`, `corrected`, `excluded`, `unresolved`) |
| `tabular_term` | searchable Tabular text linked to its code: every code description, inclusion term and `includes` note |
| `fts_index_term`, `fts_tabular` | contentless FTS5 tables over Index terms (title, nonessential modifiers and full path, weighted in that order) and over `tabular_term`; the tokenizer (`porter` by default, or `unicode61`) is chosen at build time and recorded in `meta` |
| `index_xref` | the `see` and `see_also` edges: every cross-reference as written, with its status, the term it reaches, the book, table column and instruction it names, the words left over, and for a known Index error the reason |
| `tabular_edge` | the `excludes1`, `excludes2`, `code_first`, `use_additional` and `code_also` edges: every code or range written in a note (any note kind), as written and as a range of code prefixes, with the Tabular node a single code names and a status (`ok`, `unresolved`) |

Generated codes take their description from the order file. Billable codes in `tabular_node` must
equal the billable rows of `order_code`. Known Index data errors are listed in
`src/text2icd/resources/index_exceptions.toml`: entries with a correction replace the written value
(status `corrected`), the others are kept out of the edges with a build warning (`excluded`).

Code references in notes are found with a pattern, then checked against the release. A range
resolves when at least one code lies inside it, since some ranges end at a WHO-only category
(`K00-K93`). A bare token that matches no code is a word, not a reference (`vitamin B12`,
`GM2 gangliosidosis`); the build log lists them. Any other reference with no code is kept as
`unresolved` with a warning: FY2026 has one, `G31.90` in a code-first note of F02.

At run time the graph is loaded into plain Python dicts and tuples (target: under 1 s and under
300 MB), so traversal never touches disk. SQLite was chosen over networkx (memory-heavy, no full-text
search) and over a graph database (an extra service to run): it is a single portable file, needs no
server, and can be attached to a GitHub Release because the CMS content is public domain.

## Resolution cascade

### Text understanding

`text` turns the phrase into a `ParsedPhrase`: one or more conditions, each with a head term,
modifiers (laterality, acuity, stage, episode of care, organism, pregnancy, site) and flags
(`negated`, `history_of`, `uncertain`, `ambiguous_abbreviation`). Splitting handles "c/b", "with",
"due to", "secondary to" and lists. A `setting` option (`inpatient` or `outpatient`) controls the
uncertain-diagnosis rule (Guidelines II.H, III.C, IV.H).

### L0: result cache

Key: normalized phrase, release id, resolver version and settings. Stored in a local SQLite file
under `.text2icd/` (git-ignored, because cached phrases may come from restricted data).

### L1: deterministic coder

1. **Route** each condition to the Neoplasm Table, the Drug Table, the External Cause Index, or the
   main Index, based on morphology words, poisoning or adverse-effect language, and external-cause
   language.
2. **Select the main term**: FTS5 BM25 over titles and nonessential modifiers, fuzzy matching for
   typos, and a lexicon of adjective-to-noun forms (renal → kidney). Prefer the condition over the
   anatomical site. Follow `see` redirects.
3. **Descend** subterms with a small beam search. A child's score is the share of the remaining
   modifiers it uses up. Connector subterms ("with", "due to", "in") consume linked conditions. When
   no child matches, stop and use the code at the current node (the default code, Guideline I.A.18).
4. **Pick the table column** for Neoplasm (behaviour) and Drug (intent: accidental, self-harm,
   assault, undetermined, adverse effect, underdosing) entries.

### Tabular verification

Shared by every layer, so L2 and L3 can only propose; Tabular verification decides.

- Descend from a category or partial code to a billable code, matching modifiers against child
  descriptions and inclusion terms. If the phrase gives no further detail, choose the "unspecified"
  child and add a flag.
- Apply the 7th character from the inherited definition and pad placeholders.
- Check the code exists in the release's billable set.
- Read notes: report Excludes1 conflicts within the code set, add `use_additional` and `code_first`
  requirements as companion codes (resolved when the phrase supports them, otherwise listed as
  required but unresolved), and order etiology before manifestation.
- Apply guideline rules, each citing its section (for example I.A.15 "with", I.C.9.a hypertension).

### Confidence and escalation

Features: share of input tokens consumed by the path, score margin between the two best paths,
whether a default or "unspecified" code was used, the number of Tabular children left undecided, and
fuzzy-match distance. A calibrated model (isotonic regression on the dev split) maps the features to
the probability that the primary code is correct. Below the threshold, the cascade tries L2, then L3.

### L2: local retrieval

Embeddings run on CPU through ONNX (fastembed or model2vec; the model is chosen by benchmark)
over Index full paths, Tabular descriptions and inclusion terms. Retrieved codes are fed back
into Tabular verification. No API cost.

### L3: LLM navigator

The LLM never writes a code. It chooses among options the graph provides:

- which candidate main term is right,
- which child to take at an ambiguous Index or Tabular node,
- what an unknown abbreviation means, from a short candidate list.

Prompt layout, from most stable to least, so caching works: a fixed system prompt with the coding
rules (long enough to meet the provider's minimum for prompt caching), then the phrase, the
path so far, and the numbered options. The response is a structured output whose `choice` field is
an enum of the option ids plus `none`. Requests use low reasoning effort and a small output limit.

Limits: a per-phrase call cap, a per-run dollar budget, and a cost log entry per call. When a limit is
reached the cascade returns the best deterministic answer with a `needs_review` flag.

Providers: hosted LLM services (for example Amazon Bedrock or Google Vertex AI) behind one
interface. With `require_dua_compliant_endpoint` on, only endpoints listed as approved in
[data-and-compliance.md](data-and-compliance.md) can be used. Offline evaluation uses the
provider's batch API where one exists, since batch requests are usually discounted.

## Output schema

```python
class AssignedCode(BaseModel):
    code: str  # "E11.22", always billable in the selected release
    description: str
    role: Literal["primary", "manifestation", "use_additional", "code_also", "external_cause"]
    confidence: float  # calibrated probability


class TraceStep(BaseModel):
    layer: Literal["text", "cache", "index", "tabular", "retrieval", "llm"]
    action: str  # e.g. "subterm", "see", "default_code", "seventh_character"
    detail: str  # human-readable, e.g. "Diabetes, diabetic > type 2 > with"
    guideline: str | None  # e.g. "I.A.15"


class CodingResult(BaseModel):
    input: str
    release: str  # e.g. "fy2026"
    codes: list[AssignedCode]  # ordered: etiology before manifestation
    alternatives: list[AssignedCode]
    trace: list[TraceStep]
    flags: list[str]  # e.g. "default_code_used", "unspecified_chosen", "needs_review"
    resolver: Literal["cache", "index", "retrieval", "llm"]
    llm_calls: int
    cost_usd: float
```

## Cost model

Per L3 step with a small model priced at about $0.10 per million input tokens and $0.50 per million
output tokens: about 300 uncached input tokens and a few hundred output tokens including reasoning
at low effort, so roughly $0.0002. With at most a few steps per escalated phrase and most phrases
resolved by L0 to L2,
the target is under $0.50 per 1,000 phrases. The evaluation harness measures the real figure,
including the cache hit rate, from the `usage` fields of each response.

## Releases and reproducibility

- A release is a CMS fiscal year plus an optional mid-year update: `fy2026`, `fy2026-apr`, `fy2027`.
  Every artifact, cache key and result carries its release id.
- The builder is deterministic: the same input files give the same graph content, checked by a
  content hash over the sorted rows. Source file hashes are recorded in `meta`.
- LLM-backed runs pin the model id and record it in the result, so benchmark runs can be repeated.
