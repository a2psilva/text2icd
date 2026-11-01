# Architecture

Status: design. The sections below describe the target system that [TASKS.md](../TASKS.md) builds.
Names of modules, tables and fields are the planned ones and may be refined as the phases land.

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

Cross-reference text such as `Pneumonia, in (due to)` is resolved by longest match against main-term
titles, because titles themselves contain commas (`Abdomen, abdominal`). Unresolvable references are
kept, flagged, and reported by the integrity tests.

### Billable code set

The XML lists most codes explicitly, but codes that need a 7th character exist only through an
inherited `sevenChrDef` (for example E08.3211). The builder generates them, padding with the `X`
placeholder where the code is shorter than six characters, and the result must match the CMS code
order file exactly.

### Storage

One SQLite file per release, for example `data/processed/text2icd-graph-fy2026.sqlite`:

- tables for tabular nodes, notes, edges, index terms, table cells and cross-references;
- FTS5 virtual tables over Index full paths, titles plus nonessential modifiers, Tabular descriptions
  and inclusion terms, which gives BM25 ranking without extra dependencies;
- a `meta` table with the release id, source file hashes and the builder version.

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

Embeddings run on CPU through ONNX (fastembed or model2vec; the model is chosen by benchmark in
Phase 7) over Index full paths, Tabular descriptions and inclusion terms. Retrieved codes are fed back
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
the target is under $0.50 per 1,000 phrases. Phase 3 and Phase 11 measure the real figure, including
the cache hit rate, from the `usage` fields of each response.

## Releases and reproducibility

- A release is a CMS fiscal year plus an optional mid-year update: `fy2026`, `fy2026-apr`, `fy2027`.
  Every artifact, cache key and result carries its release id.
- The builder is deterministic: the same input files give the same graph content, checked by a
  content hash over the sorted rows. Source file hashes are recorded in `meta`.
- LLM-backed runs pin the model id and record it in the result, so benchmark runs can be repeated.
