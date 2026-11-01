# Architecture decision records

Each record captures one significant design decision: the context, the decision, the alternatives
that were considered and the consequences. Records are numbered and never renumbered. A decision
that is changed later gets a new record that supersedes the old one, and the old one's status is
updated to point to it.

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-sqlite-fts5-graph-store.md) | SQLite with FTS5 as the knowledge-graph store | Accepted |
| [0002](0002-cascade-with-constrained-llm-navigator.md) | A deterministic cascade, with the LLM as a constrained navigator | Accepted |
| [0003](0003-llm-providers-and-mimic-endpoint-policy.md) | LLM providers and the MIMIC-IV endpoint policy | Accepted |
| [0004](0004-release-naming.md) | Naming ICD-10-CM releases and carrying them through artifacts | Accepted |

## Writing a new record

Copy the template below to `NNNN-short-title.md` with the next free number, fill it in, add it to the
table above, and open a pull request. Statuses: `Proposed`, `Accepted`, `Superseded by NNNN`,
`Deprecated`.

```markdown
# ADR NNNN: Title

- Status: Proposed
- Date: YYYY-MM-DD

## Context

What problem or force makes a decision necessary? What constraints apply?

## Decision

What we will do, stated plainly.

## Alternatives considered

Each option that was seriously considered, and why it was not chosen.

## Consequences

What becomes easier, what becomes harder, and what follow-up work this creates.
```
