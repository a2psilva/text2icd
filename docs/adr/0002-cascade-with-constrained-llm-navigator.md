# ADR 0002: A deterministic cascade, with the LLM as a constrained navigator

- Status: Accepted
- Date: 2026-11-01

## Context

The tool must map diagnosis phrases to ICD-10-CM code sets accurately, at a very low cost per phrase,
and with an audit trail a coder can check. Three properties pull in different directions:

- **Accuracy** needs the classification's own logic: combination codes, "with" conventions,
  instructional notes, 7th characters and companion codes.
- **Cost** rules out sending every phrase to a large model.
- **Trust** requires that every returned code exists in the selected release and that the reasoning
  can be inspected.

Clinical phrases are also full of abbreviations and paraphrases ("HCV cirrhosis c/b ascites") that a
purely lexical method handles poorly.

## Decision

Resolve each phrase through a cascade that copies the coder's procedure:

1. **L0** result cache.
2. **L1** deterministic coder: Alphabetic Index navigation, then Tabular List verification.
3. **L2** local embedding retrieval when the Index walk is unsure.
4. **L3** an LLM, used only at low-confidence decision points, and only to **choose one option the
   graph offers** (a main term, a subterm, a Tabular child, an abbreviation sense) or `none`.

Tabular verification is shared by all layers: L2 and L3 can only propose, never write a code. A
calibrated confidence score decides when to escalate. Each phrase has a cap on LLM calls and each run
has a dollar budget; when either is reached, the best deterministic answer is returned and flagged
for review.

## Alternatives considered

- **End-to-end LLM coding**, with or without retrieved candidates. Simple to build, but it costs more
  per phrase, can return codes that do not exist, and is hard to audit. It is kept as an evaluation
  baseline.
- **A classifier trained on MIMIC-IV** (for example a transformer over the code set). Strong on
  frequent codes, weak on the long tail, needs retraining for every release, and its weights could not
  be distributed because they are derived from data under the PhysioNet data use agreement.
- **Deterministic only.** The cheapest option, but weaker on shorthand and paraphrase. It remains
  available as a mode (no LLM provider configured) and is the default.

## Consequences

- The invalid-code rate is zero by construction, and every result carries a trace of Index and
  Tabular steps with guideline references.
- Cost is dominated by the share of phrases that reach L3, which is measured on every benchmark run.
- LLM prompts are small and stable (rules, path, numbered options), which suits prompt caching and
  batch APIs.
- More engineering goes into the deterministic layers (parsers, graph, rules) than an end-to-end
  approach would need. That work is reusable across releases.
- The confidence model must be calibrated, or the cascade escalates too often (cost) or too rarely
  (accuracy).
