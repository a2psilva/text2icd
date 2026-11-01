# ADR 0003: LLM providers and the MIMIC-IV endpoint policy

- Status: Accepted
- Date: 2026-11-01

## Context

The L3 navigator (ADR 0002) sends short prompts containing the input phrase to a hosted LLM. During
evaluation, those phrases come from MIMIC-IV-Note, which is covered by the PhysioNet Credentialed
Health Data Use Agreement. PhysioNet's guidance
["Responsible use of MIMIC data with online services like GPT"](https://physionet.org/news/post/gpt-responsible-use)
(2023-04-18) says the agreement forbids sharing the data with third parties, including through
general-purpose APIs, and names a short list of hosted services that are acceptable when configured
not to retain the data, not to use it for training, and not to subject it to human review. The
guidance may change.

Outside MIMIC-IV, users will run text2icd on their own data, with their own provider and their own
compliance rules. Prices and model line-ups also change often.

## Decision

- **No LLM by default.** `TEXT2ICD_LLM_PROVIDER=none` runs the deterministic cascade only.
- **One provider interface.** The navigator talks to LLMs only through an interface with one
  implementation per hosted service. The model id is configuration, pinned per run and recorded in
  every result.
- **An endpoint allow-list for restricted data.** With `TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT=true`,
  only endpoints on an explicit allow-list in code can be used, each entry citing the basis for its
  approval. MIMIC-IV evaluation refuses to start unless the flag is on. The list is reviewed whenever
  PhysioNet updates its guidance.
- **No input text in logs** unless `log_inputs=true` is set explicitly. Result and decision caches
  live in git-ignored directories.
- **Cost is accounted per call** from the provider's usage fields, against a price table kept in one
  configuration file.

## Alternatives considered

- **Never use an LLM.** Simplest for compliance, but gives up accuracy on shorthand and paraphrase.
  It remains the default mode.
- **A local open-weights model only.** Avoids data sharing entirely, but needs local hardware and
  more operational work, and small local models are less reliable at constrained choices. It can be
  added later as another provider behind the same interface.
- **One hard-wired provider.** Less code, but locks users into one vendor's prices and terms and
  makes the compliance rules impossible to adapt per deployment.

## Consequences

- Each new provider needs an adapter and, for restricted data, a documented allow-list entry.
- Documentation and defaults stay vendor-neutral; the choice of service and model is made per
  deployment and recorded in results.
- Anyone running MIMIC-IV experiments must re-check the PhysioNet guidance and their own data use
  agreement before enabling the navigator.
