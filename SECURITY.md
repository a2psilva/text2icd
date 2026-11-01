# Security Policy

## Supported versions

text2icd is pre-release software. Security fixes go to the latest `main` only.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/a2psilva/text2icd/security/advisories/new).
Do not open a public issue.

Please include the affected version or commit, the steps to reproduce, and the impact. You should get
an acknowledgement within 7 days.

**Never include patient data, PHI or MIMIC-IV content in a report**, even to show impact. Use
synthetic phrases.

## In scope

- Leaks of input text to unintended destinations (logs, caches, LLM providers that are not
  configured or not approved).
- Ways to bypass the endpoint policy that blocks unapproved LLM providers.
- Injection through input phrases that changes the resolver's behaviour beyond choosing among the
  options the graph offers.
- Vulnerabilities in the HTTP service once it exists (TASKS.md Phase 10).
