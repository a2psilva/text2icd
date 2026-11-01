# Test fixtures

Small, hand-made inputs for unit tests, for example a trimmed `<chapter>` of the Tabular List or a
single `<mainTerm>` of the Alphabetic Index.

Rules:

- **CMS content only.** The ICD-10-CM files published by CMS are US government works in the public
  domain, so excerpts of them may be committed here.
- **Never add MIMIC-IV content**, not even a single line or a "de-identified" phrase taken from a note.
  The PhysioNet data use agreement forbids redistribution. Write synthetic phrases by hand instead.
- Keep each fixture as small as the test allows (a few KB). The pre-commit hook rejects files over 500 KB.
