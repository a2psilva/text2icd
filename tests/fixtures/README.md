# Test fixtures

Small, hand-made inputs for unit tests, for example a trimmed `<chapter>` of the Tabular List or a
single `<mainTerm>` of the Alphabetic Index.

Rules:

- **CMS content only.** The ICD-10-CM files published by CMS are US government works in the public
  domain, so excerpts of them may be committed here.
- **Never add MIMIC-IV content**, not even a single line or a "de-identified" phrase taken from a note.
  The PhysioNet data use agreement forbids redistribution. Write synthetic phrases by hand instead.
- Keep each fixture as small as the test allows (a few KB). The pre-commit hook rejects files over 500 KB.

## Files

| File | Origin |
| --- | --- |
| `cms/tabular_sample.xml` | FY2027 Tabular List, trimmed to E08.32 (7th characters), E11.22 (use additional code) and T51.3 (placeholders), with their section and chapter notes. Element content is unchanged. |
| `cms/index_sample.xml` | FY2027 Index to Diseases and Injuries: Abandonment (see), Abdomen (see also), Amyloid heart (manifestation), Bonnier's syndrome (subcat), Pyelitis (seecat), and Pneumonia trimmed to its `with`, `aspiration` and `adenoviral` subterms. |
| `cms/eindex_sample.xml` | FY2027 External Cause Index: Abandonment (text after the nonessential modifiers) and Abuse. |
| `cms/neoplasm_sample.xml` | FY2027 Neoplasm Table: column headings, the main term and the `abdomen, abdominal` rows. |
| `cms/drug_sample.xml` | FY2027 Table of Drugs and Chemicals: column headings and four substances. |
| `cms/order_sample.txt` | Ten lines of the FY2027 code order file (A00, E11.22, the T51.3 chain down to the billable T51.3X1A), renumbered 1 to 10. |
