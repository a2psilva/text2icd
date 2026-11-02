# How a medical coder works, and how text2icd copies it

text2icd does not guess codes from similar-looking descriptions. It follows the procedure a certified
coder follows, as set out in the *ICD-10-CM Official Guidelines for Coding and Reporting* (FY2026).
This page maps each step of that procedure to the component that performs it.

The core rule is Guideline **I.B.1, Locating a code**:

> "first locate the term in the Alphabetic Index, and then verify the code in the Tabular List. Read
> and be guided by instructional notations that appear in both the Alphabetic Index and the Tabular
> List. [...] The Alphabetic Index does not always provide the full code. Selection of the full code,
> including laterality and any applicable 7th character can only be done in the Tabular List."

## Step-by-step mapping

| # | What the coder does | Guideline | text2icd component |
| --- | --- | --- | --- |
| 1 | Read the diagnostic statement; separate the reportable conditions; drop ruled-out ones; note "history of" and uncertainty | I.B.4–I.B.6, II.H, III.C, IV.H | `text`: splitter, negation, history and uncertainty flags |
| 2 | Expand abbreviations and resolve shorthand ("CKD", "c/b", "s/p") | I.A.6 (Index/Tabular abbreviations NEC, NOS) | `text`: abbreviation lexicon; L3 for unknown ones |
| 3 | Pick the right book: Neoplasm Table, Table of Drugs and Chemicals, External Cause Index, or the main Index | I.A.1, I.C.2, I.C.19 | `resolve`: index router |
| 4 | Find the **main term**, which is the condition, not the anatomical site ("Fracture, femur", not "Femur") | Index convention | `resolve`: main-term selection (BM25 + fuzzy + lexicon) |
| 5 | Treat nonessential modifiers in parentheses as optional; descend through essential modifiers (indented subterms). When a nonessential modifier and a subentry are mutually exclusive, the subentry wins | I.A.7 | `graph`: `nemod` stored apart from the title; `resolve`: subterm beam search |
| 6 | Follow cross-references: "see", "see also", "see category", "code also" | I.A.16, I.A.17 | `graph`: `see` / `see_also` edges resolved to Index paths |
| 7 | Apply the "with" / "in" convention: linked conditions are presumed related | I.A.15 | `resolve`: connector subterms ("with", "due to", "in") consume linked conditions |
| 8 | If nothing more specific is documented, take the code next to the main term | I.A.18 (Default codes) | `resolve`: default-code fallback, flagged in the trace |
| 9 | Go to the Tabular List; never code from the Index alone. A dash ("-") means more characters are needed | I.B.1 | `resolve`: Tabular verifier |
| 10 | Read the instructional notes: includes, inclusion terms, Excludes1, Excludes2 | I.A.10–I.A.12 | `resolve`: note checks, Excludes1 conflict detection across the code set |
| 11 | Code to the highest level of specificity; use "unspecified" only when the record gives no more detail | I.A.9, I.B.2, I.B.18 | `resolve`: descent to a billable code; "unspecified" choice is flagged |
| 12 | Add the 7th character and "X" placeholders where required | I.A.4, I.A.5 | `ontology`: inherited `sevenChrDef`; `resolve`: placeholder padding |
| 13 | Add paired codes: etiology first, manifestation second; "use additional code" and "code first" | I.A.13 | `graph`: `manif`, `use_additional`, `code_first` edges; `resolve`: companion codes |
| 14 | Prefer combination codes over separate codes when one exists | I.B.9 | `resolve`: Index connector subterms and Tabular combination rules |
| 15 | Apply chapter-specific rules (hypertension with CKD or heart disease, diabetes, sepsis, injuries, sequelae) | I.B.10, I.C.* | `resolve`: guideline rule set, each rule citing its section |

## Worked examples from the FY2026 files

The Index part of each example is checked against the FY2027 files by the test suite
(`tests/resolve/test_index_cases.py`), with 110 more Index-walk cases drawn from the Index.

**Default code (I.A.18).** "Appendicitis" with no further detail. Index: `Appendicitis (pneumococcal)
(retrocecal)` lists `K37`. Tabular: K37 is *Unspecified appendicitis* and is billable. Result: K37,
flagged `default_code_used`.

**Subterm descent.** "aspiration pneumonia due to food". Index: `Pneumonia > aspiration > due to >
food (regurgitated)` gives `J69.0`. Tabular: J69.0 *Pneumonitis due to inhalation of food and vomit*.

**Combination code and "use additional code".** "type 2 DM with diabetic CKD stage 3a". Index:
`Diabetes, diabetic > type 2 > with > chronic kidney disease` gives `E11.22`. Tabular note on E11.22:
*Use additional code to identify stage of chronic kidney disease (N18.1-N18.6)*. A second Index walk,
`Disease, diseased > kidney > chronic > stage 3a`, gives `N18.31`. Result: E11.22 (primary) and N18.31
(additional).

**Etiology and manifestation (I.A.13).** "amyloid heart disease". Index: `Amyloid heart (disease)`
gives `E85.4` with manifestation `[I43]`. Result: E85.4 *Organ-limited amyloidosis* first, then I43
*Cardiomyopathy in diseases classified elsewhere*.

**7th character (I.A.5).** "mild nonproliferative diabetic retinopathy with macular edema due to
underlying condition, right eye". Tabular: E08.321 carries a `sevenChrDef` inherited from E08.32 (1 =
right eye). E08.3211 is not listed in the XML at all; it exists only through that definition, which is
why the billable code set has to be generated.

**Placeholder and 7th character (I.A.4, I.A.5).** Accidental poisoning by 1-propanol. Table of Drugs
and Chemicals: row `1-propanol`, column *Poisoning, Accidental* gives `T51.3X1`. Tabular: category T51
defines A, D and S. For an initial encounter the code is `T51.3X1A`.

## What text2icd does not do

- It does not read full clinical notes. Input is a diagnosis phrase (extraction from notes is a
  possible later extension).
- It does not sequence the principal diagnosis for an encounter (Section II). Within a code set it
  only orders etiology before manifestation, as I.A.13 requires.
- It does not replace query-the-physician. When the documentation is not specific enough, it returns
  the "unspecified" code and says so in `flags`.
