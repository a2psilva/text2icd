# Annotation guide: `mimic-gold`

`mimic-gold` is the hand-coded reference set: 300 to 500 diagnosis phrases from MIMIC-IV discharge
summaries, each with the full ICD-10-CM code set a coder would assign to the phrase. Results on it
are the ones reported, so the labels should be as careful as a coder's work. Ideally a certified
coder reviews them.

The phrases come from MIMIC-IV. Everything in this task stays on your machine, under `data/`: never
copy a phrase into an issue, a document, a chat assistant or a screenshot
([data and compliance](data-and-compliance.md)). You need your own PhysioNet credentialing.

## The task

You see one phrase at a time: one item of a discharge summary's "Discharge Diagnosis" list. Below it
are the ICD-10-CM codes billed for that admission, with their descriptions. They are there to help:
the phrase usually names one of them, but not always.

Code **the phrase alone**, as if it were the only documentation, against the release being
evaluated (FY2027). Type the codes with the primary code first, then any companion codes.

| You type | Meaning |
| --- | --- |
| `2` or `2 5` | candidate numbers |
| `E11.22 N18.30` | codes, with or without the dot; mixing numbers and codes is fine |
| `s` *comment* | skip: not a codable diagnosis (a procedure, a plan, a fragment) |
| `?` *comment* | unsure: the phrase could mean different things |
| `l` *words* | look up Alphabetic Index terms |
| `c` *code* | show a code: where it sits, its notes and 7th characters |
| `b`, `q`, `h` | back, save and quit, help |

Every code is checked against the billable codes of the release when you type it. Answers are saved
after each phrase, so you can stop at any time and resume later.

## Rules

1. **The phrase decides.** Use the most specific code the phrase supports, and no more. If the
   admission was billed `N18.30` but the phrase says only "CKD", the code is `N18.9`. Do not use
   what you know about the rest of the record.
2. **Follow the coder's workflow**: Alphabetic Index first (`l`), then verify in the Tabular List
   (`c`), then apply the instructional notes. The billed codes are suggestions, not answers.
3. **Unspecified codes are correct** when the phrase gives no more detail ("pneumonia" is `J18.9`).
4. **Combination codes.** When the phrase links conditions that the classification combines, use
   the combination code ("type 2 diabetes with diabetic CKD" is `E11.22`). The "with" convention
   (Guideline I.A.15) applies when the phrase itself says "with" or lists the conditions together.
5. **Companion codes** go after the primary code only when an instructional note requires them
   *and* the phrase gives the detail: `E11.22 N18.30` for "DM2 with CKD stage 3a", but `E11.22` alone
   for "diabetic nephropathy" with no stage. Manifestation codes follow their etiology code.
6. **Uncertain diagnoses** ("probable", "suspected", "likely", "rule out") are coded as if
   established: discharge summaries are inpatient records (Guideline Section II.H).
7. **History and status.** "History of" a condition that no longer exists takes the personal
   history code (Z85-Z87); "s/p" a procedure takes the status code when one exists (Z95.1 for
   "s/p CABG"); a condition still present is coded as the condition.
8. **Injuries and poisonings** need a 7th character: use `A` (initial encounter) unless the phrase
   says subsequent care or sequela, since these are diagnoses of the stay being discharged.
9. **Abbreviations.** Expand them as a coder would in a discharge summary. If one could mean
   different things and the phrase does not settle it ("MS"), mark it `?`.
10. **Several diagnoses in one phrase** ("HTN, HLD"): code them all, the first one named first.
11. **Negated or non-diagnoses** ("no evidence of PE", "discharge planning"): skip with `s`.

## Steps

```bash
uv run text2icd eval mimic gold sample          # draw the candidates from mimic-silver (once)
uv run text2icd eval mimic gold annotate        # code them; run again to resume
uv run text2icd eval mimic gold build           # write mimic-gold.jsonl from the answers
uv run text2icd eval mimic freeze -d mimic-gold # freeze its test split once it is complete
```

The sample takes a third of its phrases from each frequency band of the silver codes (the codes
that make up the first half of the silver examples, the next 40%, and the last 10%), so rare codes
are represented, and spreads each band across chapters. It never takes two items from one note.

Plan about half a minute per phrase: 500 phrases take four to five hours, best in sessions of
about an hour. `build` reports how many were coded, skipped and unsure, and how often your primary
code agrees with the automatic silver label: an aggregate check of the silver labels.
