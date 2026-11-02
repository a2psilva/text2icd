"""Text understanding for diagnosis phrases.

`text2icd.text.parse.Parser` (or `parse`) reads a phrase into a `ParsedPhrase`: normalisation and
abbreviation expansion (`normalize`, `abbreviations`), modifiers (`modifiers`), status, negation
and uncertainty cues (`cues`), and the split into conditions and relations (`split`), with the
Index terms that help it (`terms`). Every step is recorded in the trace.
"""
