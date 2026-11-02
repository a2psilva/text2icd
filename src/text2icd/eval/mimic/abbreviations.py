"""Local abbreviation mining: which tokens of MIMIC-IV discharge diagnoses neither the lexicon nor
the CMS files know, and how the lexicon is used.

Only the discharge diagnosis items of dev-split patients are read; the test split is not looked
at. Each item is normalised with the lexicon, and every word of the result that is not a word of
the CMS files is counted, with the notes it appears in. The counts are written to
`data/interim/abbreviation-candidates.tsv`, which is git-ignored and never leaves the machine:
it shows which entries to write next, and the entries themselves are written by hand from
medical knowledge (`resources/abbreviations.toml`).

The summary is aggregate: counts, lexicon keys (which are not MIMIC-IV text), and only the
tokens used in at least `min_notes` notes, which are vocabulary rather than note content.
"""

import csv
import re
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from text2icd.eval.mimic.files import Note
from text2icd.eval.mimic.sections import diagnosis_items
from text2icd.eval.mimic.splits import split_of_subject
from text2icd.text.abbreviations import Lexicon
from text2icd.text.normalize import normalize
from text2icd.text.terms import words

CANDIDATES = "abbreviation-candidates.tsv"
MIN_NOTES = 50  # a token is shown only when this many notes use it
FILE_MIN_NOTES = 2  # tokens used in fewer notes are left out of the local file


@dataclass
class Mined:
    notes: int = 0  # dev notes with discharge diagnosis items
    items: int = 0
    with_expansion: int = 0  # items with at least one expansion
    with_unknown: int = 0  # items with a word the CMS files and the lexicon do not know
    keys: Counter[str] = field(default_factory=Counter)  # lexicon key -> expansions
    key_notes: Counter[str] = field(default_factory=Counter)
    chosen: Counter[tuple[str, str]] = field(default_factory=Counter)  # (key, chosen_by)
    token_notes: Counter[str] = field(default_factory=Counter)
    token_items: Counter[str] = field(default_factory=Counter)
    token_capitals: Counter[str] = field(default_factory=Counter)  # items writing it in capitals

    def add(self, texts: Iterable[str], lexicon: Lexicon, known: Collection[str]) -> None:
        """Count the items of one note."""
        self.notes += 1
        keys: set[str] = set()
        tokens: set[str] = set()
        for text in texts:
            self.items += 1
            result = normalize(text, lexicon, known)
            if result.expansions:
                self.with_expansion += 1
            for expansion in result.expansions:
                self.keys[expansion.key] += 1
                self.chosen[(expansion.key, expansion.chosen_by)] += 1
                keys.add(expansion.key)
            unknown = {word for word in words(result.text) if _candidate(word, known)}
            if unknown:
                self.with_unknown += 1
            for word in unknown:
                self.token_items[word] += 1
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(word.upper())}(?![A-Za-z0-9])", text):
                    self.token_capitals[word] += 1
            tokens |= unknown
        self.key_notes.update(sorted(keys))
        self.token_notes.update(sorted(tokens))

    def lines(self, min_notes: int = MIN_NOTES, top: int = 60) -> list[str]:
        def share(n: int, total: int) -> str:
            return f"{n / total:.1%}" if total else "n/a"

        ambiguous = sorted(
            {key for key, how in self.chosen if how != "only"},
            key=lambda key: -self.keys[key],
        )
        lines = [
            f"dev notes {self.notes:,}; items {self.items:,}",
            f"items with an expansion {self.with_expansion:,} "
            f"({share(self.with_expansion, self.items)}); expansions {self.keys.total():,}",
            f"items with a word unknown to the CMS files and the lexicon {self.with_unknown:,} "
            f"({share(self.with_unknown, self.items)}); distinct words {len(self.token_notes):,}",
            "most used lexicon keys (notes): "
            + ", ".join(f"{key} {n:,}" for key, n in _most_used(self.key_notes, 25)),
            "ambiguous keys (by hint / by default): "
            + ", ".join(
                f"{key} {self.chosen[(key, 'hint')]:,}/{self.chosen[(key, 'default')]:,}"
                for key in ambiguous
            ),
        ]
        shown = [(t, n) for t, n in _most_used(self.token_notes) if n >= min_notes][:top]
        lines.append(f"unknown words used in {min_notes}+ notes (notes, share in capitals):")
        lines += [
            f"  {token:<16} {n:>7,}  {share(self.token_capitals[token], self.token_items[token])}"
            for token, n in shown
        ]
        return lines


def _most_used(counts: Counter[str], top: int | None = None) -> list[tuple[str, int]]:
    """Most used first, ties in alphabetical order, so that summaries do not vary between runs."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered if top is None else ordered[:top]


def _candidate(word: str, known: Collection[str]) -> bool:
    return len(word) > 1 and not word.isdigit() and not _known(word, known)


def _known(word: str, known: Collection[str]) -> bool:
    """A word of the CMS files, or a plural of one (`effusions`, `metastases`, `allergies`)."""
    if word in known:
        return True
    singulars = []
    if word.endswith("s"):
        singulars.append(word[:-1])
    if word.endswith("es"):
        singulars.append(word[:-2])
    if word.endswith("ses"):
        singulars.append(word[:-3] + "sis")
    if word.endswith("ies"):
        singulars.append(word[:-3] + "y")
    return any(singular in known for singular in singulars)


def mine(
    notes: Iterable[Note],
    lexicon: Lexicon,
    known: Collection[str],
    *,
    limit: int | None = None,
) -> Mined:
    """Mine the discharge diagnosis items of dev-split patients (`limit`: dev notes read)."""
    mined = Mined()
    known = frozenset(known) | expansion_words(lexicon)
    for note in notes:
        if split_of_subject(note.subject_id) != "dev":
            continue
        items = diagnosis_items(note.text)
        if not items:
            continue
        mined.add((item.text for item in items), lexicon, known)
        if limit is not None and mined.notes >= limit:
            break
    return mined


def expansion_words(lexicon: Lexicon) -> frozenset[str]:
    """The words the lexicon itself writes, some of which the CMS files lack (procedures)."""
    return frozenset(
        word
        for entry in lexicon.entries.values()
        for expansion in entry.expansions
        for word in words(expansion)
    )


def write_candidates(path: Path, mined: Mined, min_notes: int = FILE_MIN_NOTES) -> int:
    """The unknown words used in `min_notes` or more notes, most used first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(t, n) for t, n in _most_used(mined.token_notes) if n >= min_notes]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["token", "notes", "items", "items_in_capitals"])
        for token, notes in rows:
            writer.writerow([token, notes, mined.token_items[token], mined.token_capitals[token]])
    return len(rows)
