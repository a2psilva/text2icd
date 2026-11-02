"""`mimic-silver`: discharge diagnosis items labelled with their admission's billed codes.

For every discharge summary whose admission has ICD-10 codes:

1. its discharge diagnosis items (`sections`);
2. the admission's billed ICD-10 codes (`gold`), each with its status against the target release
   (`versions`);
3. one-to-one alignment of items to codes by embedding similarity, kept above a score and a
   margin (`align`);
4. the patient's split (`splits`).

The set is written under the git-ignored data directory and never leaves it. The statistics are
aggregate counts.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from text2icd.embeddings import Embedder, Vectors
from text2icd.eval.datasets import Example, GoldCode
from text2icd.eval.mimic.align import MIN_MARGIN, MIN_SCORE, assign
from text2icd.eval.mimic.files import Note
from text2icd.eval.mimic.gold import BilledCode
from text2icd.eval.mimic.sections import DiagnosisItem, diagnosis_items
from text2icd.eval.mimic.splits import split_of_subject
from text2icd.eval.versions import Reconciler

SOURCE = "mimic-silver"
CHUNK = 5000
SCORE_BINS = tuple(round(0.5 + 0.05 * i, 2) for i in range(11))  # 0.50, 0.55, ..., 1.00


@dataclass
class SilverStats:
    """Aggregate counts of a build."""

    notes: int = 0
    with_codes: int = 0
    with_items: int = 0
    items: int = 0
    aligned: int = 0
    codes: int = 0
    no_description: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    groups: Counter[str] = field(default_factory=Counter)
    splits: Counter[str] = field(default_factory=Counter)
    best_scores: Counter[float] = field(default_factory=Counter)  # per item, by lower bin edge
    first_code: int = 0  # aligned items whose code is the admission's first (seq_num 1)

    def lines(self) -> list[str]:
        def share(n: int, total: int) -> str:
            return f"{n / total:.1%}" if total else "n/a"

        bins = ", ".join(f">={edge:.2f}: {self.best_scores[edge]:,}" for edge in SCORE_BINS)
        groups = ", ".join(f"{k} {share(v, self.aligned)}" for k, v in sorted(self.groups.items()))
        statuses = ", ".join(f"{k} {v:,}" for k, v in sorted(self.statuses.items()))
        return [
            f"notes {self.notes:,}; with ICD-10 codes {self.with_codes:,}; "
            f"with items too {self.with_items:,}",
            f"items {self.items:,}; aligned {self.aligned:,} ({share(self.aligned, self.items)}); "
            f"codes {self.codes:,} ({self.no_description:,} without a description)",
            f"aligned items whose code is the admission's first: "
            f"{share(self.first_code, self.aligned)}",
            f"groups of aligned items: {groups or 'none'}",
            f"gold statuses: {statuses or 'none'}",
            f"splits: dev {self.splits['dev']:,}, test {self.splits['test']:,}",
            f"best similarity per item: <0.50: {self.best_scores[0.0]:,}, {bins}",
        ]


@dataclass(frozen=True, slots=True)
class _Pending:
    note: Note
    items: list[DiagnosisItem]
    codes: list[BilledCode]


def build_silver(
    notes: Iterable[Note],
    codes_by_admission: Mapping[str, Sequence[BilledCode]],
    reconciler: Reconciler,
    embedder: Embedder,
    *,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    chunk: int = CHUNK,
) -> tuple[list[Example], SilverStats]:
    stats = SilverStats()
    examples: list[Example] = []
    vectors: dict[str, Vectors] = {}
    pending: list[_Pending] = []

    def flush() -> None:
        examples.extend(
            _align_chunk(pending, reconciler, embedder, vectors, stats, min_score, min_margin)
        )
        pending.clear()

    for note in notes:
        stats.notes += 1
        codes = codes_by_admission.get(note.hadm_id)
        if not codes:
            continue
        stats.with_codes += 1
        items = diagnosis_items(note.text)
        if not items:
            continue
        stats.with_items += 1
        stats.items += len(items)
        pending.append(_Pending(note, items, list(codes)))
        if len(pending) >= chunk:
            flush()
    flush()
    return examples, stats


def _align_chunk(
    pending: Sequence[_Pending],
    reconciler: Reconciler,
    embedder: Embedder,
    code_vectors: dict[str, Vectors],
    stats: SilverStats,
    min_score: float,
    min_margin: float,
) -> Iterator[Example]:
    if not pending:
        return
    texts = sorted({item.text for entry in pending for item in entry.items})
    item_vectors = dict(zip(texts, embedder.encode(texts), strict=True))
    new_codes = sorted(
        {
            billed.code
            for entry in pending
            for billed in entry.codes
            if billed.code not in code_vectors and billed.code in reconciler.descriptions
        }
    )
    if new_codes:
        encoded = embedder.encode([reconciler.descriptions[code] for code in new_codes])
        code_vectors.update(zip(new_codes, encoded, strict=True))
    for entry in pending:
        described = [b for b in entry.codes if b.code in code_vectors]
        stats.codes += len(entry.codes)
        stats.no_description += len(entry.codes) - len(described)
        if not described:
            continue
        items = np.stack([item_vectors[item.text] for item in entry.items])
        codes = np.stack([code_vectors[b.code] for b in described])
        similarity = items @ codes.T
        for best in similarity.max(axis=1):
            stats.best_scores[_bin(float(best))] += 1
        for alignment in assign(similarity, min_score, min_margin):
            item = entry.items[alignment.item]
            billed = described[alignment.code]
            result = reconciler.reconcile(billed.code)
            split = split_of_subject(entry.note.subject_id)
            stats.aligned += 1
            stats.statuses[result.status] += 1
            stats.groups[item.group or "none"] += 1
            stats.splits[split] += 1
            stats.first_code += billed.seq_num == entry.codes[0].seq_num
            yield Example(
                id=f"{entry.note.note_id}:{item.position}",
                phrase=item.text,
                gold=(
                    GoldCode(
                        code=billed.code,
                        role="primary",
                        status=result.status,
                        mapped_to=result.mapped_to,
                    ),
                ),
                release=reconciler.target,
                source=SOURCE,
                split=split,
                meta={
                    "note_id": entry.note.note_id,
                    "subject_id": entry.note.subject_id,
                    "hadm_id": entry.note.hadm_id,
                    "position": item.position,
                    "group": item.group,
                    "seq_num": billed.seq_num,
                    "score": round(alignment.score, 4),
                    "margin": round(alignment.margin, 4),
                    "admission_codes": len(entry.codes),
                },
            )


def _bin(score: float) -> float:
    if score < SCORE_BINS[0]:
        return 0.0
    return max(edge for edge in SCORE_BINS if score >= edge)
