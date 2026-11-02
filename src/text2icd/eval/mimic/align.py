"""The silver-label aligner: each discharge diagnosis item matched to one of its admission's
billed codes, by the similarity of the item to the code's description.

It is independent of the text2icd resolver on purpose (docs/evaluation.md): labels made by the
resolver would hide its own mistakes. It uses an embedding model that no evaluated system uses
(SapBERT by default, trained to match biomedical synonyms). Within an admission, item-code pairs
are taken best first, each item and each code used once; a pair is kept only when its
similarity is high enough and clearly above the item's next best code.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_MODEL = "Xenova/SapBERT-from-PubMedBERT-fulltext"
MIN_SCORE = 0.70
MIN_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class Alignment:
    item: int  # row in the similarity matrix
    code: int  # column
    score: float
    margin: float  # over the item's best other code (1.0 when the admission has one code)


def assign(
    similarity: npt.NDArray[np.float32],
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> list[Alignment]:
    """Greedy one-to-one assignment of items (rows) to codes (columns), best pair first, keeping
    pairs above both thresholds. The margin compares the item's code with its best other code,
    assigned or not, so an item close to two codes is not trusted."""
    rows, cols = similarity.shape
    if rows == 0 or cols == 0:
        return []
    order = np.argsort(-similarity, axis=None, kind="stable")
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    found = []
    for flat in order:
        i, j = divmod(int(flat), cols)
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        score = float(similarity[i, j])
        others = np.delete(similarity[i], j)
        margin = score - float(others.max()) if others.size else 1.0
        if score >= min_score and margin >= min_margin:
            found.append(Alignment(i, j, score, margin))
        if len(used_rows) == rows or len(used_cols) == cols:
            break
    return sorted(found, key=lambda a: a.item)
