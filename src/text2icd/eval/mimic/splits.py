"""Patient-level splits of the MIMIC-IV sets, and the frozen `test` split.

A patient's examples all go to the same split, chosen by a hash of `subject_id`, so the split of a
patient never changes when a set is rebuilt. Once frozen, the `test` examples of a dataset are
recorded by hash in `<eval dir>/frozen.json`; a rebuild that would change them is refused, and the
runner refuses a test run on a set that no longer matches.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from text2icd.eval.datasets import Example, Split, dataset_hash

SALT = "text2icd-mimic-split-v1"
TEST_SHARE = 0.5
FROZEN = "frozen.json"


class FrozenSplitError(RuntimeError):
    """The test split of a frozen dataset would change."""


def split_of_subject(subject_id: str) -> Split:
    digest = hashlib.sha256(f"{SALT}:{subject_id}".encode()).digest()
    return "test" if int.from_bytes(digest[:4], "big") / 2**32 < TEST_SHARE else "dev"


def test_hash(examples: Sequence[Example]) -> str:
    return dataset_hash([e for e in examples if e.split == "test"])


def frozen(eval_dir: Path) -> dict[str, str]:
    path = eval_dir / FROZEN
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def freeze(eval_dir: Path, name: str, examples: Sequence[Example]) -> str:
    """Record the test split of a dataset; returns its hash."""
    records = frozen(eval_dir)
    digest = test_hash(examples)
    if name in records and records[name] != digest:
        raise FrozenSplitError(f"the test split of {name} is already frozen with another content")
    records[name] = digest
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / FROZEN).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return digest


def check_frozen(eval_dir: Path, name: str, examples: Sequence[Example]) -> None:
    """Raise if the dataset is frozen and its test split differs from the frozen one."""
    expected = frozen(eval_dir).get(name)
    if expected is not None and test_hash(examples) != expected:
        raise FrozenSplitError(
            f"the test split of {name} differs from the one frozen in {eval_dir / FROZEN}"
        )
