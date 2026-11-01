"""Locate the CMS files of a release unpacked by `text2icd data fetch`."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReleaseFiles:
    """Paths to the files of one release under `data/raw/cms/<release>/`."""

    root: Path

    @property
    def tabular(self) -> Path:
        return self._one("tables-index", "icd10cm_tabular_*.xml")

    @property
    def index(self) -> Path:
        return self._one("tables-index", "icd10cm_index_*.xml")

    @property
    def eindex(self) -> Path:
        return self._one("tables-index", "icd10cm_eindex_*.xml")

    @property
    def neoplasm(self) -> Path:
        return self._one("tables-index", "icd10cm_neoplasm_*.xml")

    @property
    def drug(self) -> Path:
        return self._one("tables-index", "icd10cm_drug_*.xml")

    @property
    def order(self) -> Path:
        return self._one("order", "icd10cm_order_[0-9][0-9][0-9][0-9].txt")

    def _one(self, kind: str, pattern: str) -> Path:
        matches = sorted((self.root / kind).glob(pattern))
        if len(matches) != 1:
            found = ", ".join(p.name for p in matches) or "none"
            raise FileNotFoundError(
                f"expected one {pattern} in {self.root / kind} (found {found}); "
                f"run `text2icd data fetch -r {self.root.name}`"
            )
        return matches[0]
