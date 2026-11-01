"""The registry of ICD-10-CM releases: which CMS files make up each release, and their hashes."""

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Literal, cast, get_args

from text2icd.config import RELEASE_ID

FileKind = Literal[
    "tables-index", "order", "addendum", "conversion-table", "poa-exempt", "guidelines"
]
Scope = Literal["full", "history"]

FILE_KINDS: tuple[str, ...] = get_args(FileKind)
REQUIRED_FOR_FULL: frozenset[str] = frozenset({"tables-index", "order"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CMS_URL = re.compile(r"^https://www\.cms\.gov/\S+$")


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    kind: FileKind
    url: str
    sha256: str
    size: int

    @property
    def archive_name(self) -> str:
        """Local file name of the download, unique within a release."""
        return f"{self.kind}.pdf" if self.url.lower().endswith(".pdf") else f"{self.kind}.zip"


@dataclass(frozen=True, slots=True)
class Release:
    id: str
    fiscal_year: int
    effective: date
    scope: Scope
    files: tuple[ReleaseFile, ...]

    def file(self, kind: FileKind) -> ReleaseFile:
        for f in self.files:
            if f.kind == kind:
                return f
        raise KeyError(f"release {self.id} has no {kind!r} file")


@dataclass(frozen=True, slots=True)
class Registry:
    default: str
    releases: dict[str, Release]

    def get(self, release_id: str | None = None) -> Release:
        """Return a release by id; `None` returns the default (newest full) release."""
        key = self.default if release_id is None else release_id
        try:
            return self.releases[key]
        except KeyError:
            known = ", ".join(self.releases)
            raise KeyError(f"unknown release {key!r}; known releases: {known}") from None

    def by_scope(self, scope: Scope) -> list[Release]:
        return [r for r in self.releases.values() if r.scope == scope]


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate the registry. Defaults to the copy shipped in `text2icd.resources`."""
    if path is None:
        raw = resources.files("text2icd.resources").joinpath("releases.toml").read_bytes()
    else:
        raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    if data.get("schema") != 1:
        raise ValueError(f"unsupported registry schema {data.get('schema')!r}")

    releases: dict[str, Release] = {}
    for entry in data.get("release", []):
        release = _parse_release(entry)
        if release.id in releases:
            raise ValueError(f"duplicate release id {release.id!r}")
        releases[release.id] = release

    default = data.get("default")
    if default not in releases:
        raise ValueError(f"default release {default!r} is not defined")
    if releases[default].scope != "full":
        raise ValueError(f"default release {default!r} must have scope 'full'")
    return Registry(default=default, releases=releases)


def _parse_release(entry: dict[str, object]) -> Release:
    release_id = entry.get("id")
    if not isinstance(release_id, str) or not RELEASE_ID.fullmatch(release_id):
        raise ValueError(f"invalid release id {release_id!r}")
    fiscal_year = entry.get("fiscal_year")
    if fiscal_year != int(release_id[2:6]):
        raise ValueError(f"{release_id}: fiscal_year {fiscal_year!r} does not match the id")
    effective = entry.get("effective")
    if not isinstance(effective, date):
        raise ValueError(f"{release_id}: effective must be a TOML date")
    scope = entry.get("scope")
    if scope not in get_args(Scope):
        raise ValueError(f"{release_id}: invalid scope {scope!r}")

    raw_files = entry.get("file", [])
    if not isinstance(raw_files, list):
        raise ValueError(f"{release_id}: 'file' must be an array of tables")
    files = tuple(_parse_file(release_id, f) for f in raw_files)
    kinds = [f.kind for f in files]
    if len(kinds) != len(set(kinds)):
        raise ValueError(f"{release_id}: each file kind may appear only once")
    if scope == "full" and not REQUIRED_FOR_FULL.issubset(kinds):
        raise ValueError(f"{release_id}: a full release needs {sorted(REQUIRED_FOR_FULL)}")
    return Release(release_id, int(release_id[2:6]), effective, cast(Scope, scope), files)


def _parse_file(release_id: str, entry: object) -> ReleaseFile:
    if not isinstance(entry, dict):
        raise ValueError(f"{release_id}: file entries must be tables")
    kind, url, sha, size = (entry.get(k) for k in ("kind", "url", "sha256", "size"))
    if kind not in FILE_KINDS:
        raise ValueError(f"{release_id}: unknown file kind {kind!r}")
    if not isinstance(url, str) or not CMS_URL.fullmatch(url):
        raise ValueError(f"{release_id} {kind}: url must be an https://www.cms.gov/ address")
    if not isinstance(sha, str) or not SHA256.fullmatch(sha):
        raise ValueError(f"{release_id} {kind}: sha256 must be 64 lowercase hex characters")
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"{release_id} {kind}: size must be a positive integer")
    return ReleaseFile(kind=cast(FileKind, kind), url=url, sha256=sha, size=size)
