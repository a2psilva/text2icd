"""Download, verify and unpack the CMS files of a release into `data/raw/cms/<release>/`.

Layout per release: `archives/<kind>.zip|pdf` holds the downloads exactly as published, and
`<kind>/` holds their extracted contents (a single top-level folder in a zip is dropped). Nothing is
ever overwritten: an existing file with different content is an error.
"""

import hashlib
import stat
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO

from text2icd import __version__
from text2icd.releases import Release, ReleaseFile

Opener = Callable[[urllib.request.Request, float], IO[bytes]]

USER_AGENT = f"text2icd/{__version__} (+https://github.com/a2psilva/text2icd)"
CHUNK = 1 << 20
MAX_MEMBERS = 1_000
MAX_UNCOMPRESSED_BYTES = 1 << 30
COMPLETE_MARKER = ".complete"
# The files `build-graph` and `data stats` read: the Tabular List and Index XML, and the order file.
GRAPH_FILES = ("tables-index", "order")


class FetchError(Exception):
    """A download, verification or extraction step failed."""


@dataclass(frozen=True, slots=True)
class FileResult:
    kind: str
    downloaded: bool
    extracted: int


def fetch_release(
    release: Release,
    raw_dir: Path,
    *,
    opener: Opener | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 3,
    timeout: float = 120.0,
    kinds: Iterable[str] | None = None,
) -> list[FileResult]:
    """Make every file of `release` (or those of `kinds`) present and verified under `raw_dir`.
    Safe to re-run."""
    open_url = opener or _urlopen
    wanted = None if kinds is None else frozenset(kinds)
    results = []
    for release_file in release.files:
        if wanted is not None and release_file.kind not in wanted:
            continue
        archive = raw_dir / "archives" / release_file.archive_name
        downloaded = ensure_archive(
            release_file, archive, opener=open_url, sleep=sleep, retries=retries, timeout=timeout
        )
        extracted = unpack(release_file, archive, raw_dir / release_file.kind)
        results.append(FileResult(release_file.kind, downloaded, extracted))
    return results


def is_complete(release: Release, raw_dir: Path, kinds: Iterable[str] | None = None) -> bool:
    """True when every file of `release` (or those of `kinds`) has been verified and unpacked
    under `raw_dir`."""
    wanted = None if kinds is None else frozenset(kinds)
    for release_file in release.files:
        if wanted is not None and release_file.kind not in wanted:
            continue
        marker = raw_dir / release_file.kind / COMPLETE_MARKER
        if not marker.is_file() or marker.read_text().strip() != release_file.sha256:
            return False
    return True


def ensure_archive(
    release_file: ReleaseFile,
    dest: Path,
    *,
    opener: Opener,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 3,
    timeout: float = 120.0,
) -> bool:
    """Download `release_file` to `dest` unless a verified copy exists; True if it downloaded."""
    return download(
        release_file.url,
        release_file.sha256,
        release_file.size,
        dest,
        opener=opener,
        sleep=sleep,
        retries=retries,
        timeout=timeout,
    )


def download(
    url: str,
    sha256: str,
    size: int,
    dest: Path,
    *,
    opener: Opener | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 3,
    timeout: float = 120.0,
) -> bool:
    """Download `url` to `dest` unless a verified copy exists, and check it has this SHA-256 and
    size; True if it downloaded. Server errors and dropped connections are retried."""
    if dest.exists():
        if sha256_of(dest) != sha256:
            raise FetchError(
                f"{dest} exists but does not match the registry SHA-256; refusing to overwrite it"
            )
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    open_url = opener or _urlopen
    for attempt in range(retries + 1):
        try:
            digest, got = _download(open_url(request, timeout), part)
            break
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code < 500 or attempt == retries:
                raise FetchError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise FetchError(f"could not download {url}: {exc}") from exc
        sleep(2.0**attempt)

    if digest != sha256 or got != size:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"{url} does not match the registry "
            f"(sha256 {digest}, {got} bytes; expected {sha256}, {size})"
        )
    part.replace(dest)
    return True


def unpack(release_file: ReleaseFile, archive: Path, target: Path) -> int:
    """Extract (or copy, for a PDF) `archive` into `target`. Returns the number of files written."""
    marker = target / COMPLETE_MARKER
    if marker.is_file() and marker.read_text().strip() == release_file.sha256:
        return 0
    target.mkdir(parents=True, exist_ok=True)
    if release_file.archive_name.endswith(".pdf"):
        name = PurePosixPath(release_file.url).name
        written = int(_place(archive.read_bytes(), target / name))
    else:
        written = _extract_zip(archive, target)
    marker.write_text(release_file.sha256 + "\n")
    return written


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urlopen(request: urllib.request.Request, timeout: float) -> IO[bytes]:
    response: IO[bytes] = urllib.request.urlopen(request, timeout=timeout)
    return response


def _download(response: IO[bytes], part: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with response, part.open("wb") as out:
        for chunk in iter(lambda: response.read(CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    return digest.hexdigest(), size


def _extract_zip(archive: Path, target: Path) -> int:
    with zipfile.ZipFile(archive) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) > MAX_MEMBERS:
            raise FetchError(f"{archive} has {len(members)} members; refusing to extract")
        if sum(info.file_size for info in members) > MAX_UNCOMPRESSED_BYTES:
            raise FetchError(f"{archive} expands to more than {MAX_UNCOMPRESSED_BYTES} bytes")
        paths = [_safe_member_path(archive, info) for info in members]
        tops = {p.parts[0] for p in paths}
        strip = len(tops) == 1 and all(len(p.parts) > 1 for p in paths)
        written = 0
        for info, member_path in zip(members, paths, strict=True):
            relative = PurePosixPath(*member_path.parts[1:]) if strip else member_path
            written += _place(zf.read(info), target.joinpath(*relative.parts))
        return written


def _safe_member_path(archive: Path, info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise FetchError(f"{archive} contains an unsafe path: {info.filename!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise FetchError(f"{archive} contains a symbolic link: {info.filename!r}")
    return path


def _place(content: bytes, dest: Path) -> int:
    """Write `content` to `dest` unless an identical file exists. Returns the files written."""
    if dest.exists():
        if dest.read_bytes() == content:
            return 0
        raise FetchError(f"{dest} exists with different content; refusing to overwrite it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(dest)
    return 1
