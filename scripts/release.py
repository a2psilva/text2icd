"""Check a release tag and write its notes (task 10.7), for .github/workflows/release.yml.

    python scripts/release.py v0.2.0 --notes notes.md

- The tag must be `v` followed by the version in pyproject.toml, written as MAJOR.MINOR.PATCH
  with an optional pre-release (a, b or rc and a number: 0.2.0rc1).
- CHANGELOG.md must have a `## [<version>] - YYYY-MM-DD` section. Its text becomes the notes,
  followed by how to install the release's wheel and check its files.

It prints `version=...` and `prerelease=true|false`, for $GITHUB_OUTPUT. Without a tag it checks
the current version, as the workflow's dry run does. It exits with 1 and a message when they
disagree. Standard library only.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPOSITORY = "a2psilva/text2icd"
VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")


class ReleaseError(ValueError):
    pass


def project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as fh:
        version = str(tomllib.load(fh)["project"]["version"])
    if not VERSION.fullmatch(version):
        raise ReleaseError(
            f"pyproject.toml's version {version!r} is not MAJOR.MINOR.PATCH, with an optional "
            "a, b or rc pre-release (0.2.0, 0.2.0rc1)"
        )
    return version


def changes(root: Path, version: str) -> str:
    """The text of the version's section of CHANGELOG.md."""
    text = (root / "CHANGELOG.md").read_text("utf-8")
    heading = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}[ \t]*\n"
    found = re.search(heading + r"(.*?)(?=^## \[|\Z)", text, re.MULTILINE | re.DOTALL)
    if found is None:
        raise ReleaseError(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")
    body = found.group(1).strip()
    if not body:
        raise ReleaseError(f"CHANGELOG.md's section for {version} is empty")
    return body


def notes(version: str, body: str) -> str:
    wheel = f"text2icd-{version}-py3-none-any.whl"
    url = f"https://github.com/{REPOSITORY}/releases/download/v{version}/{wheel}"
    return f"""{body}

## Install

```bash
uv tool install "text2icd[all] @ {url}"
text2icd setup
```

`pip install "text2icd[all] @ {url}"` works too, and so does the tag:
`git+https://github.com/{REPOSITORY}@v{version}`. `SHA256SUMS` lists each file's checksum, and
`gh attestation verify {wheel} -R {REPOSITORY}` checks that the release workflow built it.

The image, with the graph, L2's vectors and the model built in (no network needed):

```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/{REPOSITORY}:{version}
```
"""


def check(root: Path, tag: str | None) -> tuple[str, str]:
    """The version and the notes; `ReleaseError` when the tag, the version and the changelog
    disagree."""
    version = project_version(root)
    if tag is not None and tag != f"v{version}":
        raise ReleaseError(f"the tag {tag!r} is not v{version}, pyproject.toml's version")
    return version, notes(version, changes(root, version))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", nargs="?", help="The tag, such as v0.2.0 (default: none)")
    parser.add_argument("--notes", type=Path, help="Write the release notes here")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        version, text = check(args.root, args.tag)
    except (ReleaseError, OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.notes is not None:
        args.notes.write_text(text, encoding="utf-8")
    print(f"version={version}")
    print(f"prerelease={'true' if re.search('(a|b|rc)[0-9]+$', version) else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
