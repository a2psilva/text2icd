# Releasing

text2icd is released on GitHub, not on PyPI. A release is a version tag: the *Release* workflow
turns it into a GitHub Release with the wheel and the sdist, and the *Pages demo* workflow deploys
the web demo. Graphs and vectors are released separately, as data releases.

## A code release

1. **Data first**, when the release should offer a new graph.
   - Run the *Data release* workflow for the release id (`fy2027`): first without `publish`,
     then with it.
   - Merge the pull request it opens, which adds the files to
     `src/text2icd/resources/graphs.toml`.

   A wheel offers the graphs listed in that file when it is built.
2. **Version and changelog**, in one pull request:
   - set `version` in `pyproject.toml`, then run `uv lock`, since `uv.lock` records it. Use
     MAJOR.MINOR.PATCH, or a pre-release such as `0.2.0rc1`;
   - in `CHANGELOG.md`, rename `## [Unreleased]` to `## [0.2.0] - YYYY-MM-DD`, open it with a
     line on what the release brings, and add a new, empty `## [Unreleased]` above it.

   `python scripts/release.py` checks that the version and the changelog agree, and so does the
   test suite.
3. **Dry run.** Run the *Release* workflow by hand on `main`. It checks the version and builds
   the sdist and the wheel. It then runs the test suite against the wheel, installed with every
   extra in a clean environment. It releases nothing.
4. **Tag** the merged commit:

   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "text2icd 0.2.0"
   git push origin v0.2.0
   ```

   - *Release* checks the tag against the version and the changelog, then builds and tests the
     wheel again. It attests the build provenance of both files and creates the GitHub Release,
     with the changelog's section and install lines as notes. A pre-release version (`a`, `b`
     or `rc`) becomes a pre-release.
   - *Pages demo* exports the demo and deploys it. Before the first deploy, set Settings →
     Pages → Source to GitHub Actions.
5. **Check the release page**: the wheel, the sdist and `SHA256SUMS` should be there, and
   `gh attestation verify text2icd-0.2.0-py3-none-any.whl -R a2psilva/text2icd` should pass on
   the downloaded wheel.

If a workflow fails after the tag is pushed, fix the cause on `main` and release the next patch
version. Never move a tag once its release is published.

## Versions

text2icd follows [semantic versioning](https://semver.org/). Before 1.0, a minor version may
change the command line, the Python API or the `CodingResult` schema; the schema carries its own
`schema_version`. The graph file has a schema number of its own (`graph/build.py`), which data
releases carry in their tag (`data-fy2027-s1`). A new graph schema means new data releases, and
the code release that reads it lists them.
