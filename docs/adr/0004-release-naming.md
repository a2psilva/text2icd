# ADR 0004: Naming ICD-10-CM releases and carrying them through artifacts

- Status: Accepted
- Date: 2026-11-01

## Context

CMS publishes a new ICD-10-CM code set every fiscal year, effective October 1, and some years also
publish an update effective April 1 (for example April 1, 2026). Codes are added, deleted and revised
between releases. The same phrase can therefore have a different correct code in different releases.

MIMIC-IV codes were assigned under the releases in force at each admission, roughly FY2016 to FY2022,
while the resolver targets current releases (FY2026, FY2027). Results, caches and benchmark numbers
are only meaningful if they state which release they were produced for.

CMS file names are not a usable identifier: their patterns change from year to year (for example
`2026-code-tables-tabular-and-index.zip` versus `2027-code-tables-tabular-index.zip`).

## Decision

- A **release id** names each code set: `fy<YEAR>` for the October release and `fy<YEAR>-apr` for an
  April update, for example `fy2026`, `fy2026-apr`, `fy2027`.
- A **release registry** in the package lists, per release id, the CMS download URLs and the SHA-256
  of every file.
- The release id appears in every path and artifact name: `data/raw/cms/<release>/`,
  `text2icd-graph-<release>.sqlite`, cache keys, benchmark result files, and the `release` field of
  every `CodingResult`.
- The default release is the newest one the installed version supports. Choosing another is always
  explicit (`--release fy2026`).

## Alternatives considered

- **Effective dates** (`2025-10-01`). Unambiguous, but less familiar to coders, who speak of fiscal
  years.
- **CMS file names.** Inconsistent across years, as shown above.
- **Only "latest".** Simpler, but results could not be reproduced once a new release appears.

## Consequences

- Several releases can be installed and compared side by side, which the MIMIC-IV version
  reconciliation and the annual update both need.
- Supporting a new release means adding a registry entry, building its graph and re-running the
  integrity suite and benchmarks.
- The files already in `data/raw/cms/fy2026/` must be matched against the registry hashes. If they
  turn out to be the April update, the directory is renamed to `fy2026-apr`.
