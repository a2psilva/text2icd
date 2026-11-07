"""text2icd: map clinical diagnosis phrases to ICD-10-CM codes like a medical coder would.

    import text2icd

    result = text2icd.code("type 2 DM with diabetic CKD stage 3a")
    [c.code for c in result.codes]  # ['E11.22', 'N18.31']

`code` and `code_many` share one `Resolver` per release, built on first use (about 2 s) from the
settings (`TEXT2ICD_*`, `.env`); `text2icd setup` puts the graph in place. For other settings, L3
or an explicit lifecycle, build a `text2icd.Resolver` yourself.
"""

import threading
from collections.abc import Iterable, Iterator
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("text2icd")
__all__ = ["CodingResult", "Resolver", "code", "code_many"]

if TYPE_CHECKING:
    from text2icd.resolve.resolver import Resolver
    from text2icd.resolve.tabular import Episode
    from text2icd.result import CodingResult
    from text2icd.text.model import Setting

_resolvers: dict[str | None, "Resolver"] = {}
_building = threading.Lock()


def code(
    phrase: str,
    *,
    release: str | None = None,
    setting: "Setting" = "inpatient",
    episode: "Episode | None" = None,
    explain: bool = False,
) -> "CodingResult":
    """Code one phrase with the shared resolver of `release` (default: TEXT2ICD_RELEASE, else
    the newest)."""
    return _resolver(release).code(phrase, setting=setting, episode=episode, explain=explain)


def code_many(
    phrases: Iterable[str],
    *,
    release: str | None = None,
    setting: "Setting" = "inpatient",
    episode: "Episode | None" = None,
    explain: bool = False,
) -> Iterator["CodingResult"]:
    """Code phrases in order, streamed (`Resolver.code_many`)."""
    return _resolver(release).code_many(phrases, setting=setting, episode=episode, explain=explain)


def _resolver(release: str | None) -> "Resolver":
    found = _resolvers.get(release)
    if found is None:
        with _building:
            found = _resolvers.get(release)
            if found is None:
                from text2icd.resolve.resolver import Resolver

                found = _resolvers[release] = Resolver(release)
    return found


def __getattr__(name: str) -> Any:
    """`text2icd.Resolver` and `text2icd.CodingResult`, imported on first use, so that
    `import text2icd` stays cheap."""
    if name == "Resolver":
        from text2icd.resolve.resolver import Resolver

        return Resolver
    if name == "CodingResult":
        from text2icd.result import CodingResult

        return CodingResult
    raise AttributeError(f"module 'text2icd' has no attribute {name!r}")
