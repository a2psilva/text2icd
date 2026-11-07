"""Structured logs that keep phrases private (task 9.7).

text2icd logs to the `text2icd` logger, one record per event, with its fields in `extra`
(`fields`). A phrase never appears in a record unless `TEXT2ICD_LOG_INPUTS=true`. By default
it is logged by its length and a salted hash: the hash tells repeated phrases apart within one
process, and the salt, new in each process, keeps it from being looked up or compared across
runs. The codes themselves are left out too, since they are derived from the phrase; only their
number is logged.

Nothing is printed until a level and a handler are set, by the application or by `configure()`,
which writes JSON lines to stderr. Until then an event costs one level check.
"""

import hashlib
import json
import logging
import secrets
import sys
import time
from typing import IO, Any

LOGGER = logging.getLogger("text2icd")
_SALT = secrets.token_bytes(16)  # per process


def phrase_hash(phrase: str) -> str:
    """A salted hash of a phrase, stable within this process only."""
    return hashlib.sha256(_SALT + phrase.encode("utf-8")).hexdigest()[:16]


def phrase_fields(phrase: str, log_inputs: bool) -> dict[str, Any]:
    """How a phrase appears in a record: its hash and length, and itself only when asked."""
    fields: dict[str, Any] = {"phrase": phrase_hash(phrase), "chars": len(phrase)}
    if log_inputs:
        fields["input"] = phrase
    return fields


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log one event with its fields, if the logger takes this level."""
    if LOGGER.isEnabledFor(level):
        LOGGER.log(level, name, extra={"fields": fields})


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, event, then the event's fields."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}))
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def configure(level: int | str = logging.INFO, stream: IO[str] | None = None) -> None:
    """Write text2icd's records as JSON lines (to stderr by default), once per process."""
    LOGGER.setLevel(level)
    if any(getattr(h, "_text2icd", False) for h in LOGGER.handlers):
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._text2icd = True  # type: ignore[attr-defined]
    LOGGER.addHandler(handler)
