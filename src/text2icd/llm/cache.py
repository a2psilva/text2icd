"""The decision cache (task 8.9): identical decisions reuse the stored choice.

A decision's key hashes its kind, phrase, path and options (`Decision.key`); the cache key adds
the provider, model, effort and prompt version, so a change of any of them asks again. The store
is one SQLite file in the cache directory (`TEXT2ICD_CACHE_DIR/decisions.sqlite`) and holds
only those hashes and the option ids chosen, never the phrase. Only the model's own answers are
stored, `none` included; limits and errors are not, so a later run asks again.

The batch path (task 8.8) fills the same cache from a Message Batch, so a run that follows reads
every answer from it.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from types import TracebackType

from text2icd.llm.decisions import Choice, Decision
from text2icd.llm.prompts import PROMPT_VERSION

FILE = "decisions.sqlite"
BUSY_TIMEOUT_S = 10.0  # how long a write waits for another process's write
_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision (
    key TEXT PRIMARY KEY,
    option TEXT,  -- the option id chosen; NULL for none
    created TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def namespace(provider: str, model: str, effort: str) -> str:
    """What answers depend on besides the decision itself."""
    return f"{provider}/{model}/{effort}/prompt-{PROMPT_VERSION}"


def cache_key(decision: Decision, space: str) -> str:
    return hashlib.sha256(f"{space}|{decision.key}".encode()).hexdigest()


class DecisionCache:
    """Thread-safe (task 9.10): one connection behind a lock. WAL lets other processes (an
    `eval run` beside a server) read while one writes, and a busy timeout waits out their
    writes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_S, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, decision: Decision, space: str) -> Choice | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT option FROM decision WHERE key = ?", (cache_key(decision, space),)
            ).fetchone()
        if row is None:
            return None
        option = row[0] if decision.option(row[0]) is not None else None
        return Choice(option=option, by="cache", reason=None if option else "no option fits")

    def put(self, decision: Decision, space: str, option: str | None) -> None:
        self.put_key(cache_key(decision, space), option)

    def put_key(self, key: str, option: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decision (key, option) VALUES (?, ?)", (key, option)
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM decision").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DecisionCache":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
