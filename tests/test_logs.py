"""Task 9.7: logs show a phrase only as its length and a salted hash, unless `log_inputs` is set."""

import hashlib
import io
import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from text2icd.config import Settings
from text2icd.graph.model import Graph
from text2icd.logs import LOGGER, JsonFormatter, configure, phrase_hash
from text2icd.resolve.resolver import Resolver

PNEUMONIA = "aspiration pneumonia due to food"
PROPANOL = "poisoning by propanol"
PRIVATE = ("aspiration", "pneumonia", "propanol", "J69.0", "T51")  # words and codes of the phrases


def coded(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [r.fields for r in caplog.records if r.getMessage() == "coded"]  # type: ignore[attr-defined]


@pytest.fixture
def logger() -> Iterator[None]:
    """Puts text2icd's logger back as it was after `configure()`."""
    level, handlers = LOGGER.level, list(LOGGER.handlers)
    yield
    LOGGER.setLevel(level)
    LOGGER.handlers[:] = handlers


def test_logs_leave_the_phrase_out(graph: Graph, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="text2icd")
    resolver = Resolver(graph=graph)
    resolver.code(PNEUMONIA)
    resolver.code(PNEUMONIA)
    list(resolver.code_many([PROPANOL]))
    first, again, other = coded(caplog)
    assert set(first) == {
        "phrase",
        "chars",
        "codes",
        "resolver",
        "escalated",
        "cached",
        "llm_calls",
        "cost_usd",
        "ms",
    }
    assert first["phrase"] == again["phrase"] != other["phrase"]
    assert (first["chars"], first["codes"], first["resolver"]) == (len(PNEUMONIA), 1, "index")
    assert (first["cached"], again["cached"]) == (False, True)
    assert "ms" not in other  # a batch is timed as a whole
    written = [JsonFormatter().format(r) for r in caplog.records]
    assert any('"event": "batch"' in line for line in written)
    for word in PRIVATE:
        assert word not in caplog.text
        assert not any(word in line for line in written)


def test_log_inputs_adds_the_phrase_and_codes(
    graph: Graph, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger="text2icd")
    monkeypatch.setenv("TEXT2ICD_LOG_INPUTS", "true")
    Resolver(graph=graph).code(PNEUMONIA)
    (fields,) = coded(caplog)
    assert (fields["input"], fields["coded"]) == (PNEUMONIA, ["J69.0"])
    (record,) = caplog.records
    assert PNEUMONIA in JsonFormatter().format(record)  # what the test above checks is absent


def test_nothing_is_logged_unless_asked(graph: Graph, caplog: pytest.LogCaptureFixture) -> None:
    """text2icd's logger takes the root's level (WARNING) until a level is set."""
    Resolver(graph=graph, settings=Settings()).code(PNEUMONIA)
    assert not [r for r in caplog.records if r.name == "text2icd"]


def test_the_hash_is_salted_per_process() -> None:
    assert phrase_hash(PNEUMONIA) == phrase_hash(PNEUMONIA) != phrase_hash(PROPANOL)
    assert len(phrase_hash(PNEUMONIA)) == 16
    assert phrase_hash(PNEUMONIA) != hashlib.sha256(PNEUMONIA.encode()).hexdigest()[:16]
    script = f"from text2icd.logs import phrase_hash; print(phrase_hash({PNEUMONIA!r}))"
    elsewhere = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert elsewhere.stdout.strip() != phrase_hash(PNEUMONIA)


@pytest.mark.usefixtures("logger")
def test_configure_writes_json_lines(graph: Graph) -> None:
    stream = io.StringIO()
    configure(stream=stream)
    configure(stream=stream)  # a second call adds no handler
    Resolver(graph=graph).code(PNEUMONIA)
    (line,) = [json.loads(text) for text in stream.getvalue().splitlines()]
    assert (line["event"], line["level"], line["logger"]) == ("coded", "info", "text2icd")
    assert line["time"].endswith("Z")
    assert line["chars"] == len(PNEUMONIA)
    for word in PRIVATE:
        assert word not in stream.getvalue()
