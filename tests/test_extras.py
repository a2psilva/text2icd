"""Task 10.6: the optional extras, and hints that install them from GitHub, never from PyPI."""

import tomllib
from pathlib import Path

from text2icd.extras import REPOSITORY, install_hint

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_hint_installs_from_github() -> None:
    hint = install_hint("embeddings")
    assert "uv sync --extra embeddings" in hint
    assert f'"text2icd[embeddings] @ git+{REPOSITORY}"' in hint


def test_extras() -> None:
    extras = tomllib.loads(PYPROJECT.read_text("utf-8"))["project"]["optional-dependencies"]
    names = {extra: {_name(req) for req in reqs} for extra, reqs in extras.items()}
    # L2 imports all three; model2vec is the evaluation's baseline only.
    assert names["embeddings"] == {"onnxruntime", "tokenizers", "huggingface-hub"}
    assert "model2vec" in names["eval"]
    assert names["api"] == {"fastapi", "uvicorn"}
    assert extras["all"] == ["text2icd[api,embeddings,llm]"]


def _name(requirement: str) -> str:
    for stop in "[<>=!~ ;":
        requirement = requirement.split(stop, 1)[0]
    return requirement.strip().lower().replace("_", "-")
