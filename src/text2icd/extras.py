"""The optional extras and how to install one.

text2icd is not published on PyPI: it installs from a clone or straight from GitHub. So a hint
never says `pip install text2icd[...]`, which would fetch whatever package of that name PyPI
holds.

- `embeddings`: L2, local retrieval (onnxruntime, tokenizers, huggingface_hub).
- `llm`, `bedrock`, `vertex`: L3, the LLM navigator, on the Anthropic API, Amazon Bedrock or
  Google Cloud Vertex AI.
- `eval`: the evaluation's static-embedding baseline and calibration fitting.
- `all`: `embeddings` and `llm`.
"""

REPOSITORY = "https://github.com/a2psilva/text2icd"


def install_hint(extra: str) -> str:
    """How to install `extra`, in a checkout or from GitHub."""
    return (
        f"uv sync --extra {extra} in a checkout, or "
        f'pip install "text2icd[{extra}] @ git+{REPOSITORY}"'
    )
