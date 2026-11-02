from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.main_terms import MainTerms, Recall, main_term_recall
from text2icd.graph.model import Graph


def example(i: int, phrase: str, code: str, **gold: Any) -> Example:
    return Example.model_validate(
        {
            "id": f"x{i}",
            "phrase": phrase,
            "gold": [{"code": code, "role": "primary", **gold}],
            "release": "fy2099",
            "source": "test",
            "split": "dev",
        }
    )


def test_right_main_terms(graph: Graph) -> None:
    index = MainTerms(graph)
    pneumonia = graph.term_at("Pneumonia")
    amyloid = graph.term_at("Amyloid heart")
    assert pneumonia is not None
    assert amyloid is not None
    assert index.right(example(1, "x", "J69.0")) == {pneumonia.id}
    # The manifestation code is reached from the same main term.
    assert index.right(example(2, "x", "I43")) == {amyloid.id}
    # A 7th-character code is reached through the code above it (a Table of Drugs cell).
    propanol = graph.term_at("1-propanol", "drug")
    assert propanol is not None
    assert index.right(example(3, "x", "T51.3X1A")) == {propanol.id}
    mapped = example(4, "x", "J18.0", status="mapped", mapped_to=["J69.0"])
    assert index.right(mapped) == {pneumonia.id}
    assert index.right(example(5, "x", "E08.3211")) == set()
    food = graph.term_at("Pneumonia > aspiration > due to > food")
    assert food is not None
    assert index.root(food.id) == pneumonia.id


def test_recall(graph: Graph) -> None:
    examples = [
        example(1, "aspiration pneumonia due to food", "J69.0"),
        example(2, "amyloid heart", "E85.4"),
        example(3, "zzz", "J69.0"),
        example(4, "diabetes with retinopathy", "E08.3211"),  # no main term leads to it
    ]
    results = {(r.mode, r.k): r for r in main_term_recall(graph, examples)}
    assert results[("main", 1)].total == 3
    assert results[("any", 10)].found == 2
    assert results[("any", 1)].found == 2
    assert results[("coder", 1)].found == 2
    assert results[("main", 10)].line().startswith("main  recall@10 66.7% (95% CI ")
    assert Recall("any", 1, 0, 0).line() == "any recall@1: no examples"


def test_main_terms_command(graph_file: Path, tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    write_dataset(path, [example(1, "aspiration pneumonia", "J69.0")])
    result = CliRunner().invoke(
        cli.app, ["eval", "main-terms", "-d", str(path), "--graph", str(graph_file)]
    )
    assert result.exit_code == 0, result.output
    assert "set.jsonl (dev), tokenizer porter, 1 examples" in result.output
    assert "any   recall@1  100.0%" in result.output
    assert "coder recall@1  100.0%" in result.output
    missing = CliRunner().invoke(
        cli.app, ["eval", "main-terms", "-d", "nothing", "--graph", str(graph_file)]
    )
    assert missing.exit_code == 1
