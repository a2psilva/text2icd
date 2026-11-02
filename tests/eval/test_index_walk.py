"""`eval index`, the Index walk measured alone, and the `index` system."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.datasets import Example, write_dataset
from text2icd.eval.index import Tally, index_stats
from text2icd.eval.systems import SystemSetupError, build_system
from text2icd.graph.model import Graph
from text2icd.resolve.walk import Candidate, IndexWalker


def example(i: int, phrase: str, code: str, generator: str = "index", **gold: Any) -> Example:
    return Example.model_validate(
        {
            "id": f"x{i}",
            "phrase": phrase,
            "gold": [{"code": code, "role": "primary", **gold}],
            "release": "fy2099",
            "source": "cms-synthetic",
            "split": "dev",
            "meta": {"generator": generator},
        }
    )


EXAMPLES = [
    example(1, "aspiration pneumonia due to food", "J69.0"),
    example(2, "chronic pyelitis with calculus with hydronephrosis", "N13.6"),
    example(3, "amyloid heart", "E85.4"),
    example(4, "pyelitis with contracted kidney", "N11.0", "inclusion"),  # N11.9 is found
    example(5, "zzz", "J69.0", "inclusion"),
    example(6, "aspiration pneumonia", "J18.0", status="mapped", mapped_to=["J69.0"]),
]


def test_index_stats(graph: Graph) -> None:
    stats = index_stats(EXAMPLES, IndexWalker(graph), keep=5)
    total = stats.tallies["all"]
    assert total.phrases == 6
    assert total.found[1] == 4
    assert total.category == 5  # N11.9 is in the gold category N11
    assert total.default_code == 1  # Amyloid heart has no subterm
    assert total.none == 1
    assert stats.tallies["inclusion"].found[1] == 0
    assert [miss.id for miss in stats.misses] == ["x4", "x5"]
    assert stats.misses[1].found == "no candidate"
    lines = stats.lines()
    assert "prefix match of the primary gold code" in lines[1]
    assert any(line.startswith("  index (4): top-1 100.0%") for line in lines)
    assert any(line.startswith("  inclusion (2): top-1 0.0%") for line in lines)


def test_partial_codes_match_by_prefix() -> None:
    tally = Tally()
    partial = Candidate(code="S72.9", partial=True, score=2, path="p", coded_at="p")
    assert tally.add([partial], ["S72.90XA"]) == 0
    assert tally.add([partial], ["S73.0"]) is None
    assert (tally.found[1], tally.found[5], tally.category) == (1, 1, 1)


def test_index_command(graph_file: Path, tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    write_dataset(path, EXAMPLES)
    result = CliRunner().invoke(
        cli.app,
        ["eval", "index", "-d", str(path), "--graph", str(graph_file), "--show", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "set.jsonl (dev), 6 examples" in result.output
    assert "all (6): top-1 66.7%" in result.output
    assert "  x4: pyelitis with contracted kidney | gold N11.0 | N11.9 at" in result.output


def test_index_command_refuses_to_show_restricted(graph_file: Path, tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    restricted = example(1, "aspiration pneumonia", "J69.0").model_copy(
        update={"source": "mimic-silver"}
    )
    write_dataset(path, [restricted])
    args = ["eval", "index", "-d", str(path), "--graph", str(graph_file)]
    assert CliRunner().invoke(cli.app, [*args, "--show", "1"]).exit_code == 2
    assert CliRunner().invoke(cli.app, args).exit_code == 0


def test_walk_command(graph_file: Path) -> None:
    runner = CliRunner()
    args = ["walk", "aspiration pneumonia due to food", "--graph", str(graph_file)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1. J69.0" in result.output
    assert "subterm: food (uses food)" in result.output
    as_json = runner.invoke(cli.app, [*args, "--json"])
    assert '"code": "J69.0"' in as_json.output
    assert runner.invoke(cli.app, [*args, "--setting", "x"]).exit_code == 2


def test_index_system(graph: Graph) -> None:
    system = build_system("index", graph, Settings(), {"width": "3"})
    assert system.settings() == {"width": 3, "setting": "inpatient"}
    amyloid = system.predict("amyloid heart")
    assert amyloid.codes == ("E85.4", "I43")  # the manifestation code follows
    both = system.predict("aspiration pneumonia and chronic pyelitis")
    assert both.codes == ("J69.0", "N11.9")
    assert system.predict("zzz").codes == ()
    with pytest.raises(SystemSetupError, match="unknown options"):
        build_system("index", graph, Settings(), {"k": "3"})
    with pytest.raises(SystemSetupError, match="setting"):
        build_system("index", graph, Settings(), {"setting": "x"})
