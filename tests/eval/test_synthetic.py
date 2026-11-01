import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.eval import synthetic
from text2icd.eval.datasets import Example, dataset_hash, read_dataset
from text2icd.eval.synthetic import (
    Seed,
    abbreviate,
    add_typo,
    chart_phrase,
    clean,
    drop_parentheses,
    generate,
    index_phrase,
    is_adjective,
    lower_first,
    reorder,
    sample,
    seeds,
    unwrap_one,
)
from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.ontology.codes import category

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SAMPLE = REPO_ROOT / "benchmarks" / "cms-synthetic" / "fy2027-sample.jsonl"


def by_source(graph: Graph) -> dict[str, Seed]:
    return {seed.source: seed for seed in seeds(graph)}


def test_index_seeds(graph: Graph) -> None:
    found = by_source(graph)
    food = found["Pneumonia > aspiration > due to > food"]
    assert (food.phrase, food.gold) == ("aspiration pneumonia due to food", (("J69.0", "primary"),))
    assert food.reordered == "pneumonia, aspiration, due to food"
    assert found["Pneumonia > aspiration"].phrase == "aspiration pneumonia"
    assert found["Pneumonia > with > lung abscess"].phrase == "pneumonia with lung abscess"
    assert found["Pyelitis > chronic"].phrase == "chronic pyelitis"
    assert found["Pyelitis > tuberculous"].phrase == "tuberculous pyelitis"
    hydronephrosis = found["Pyelitis > with > calculus > with hydronephrosis"]
    assert hydronephrosis.phrase == "pyelitis with calculus, with hydronephrosis"
    amyloid = found["Amyloid heart"]
    assert amyloid.gold == (("E85.4", "primary"), ("I43", "companion"))
    assert amyloid.with_modifier == "amyloid heart disease"
    # Terms whose codes are not in the release give no phrase.
    assert "Pneumonia > aspiration > newborn" not in found


def test_inclusion_seed(graph: Graph) -> None:
    seed = by_source(graph)["Pneumonitis due to inhalation of food (regurgitated)"]
    assert seed.generator == "inclusion"
    assert seed.phrase == "pneumonitis due to inhalation of food"
    assert seed.with_modifier == "pneumonitis due to inhalation of food regurgitated"
    assert seed.reordered == "food pneumonitis due to inhalation"
    # Inclusion terms of codes that need more characters are left out.
    assert "Toxic effect of fusel oil NOS" not in by_source(graph)


def test_table_seeds(graph: Graph) -> None:
    table = [seed for seed in seeds(graph) if seed.generator in ("neoplasm", "drug")]
    neoplasm = {seed.source.split(" [")[0]: seed for seed in table if seed.generator == "neoplasm"}
    assert neoplasm["Neoplasm, neoplastic > abdomen, abdominal"].phrase in (
        "malignant neoplasm of abdomen",
        "benign neoplasm of abdomen",
    )
    (drug,) = [seed for seed in table if seed.generator == "drug"]
    pattern = r"(accidental poisoning by|intentional self-poisoning with|assault by poisoning with)"
    pattern += r" 1-propanol, (initial encounter|subsequent encounter|sequela)"
    assert re.fullmatch(pattern, drug.phrase), drug.phrase
    code = drug.gold[0][0]
    assert code.startswith("T51.3X")
    assert code[-1] == drug.seventh


def test_generate(graph: Graph) -> None:
    examples = generate(graph)
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))
    for example in examples:
        assert all(graph.code(gold.code).billable for gold in example.gold)
        assert example.meta["perturbation"] in synthetic.PERTURBATIONS
        assert example.release == "fy2099"
    base = Counter(str(e.meta["from"]) for e in examples if e.meta["perturbation"] == "base")
    assert all(n == 1 for n in base.values())
    per_seed = Counter(e.id.rsplit("-", 1)[0] for e in examples if "-word-order" not in e.id)
    assert max(per_seed.values()) <= 2
    splits = {category(e.primary.code): e.split for e in examples}
    assert all(splits[category(e.primary.code)] == e.split for e in examples)
    assert generate(graph) == examples
    assert dataset_hash(generate(graph, rng_seed=1)) != dataset_hash(examples)


def test_ambiguous_phrases_are_dropped(graph: Graph, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(graph: Graph, rng_seed: int = 0) -> Iterator[Seed]:
        yield Seed("a", "index", "same words", (("J69.0", "primary"),), "A")
        yield Seed("b", "index", "Same  words", (("J12.9", "primary"),), "B")
        yield Seed("c", "index", "other words", (("J12.9", "primary"),), "C")
        yield Seed("d", "inclusion", "other words", (("J12.9", "primary"),), "D")

    monkeypatch.setattr(synthetic, "seeds", fake)
    assert {e.id for e in generate(graph) if e.meta["perturbation"] == "base"} == {"c-base"}


def example(i: int, generator: str, perturbation: str) -> Example:
    return Example.model_validate(
        {
            "id": f"{generator}-{i:03d}-{perturbation}",
            "phrase": f"phrase {i}",
            "gold": [{"code": "J69.0", "role": "primary"}],
            "release": "fy2027",
            "source": "cms-synthetic",
            "split": "dev",
            "meta": {"generator": generator, "perturbation": perturbation},
        }
    )


def test_sample_is_stratified() -> None:
    examples = [example(i, "index", "base") for i in range(100)]
    examples += [example(i, "index", "typo") for i in range(50)]
    examples += [example(i, "drug", "base") for i in range(5)]
    chosen = sample(examples, size=30)
    counts = Counter((str(e.meta["generator"]), str(e.meta["perturbation"])) for e in chosen)
    assert counts == {("index", "base"): 19, ("index", "typo"): 10, ("drug", "base"): 5}
    assert [e.id for e in chosen] == sorted(e.id for e in chosen)
    assert sample(examples, size=30) == chosen
    assert sample([], size=30) == []


@pytest.mark.parametrize(
    ("path", "phrase"),
    [
        ("Pneumonia > aspiration > due to > food", "aspiration pneumonia due to food"),
        ("Disease, diseased > Brill-Zinsser", "Brill-Zinsser disease"),
        ("Mesothelioma > benign > peritoneum", "benign mesothelioma of peritoneum"),
        ("Aplasia > lung, congenital", "congenital lung aplasia"),
        ("Web, webbed > neck", "neck web"),
        ("Fibroadenosis, breast", "breast fibroadenosis"),
        ("Cleft > mitral valve leaflet", "cleft of mitral valve leaflet"),
        (
            "Calculus, calculi, calculous > bile duct > with > cholecystitis > with obstruction",
            "bile duct calculus with cholecystitis, with obstruction",
        ),
        (
            "Abnormal, abnormality > Papanicolaou > cervix > low grade",
            "abnormal Papanicolaou cervix, low grade",
        ),
        ("History > personal > fracture", "personal history of fracture"),
        ("Hyperlipemia > specified NEC", "specified hyperlipemia"),
        ("HIV > disease", "HIV disease"),
    ],
)
def test_index_phrase(path: str, phrase: str) -> None:
    assert index_phrase(path.split(" > ")) == phrase


def test_chart_phrase() -> None:
    titles = ["Hernia, hernial", "diaphragm, diaphragmatic", "congenital"]
    assert chart_phrase(titles) == "hernia, diaphragm, congenital"
    assert chart_phrase(["Pneumonia"]) == "pneumonia"


def test_text_helpers() -> None:
    assert clean("Uveitis NOS [any site] ,") == "Uveitis"
    assert drop_parentheses("Abscess (septic (deep)) of kidney") == "Abscess of kidney"
    rng = random.Random(0)
    assert unwrap_one("Acute (congestive) heart failure", rng) == "Acute congestive heart failure"
    assert unwrap_one("Bleb (s)", rng) is None
    assert lower_first("Pneumonia") == "pneumonia"
    assert lower_first("HIV disease") == "HIV disease"
    assert lower_first("Dawson's encephalitis") == "Dawson's encephalitis"
    assert lower_first("McArdle disease") == "McArdle disease"
    assert is_adjective("acute")
    assert is_adjective("sleep-related")
    assert is_adjective("third degree")
    assert not is_adjective("lung")
    assert not is_adjective("ovary")


def test_reorder_and_abbreviate() -> None:
    assert reorder("tuberculosis of sinus") == "sinus tuberculosis"
    assert reorder("conduct disorder, group type") == "group type conduct disorder"
    assert reorder("viral thyroiditis") is None
    assert abbreviate("type 2 diabetes mellitus with CKD") == "T2DM with CKD"
    assert abbreviate("Congestive heart failure") == "CHF"
    assert abbreviate("pneumonia") is None


def test_add_typo() -> None:
    rng = random.Random(3)
    typos = {add_typo("acute pneumonia", rng) for _ in range(50)}
    assert None not in typos
    for typo in typos:
        assert typo is not None
        words = typo.split(" ")
        assert words[0] == "acute" or words[1] == "pneumonia"
        assert typo != "acute pneumonia"
    assert add_typo("flu", rng) is None


runner = CliRunner()


def test_synth_command(graph_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "set.jsonl"
    sample_output = tmp_path / "sample.jsonl"
    args = ["eval", "synth", "--graph", str(graph_file), "-o", str(output)]
    args += ["--sample-output", str(sample_output), "--sample-size", "5"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "index:" in result.output
    examples = read_dataset(output)
    manifest = json.loads(output.with_name("set.manifest.json").read_text())
    assert manifest["sha256"] == dataset_hash(examples)
    assert (manifest["name"], manifest["seed"]) == ("cms-synthetic", 0)
    assert manifest["release"] == "fy2099"
    sampled = json.loads(sample_output.with_name("sample.manifest.json").read_text())
    assert sampled["sample_of"] == manifest["sha256"]
    assert len(read_dataset(sample_output)) <= len(examples)


def any_example(examples: list[Example], **meta: Any) -> list[Example]:
    return [e for e in examples if all(e.meta.get(k) == v for k, v in meta.items())]


@pytest.mark.slow
def test_release_set(release_graph: Callable[[str], BuildReport]) -> None:
    with load(path=release_graph("fy2027").path) as graph:
        examples = generate(graph)
        for example in examples:
            assert all(graph.code(gold.code).billable for gold in example.gold), example.id
    assert len(examples) > 100_000
    phrases = {e.phrase: e for e in examples}
    assert phrases["aspiration pneumonia due to food"].primary.code == "J69.0"
    assert sum(bool(e.companions) for e in examples) > 500
    assert len(any_example(examples, generator="drug", perturbation="base")) > 5_000
    # The committed sample is what the generator makes: regenerate it when this fails.
    committed = read_dataset(COMMITTED_SAMPLE)
    assert dataset_hash(committed) == dataset_hash(sample(examples))
