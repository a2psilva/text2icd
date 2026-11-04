"""Tabular verification: completing Index candidates to billable codes, and the coder."""

from collections.abc import Callable, Iterator

import pytest
from typer.testing import CliRunner

from text2icd import cli
from text2icd.config import Settings
from text2icd.eval.systems import SystemSetupError, build_system
from text2icd.graph.build import BuildReport
from text2icd.graph.model import Graph, load
from text2icd.resolve.coder import Coder, describe_code_set
from text2icd.resolve.confidence import read_calibration
from text2icd.resolve.tabular import ASSUMED, REVIEW, UNSPECIFIED, Evidence, TabularVerifier
from text2icd.resolve.words import default_word_forms
from text2icd.text.model import Modifier

FORMS = default_word_forms()


def evidence(text: str, *modifiers: tuple[str, str]) -> Evidence:
    found = tuple(
        Modifier.model_validate({"kind": kind, "value": value, "text": value, "start": 0, "end": 1})
        for kind, value in modifiers
    )
    return Evidence(frozenset(FORMS.content(text)), found, text)


def test_billable_code_is_kept(graph: Graph) -> None:
    done = TabularVerifier(graph).complete("E11.22", evidence("type 2 diabetes with ckd"))
    assert done is not None
    assert (done.code, done.flags, done.trace) == ("E11.22", (), ())
    assert TabularVerifier(graph).complete("Z99.99", evidence("x")) is None


def test_children_matched_by_their_words(graph: Graph) -> None:
    verifier = TabularVerifier(graph)
    words = "mild nonproliferative diabetic retinopathy with macular edema, right eye"
    done = verifier.complete("E08.32", evidence(words))
    assert done is not None
    assert done.code == "E08.3211"  # with macular edema, then 1 = right eye
    assert done.trace[-1].action == "complete"
    assert done.trace[-1].guideline == "I.B.1"


def test_unspecified_when_the_phrase_says_no_more(graph: Graph) -> None:
    done = TabularVerifier(graph).complete(
        "E08.321", evidence("mild nonproliferative diabetic retinopathy with macular edema")
    )
    assert done is not None
    assert done.code == "E08.3219"  # unspecified eye
    assert UNSPECIFIED in done.flags
    step = next(step for step in done.trace if step.action == "unspecified")
    assert step.guideline == "I.A.9.b, I.B.18"


def test_seventh_character_from_the_episode(graph: Graph) -> None:
    verifier = TabularVerifier(graph)
    subsequent = verifier.complete(
        "T51.3X1", evidence("subsequent encounter", ("episode", "subsequent"))
    )
    assert subsequent is not None
    assert (subsequent.code, subsequent.flags) == ("T51.3X1D", ())
    silent = verifier.complete("T51.3X1", evidence("fusel oil"))
    assert silent is not None
    assert (silent.code, silent.flags) == ("T51.3X1A", (ASSUMED,))
    sequela = TabularVerifier(graph, default_episode="sequela").complete(
        "T51.3X1", evidence("fusel oil")
    )
    assert sequela is not None
    assert sequela.code == "T51.3X1S"


def test_coder(graph: Graph) -> None:
    coder = Coder(graph)
    poisoning = coder.code("accidental poisoning by 1-propanol, subsequent encounter")
    assert poisoning.codes == ("T51.3X1D",)
    (coded,) = poisoning.diagnoses
    assert coded.candidate.code == "T51.3X1"
    assert [step.layer for step in coded.trace][-1] == "tabular"
    assert coder.code("amyloid heart disease").codes == ("E85.4", "I43")
    assert coder.code("aspiration pneumonia due to food").codes == ("J69.0",)
    # The fixture's Tabular List has no J18.9: no code, never an invalid one.
    missing = coder.code("pneumonia")
    assert (missing.codes, missing.flags) == ((), (REVIEW,))
    lines = describe_code_set(coder.code("amyloid heart disease"))
    assert lines[1] == "codes: E85.4, I43 (manifestation)"
    assert lines[2].startswith("confidence: ")  # from the shipped calibration table
    assert lines[3].startswith("1. E85.4 [I43]  Organ-limited amyloidosis")
    assert describe_code_set(missing)[1] == "no code"


def test_coder_system(graph: Graph) -> None:
    system = build_system("coder", graph, Settings(), {"episode": "subsequent"})
    settings = system.settings()
    shipped = read_calibration()
    assert settings.pop("calibration") == (shipped.fitted.get("name") if shipped else None)
    assert settings == {
        "width": 5,
        "setting": "inpatient",
        "episode": "subsequent",
        "retrieval": None,
        "llm": None,
    }
    assert system.predict("accidental poisoning by 1-propanol").codes == ("T51.3X1D",)
    with pytest.raises(SystemSetupError, match="episode"):
        build_system("coder", graph, Settings(), {"episode": "later"})
    with pytest.raises(SystemSetupError, match="retrieval must be on or off"):
        build_system("coder", graph, Settings(), {"retrieval": "maybe"})
    with pytest.raises(SystemSetupError, match="unknown options"):
        build_system("coder", graph, Settings(), {"k": "1"})


def test_code_command(graph_file: object) -> None:
    runner = CliRunner()
    args = ["code", "amyloid heart disease", "--graph", str(graph_file)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1. E85.4 [I43]" in result.output
    assert '"code": "E85.4"' in runner.invoke(cli.app, [*args, "--json"]).output


@pytest.fixture(scope="module")
def release(release_graph: Callable[[str], BuildReport]) -> Iterator[Graph]:
    with load(path=release_graph("fy2027").path) as graph:
        yield graph


@pytest.mark.slow
@pytest.mark.parametrize(
    ("phrase", "codes"),
    [
        ("femur fracture, initial encounter", ("S72.90XA",)),
        ("hip fracture due to fall", ("S72.009A", "W19.XXXA")),
        (
            "open fracture of shaft of right femur, subsequent encounter with delayed healing",
            ("S72.301H",),
        ),
        ("malignant neoplasm of right upper lobe of lung", ("C34.11",)),
        ("malignant neoplasm of upper lobe of lung", ("C34.10",)),
        ("adverse effect of digoxin, initial encounter", ("T46.0X5A",)),
        ("type 2 DM with diabetic CKD stage 3a", ("E11.22", "N18.31")),
        ("amyloid heart disease", ("E85.4", "I43")),
        (
            "mild nonproliferative diabetic retinopathy with macular edema due to underlying "
            "condition, right eye",
            ("E08.3211",),
        ),
        ("accidental poisoning by 1-propanol, initial encounter", ("T51.3X1A",)),
    ],
)
def test_release_code_sets(release: Graph, phrase: str, codes: tuple[str, ...]) -> None:
    assert Coder.from_graph(release).code(phrase).codes == codes


@pytest.mark.slow
def test_release_fracture_defaults(release: Graph) -> None:
    verifier = TabularVerifier(release)
    displaced = verifier.complete("S72.03", evidence("midcervical fracture of left femur"))
    assert displaced is not None
    assert displaced.code == "S72.032A"  # displaced (I.C.19.c.1), closed, initial assumed
    actions = {step.action for step in displaced.trace}
    assert {"displaced", "closed", "seventh_character"} <= actions
    assert ASSUMED in displaced.flags
    open_three = verifier.complete(
        "S72.301",
        evidence("open fracture type IIIB", ("episode", "initial"), ("fracture", "open")),
    )
    assert open_three is not None
    assert open_three.code == "S72.301C"
    malunion = verifier.complete(
        "S72.301", evidence("malunion", ("episode", "subsequent"), ("episode", "malunion"))
    )
    assert malunion is not None
    assert malunion.code == "S72.301P"
