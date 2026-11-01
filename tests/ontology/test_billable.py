from collections.abc import Callable
from pathlib import Path

import pytest

from text2icd.ontology.billable import generate_billable
from text2icd.ontology.files import ReleaseFiles
from text2icd.ontology.order import billable_codes, parse_order_file
from text2icd.ontology.tabular import parse_tabular

FIXTURES = Path(__file__).parents[1] / "fixtures" / "cms"


def test_sample_billable_set() -> None:
    billable = generate_billable(parse_tabular(FIXTURES / "tabular_sample.xml"))
    retinopathy = {f"E08.32{sixth}{eye}" for sixth in "19" for eye in "1239"}
    toxic = {f"T51.3X{intent}{encounter}" for intent in "1234" for encounter in "ADS"}
    assert billable == retinopathy | toxic | {"E11.22"}


def test_sample_agrees_with_the_order_file_sample() -> None:
    generated = generate_billable(parse_tabular(FIXTURES / "tabular_sample.xml"))
    official = billable_codes(parse_order_file(FIXTURES / "order_sample.txt"))
    shared = {"E11.22", "T51.3X1A"}
    assert shared <= generated
    assert shared <= official
    assert "T51.3X1" not in generated


S06_NOTE = (
    "7th characters D and S do not apply to codes in category S06 with 6th character 7 - death due "
    "to brain injury prior to regaining consciousness, or 8 - death due to other cause prior to "
    "regaining consciousness."
)


def s06_tabular(tmp_path: Path, note: str) -> Path:
    path = tmp_path / "tabular.xml"
    path.write_text(
        "<ICD10CM.tabular><version>2099</version><chapter><name>19</name><desc>Injury</desc>"
        "<section id='S00-S09'><desc>Head</desc>"
        "<diag><name>S06</name><desc>Intracranial injury</desc>"
        f"<notes><note>{note}</note></notes>"
        "<sevenChrDef><extension char='A'>initial encounter</extension>"
        "<extension char='D'>subsequent encounter</extension>"
        "<extension char='S'>sequela</extension></sevenChrDef>"
        "<diag><name>S06.1</name><desc>Traumatic cerebral edema</desc>"
        "<diag placeholder='true'><name>S06.1X</name><desc>Traumatic cerebral edema</desc>"
        "<diag><name>S06.1X1</name><desc>with loss of consciousness of 30 minutes</desc></diag>"
        "<diag><name>S06.1X7</name><desc>with death due to brain injury</desc></diag>"
        "</diag></diag></diag></section></chapter></ICD10CM.tabular>"
    )
    return path


def test_s06_death_codes_take_only_the_initial_encounter(tmp_path: Path) -> None:
    billable = generate_billable(parse_tabular(s06_tabular(tmp_path, S06_NOTE)))
    assert billable == {"S06.1X1A", "S06.1X1D", "S06.1X1S", "S06.1X7A"}


def test_exception_fails_loudly_if_its_note_disappears(tmp_path: Path) -> None:
    tabular = parse_tabular(s06_tabular(tmp_path, "Code also any associated open wound."))
    with pytest.raises(ValueError, match="review SEVENTH_CHARACTER_EXCEPTIONS"):
        generate_billable(tabular)


@pytest.mark.slow
@pytest.mark.parametrize("release_id", ["fy2026", "fy2026-apr", "fy2027"])
def test_generated_set_equals_the_order_file(
    cms_release: Callable[[str], ReleaseFiles], release_id: str
) -> None:
    files = cms_release(release_id)
    generated = generate_billable(parse_tabular(files.tabular))
    official = billable_codes(parse_order_file(files.order))
    assert sorted(official - generated) == []
    assert sorted(generated - official) == []
