from pathlib import Path

import pytest

from text2icd.ontology.files import ReleaseFiles


def test_paths_resolve(fixture_release: ReleaseFiles) -> None:
    assert fixture_release.tabular.name == "icd10cm_tabular_2099.xml"
    assert fixture_release.index.name == "icd10cm_index_2099.xml"
    assert fixture_release.eindex.name == "icd10cm_eindex_2099.xml"
    assert fixture_release.neoplasm.name == "icd10cm_neoplasm_2099.xml"
    assert fixture_release.drug.name == "icd10cm_drug_2099.xml"
    assert fixture_release.order.name == "icd10cm_order_2099.txt"


def test_order_addenda_file_is_not_mistaken_for_the_order_file(
    fixture_release: ReleaseFiles,
) -> None:
    (fixture_release.root / "order" / "icd10cm_order_addenda_2099.txt").write_text("")
    assert fixture_release.order.name == "icd10cm_order_2099.txt"


def test_missing_file_explains_how_to_fetch(tmp_path: Path) -> None:
    files = ReleaseFiles(tmp_path / "fy2099")
    with pytest.raises(
        FileNotFoundError, match=r"found none\); run `text2icd data fetch -r fy2099`"
    ):
        _ = files.tabular


def test_ambiguous_files_are_an_error(fixture_release: ReleaseFiles) -> None:
    extra = fixture_release.root / "tables-index" / "icd10cm_tabular_2098.xml"
    extra.write_text("<ICD10CM.tabular/>")
    with pytest.raises(
        FileNotFoundError, match=r"icd10cm_tabular_2098\.xml, icd10cm_tabular_2099\.xml"
    ):
        _ = fixture_release.tabular
