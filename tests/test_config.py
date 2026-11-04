import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from text2icd.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("TEXT2ICD_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def test_defaults() -> None:
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.cache_dir == Path(".text2icd")
    assert settings.release is None
    assert settings.llm_provider == "none"
    assert settings.llm_api_key is None
    assert settings.require_dua_compliant_endpoint is True
    assert settings.llm_max_calls_per_phrase == 4
    assert settings.llm_budget_usd == 1.0
    assert (settings.llm_region, settings.llm_project) == (None, None)
    assert (settings.llm_timeout_s, settings.llm_max_retries) == (30.0, 2)


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT2ICD_DATA_DIR", "/srv/text2icd-data")
    monkeypatch.setenv("TEXT2ICD_RELEASE", "fy2026-apr")
    monkeypatch.setenv("TEXT2ICD_LLM_PROVIDER", " Bedrock ")
    monkeypatch.setenv("TEXT2ICD_REQUIRE_DUA_COMPLIANT_ENDPOINT", "false")
    settings = Settings()
    assert settings.data_dir == Path("/srv/text2icd-data")
    assert settings.release == "fy2026-apr"
    assert settings.llm_provider == "bedrock"
    assert settings.require_dua_compliant_endpoint is False


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TEXT2ICD_RELEASE=fy2027\nTEXT2ICD_LLM_MAX_CALLS_PER_PHRASE=2\n")
    settings = Settings()
    assert settings.release == "fy2027"
    assert settings.llm_max_calls_per_phrase == 2


def test_blank_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT2ICD_RELEASE", "")
    monkeypatch.setenv("TEXT2ICD_LLM_API_KEY", "  ")
    monkeypatch.setenv("TEXT2ICD_LLM_REGION", "")
    monkeypatch.setenv("TEXT2ICD_LLM_PROJECT", " ")
    settings = Settings()
    assert settings.release is None
    assert settings.llm_api_key is None
    assert (settings.llm_region, settings.llm_project) == (None, None)


@pytest.mark.parametrize("release", ["2026", "FY2026", "fy26", "fy2026-oct", "fy2026apr"])
def test_invalid_release_is_rejected(monkeypatch: pytest.MonkeyPatch, release: str) -> None:
    monkeypatch.setenv("TEXT2ICD_RELEASE", release)
    with pytest.raises(ValidationError, match="release must look like"):
        Settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [("TEXT2ICD_LLM_BUDGET_USD", "-0.01"), ("TEXT2ICD_LLM_MAX_CALLS_PER_PHRASE", "-1")],
)
def test_negative_limits_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_api_key_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT2ICD_LLM_API_KEY", "sk-test-not-a-real-key")
    settings = Settings()
    assert "sk-test-not-a-real-key" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-test-not-a-real-key"


def test_cms_raw_dir() -> None:
    assert Settings(data_dir=Path("d")).cms_raw_dir("fy2026") == Path("d/raw/cms/fy2026")


def test_env_example_lists_exactly_the_settings_fields() -> None:
    lines = (REPO_ROOT / ".env.example").read_text().splitlines()
    documented = {
        line.split("=", 1)[0].removeprefix("TEXT2ICD_").lower()
        for line in lines
        if line.startswith("TEXT2ICD_")
    }
    assert documented == set(Settings.model_fields)


def test_graph_path() -> None:
    assert Settings(data_dir=Path("d")).graph_path("fy2027") == Path(
        "d/processed/text2icd-graph-fy2027.sqlite"
    )
