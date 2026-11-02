"""Runtime settings, read from `TEXT2ICD_*` environment variables and an optional `.env` file."""

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RELEASE_ID = re.compile(r"^fy\d{4}(-apr)?$")


class Settings(BaseSettings):
    """All configuration in one place. Every field maps to `TEXT2ICD_<FIELD_NAME>`."""

    model_config = SettingsConfigDict(
        env_prefix="TEXT2ICD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".text2icd")
    release: str | None = Field(
        default=None,
        description="Release id such as fy2026 or fy2026-apr; None selects the newest one.",
    )

    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    require_dua_compliant_endpoint: bool = True
    llm_max_calls_per_phrase: int = Field(default=4, ge=0)
    llm_budget_usd: float = Field(default=1.0, ge=0)
    # The 7th character's episode of care when an injury phrase states none.
    default_episode: Literal["initial", "subsequent", "sequela"] = "initial"

    @field_validator("release", "llm_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("release")
    @classmethod
    def _check_release(cls, value: str | None) -> str | None:
        if value is not None and not RELEASE_ID.fullmatch(value):
            raise ValueError(f"release must look like 'fy2026' or 'fy2026-apr', got {value!r}")
        return value

    @field_validator("llm_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.strip().lower() or "none"

    def cms_raw_dir(self, release: str) -> Path:
        """Directory holding the unmodified CMS files of one release."""
        return self.data_dir / "raw" / "cms" / release

    def eval_dir(self) -> Path:
        """Local evaluation datasets and result files (git-ignored, with the rest of data/)."""
        return self.data_dir / "processed" / "eval"

    def graph_path(self, release: str) -> Path:
        """The knowledge-graph file built from one release by `text2icd build-graph`."""
        return self.data_dir / "processed" / f"text2icd-graph-{release}.sqlite"
