"""Runtime settings, read from `TEXT2ICD_*` environment variables and an optional `.env` file.

Where files go, unless `TEXT2ICD_DATA_DIR` and `TEXT2ICD_CACHE_DIR` (or `--data-dir`) say
otherwise: the per-user directories of the platform (`default_data_dir`, `default_cache_dir`),
so that an installed text2icd finds its graph from any working directory. A checkout's `.env`
(`make install` copies `.env.example`) keeps them in `./data` and `./.text2icd`.
"""

import re
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RELEASE_ID = re.compile(r"^fy\d{4}(-apr)?$")
APP = "text2icd"


def default_data_dir() -> Path:
    """Graphs, CMS files and local datasets: `~/Library/Application Support/text2icd` on macOS,
    `~/.local/share/text2icd` on Linux (`$XDG_DATA_HOME`), `%LOCALAPPDATA%\\text2icd` on
    Windows."""
    return platformdirs.user_data_path(APP, appauthor=False)


def default_cache_dir() -> Path:
    """The LLM decision cache: `~/Library/Caches/text2icd` on macOS, `~/.cache/text2icd` on
    Linux (`$XDG_CACHE_HOME`), `%LOCALAPPDATA%\\text2icd\\Cache` on Windows."""
    return platformdirs.user_cache_path(APP, appauthor=False)


class Settings(BaseSettings):
    """All configuration in one place. Every field maps to `TEXT2ICD_<FIELD_NAME>`."""

    model_config = SettingsConfigDict(
        env_prefix="TEXT2ICD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default_factory=default_data_dir)
    cache_dir: Path = Field(default_factory=default_cache_dir)
    release: str | None = Field(
        default=None,
        description="Release id such as fy2026 or fy2026-apr; None selects the newest one.",
    )

    llm_provider: str = "none"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    # The AWS region (bedrock) or Google Cloud region (vertex), and the Google Cloud project;
    # None leaves them to the platform's own settings (AWS_REGION, CLOUD_ML_REGION).
    llm_region: str | None = None
    llm_project: str | None = None
    require_dua_compliant_endpoint: bool = True
    llm_max_calls_per_phrase: int = Field(default=4, ge=0)
    llm_budget_usd: float = Field(default=1.0, ge=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)  # per attempt
    llm_max_retries: int = Field(default=2, ge=0)  # of retryable errors only (429, 5xx, network)
    # The 7th character's episode of care when an injury phrase states none.
    default_episode: Literal["initial", "subsequent", "sequela"] = "initial"
    # A confidence calibration table (`text2icd eval calibrate`); None uses the shipped one.
    calibration: Path | None = None
    # Logs show a phrase only as its length and a salted hash, unless this is true (task 9.7).
    # Leave it off whenever inputs may come from restricted data such as MIMIC-IV.
    log_inputs: bool = False

    @field_validator(
        "release", "llm_api_key", "llm_region", "llm_project", "calibration", mode="before"
    )
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
