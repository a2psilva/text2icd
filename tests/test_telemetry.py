"""onnxruntime's telemetry stays off: importing text2icd sets ORT_DISABLE_TELEMETRY first."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# onnxruntime turns its telemetry off by itself on CI runners; without these, the test means
# the same on a CI runner as on a laptop.
CI = ("CI", "GITHUB_ACTIONS", "TF_BUILD", "GITLAB_CI", "CIRCLECI", "TRAVIS", "JENKINS_URL",
      "CODEBUILD_BUILD_ID", "BUILDKITE", "TEAMCITY_VERSION", "APPVEYOR",
      "BITBUCKET_BUILD_NUMBER", "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI")  # fmt: skip


def run(code: str, home: Path, **extra: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in CI and k != "ORT_DISABLE_TELEMETRY"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "HOME": str(home), **extra},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_onnxruntime_telemetry_never_starts(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    code = "import text2icd, os, onnxruntime; print(os.environ['ORT_DISABLE_TELEMETRY'])"
    assert run(code, tmp_path) == "1"
    assert not list(tmp_path.rglob("onnxruntime.db*"))  # its event store is never created
    assert not list(tmp_path.rglob("deviceid"))


def test_a_value_the_user_set_is_kept(tmp_path: Path) -> None:
    code = "import text2icd, os; print(os.environ['ORT_DISABLE_TELEMETRY'])"
    assert run(code, tmp_path, ORT_DISABLE_TELEMETRY="0") == "0"
