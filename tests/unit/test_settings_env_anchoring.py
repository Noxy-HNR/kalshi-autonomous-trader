"""Configuration must not depend on where the process was started.

A bare ``env_file=".env"`` resolves against the current working directory, so a
process launched from anywhere else read *no configuration at all* -- and every
value it fell back to was a plausible one: environment paper, market data demo,
credentials absent. Nothing errors, nothing warns, and a process reading
production with real keys is indistinguishable from one reading nothing.

That is how the launcher came to health-check 127.0.0.1 while the engine was
bound to a tailnet address: its probe ran python from the parent directory, got
defaults back, and concluded remote access was off.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from kat.config.settings import REPO_ROOT, AppSettings


def test_the_env_file_is_an_absolute_path_under_the_repo() -> None:
    env_file = AppSettings.model_config.get("env_file")
    assert env_file is not None
    path = Path(str(env_file))
    assert path.is_absolute(), "a relative env_file resolves against the working directory"
    assert path.parent == REPO_ROOT
    assert path.name == ".env"


def test_settings_load_identically_from_an_unrelated_directory(tmp_path: Path) -> None:
    """The behaviour the anchoring exists to guarantee.

    Run as a subprocess because the working directory is process-global and the
    settings cache is not: importing once and chdir-ing would prove nothing.
    """
    code = (
        "from kat.config.settings import AppSettings;"
        "s = AppSettings();"
        "print(s.environment.value, s.api_remote_access, s.kalshi.reads_production)"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "packages")}

    from_repo = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]
    from_elsewhere = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]

    assert from_repo == from_elsewhere, (
        f"settings differ by working directory: repo={from_repo!r} "
        f"elsewhere={from_elsewhere!r}"
    )
