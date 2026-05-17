"""Project command entry points for uv run."""

from __future__ import annotations

import subprocess
import sys


def _run(command: list[str]) -> int:
    """Run a subprocess command and return its exit code."""
    return subprocess.call(command)


def test() -> None:
    """Run the full unittest suite."""
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        *sys.argv[1:],
    ]
    raise SystemExit(_run(command))


def streamlit() -> None:
    """Launch the Streamlit dashboard."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        *sys.argv[1:],
    ]
    raise SystemExit(_run(command))
