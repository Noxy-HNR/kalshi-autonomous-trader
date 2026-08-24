"""Logging call sites.

These exist because of a real crash. ``structlog``'s bound-logger methods take
the message as a parameter literally named ``event``, so a call site that passes
``event=`` as a keyword raises
``TypeError: BoundLogger.info() got multiple values for argument 'event'`` --
at *runtime*, on the line that logs, which in the case that prompted this was
inside application start-up and took the whole process down.

The collision is easy to introduce here because ``event`` is also the domain word
for one of the two things this platform trades, so it is a natural thing to want
to name a field. These tests make the mistake impossible to commit quietly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "packages" / "kat"
APPS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "apps"

LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}

#: Keywords structlog's own signature already uses.
RESERVED = {"event"}


def python_files() -> list[pathlib.Path]:
    files = list(PACKAGE_ROOT.rglob("*.py"))
    files += [p for p in APPS_ROOT.rglob("*.py") if "dashboard" not in p.parts]
    return files


def offending_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every ``log.<level>(..., event=...)`` call in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in LOG_METHODS:
            continue
        # Only calls on something plausibly a logger.
        target = node.func.value
        name = target.id if isinstance(target, ast.Name) else None
        if name not in {"log", "logger", "self"} and not isinstance(target, ast.Attribute):
            continue
        for keyword in node.keywords:
            if keyword.arg in RESERVED:
                found.append((node.lineno, keyword.arg))
    return found


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.name))
def test_no_log_call_uses_a_reserved_structlog_keyword(path: pathlib.Path):
    """``event=`` collides with structlog's own message parameter.

    Use ``event_contracts=`` for the domain meaning, which is what it almost
    always is in this codebase.
    """
    offenders = offending_calls(path)
    assert not offenders, (
        f"{path.name} passes a reserved structlog keyword to a log call at "
        f"line(s) {[line for line, _ in offenders]}; rename it "
        f"(e.g. event -> event_contracts)"
    )


def test_the_detector_actually_detects(tmp_path: pathlib.Path):
    """Guard against the test above silently passing because it finds nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text('log.info("something", event=1)\n', encoding="utf-8")
    assert offending_calls(bad) == [(1, "event")]

    good = tmp_path / "good.py"
    good.write_text('log.info("something", event_contracts=1)\n', encoding="utf-8")
    assert offending_calls(good) == []


def test_at_least_one_file_was_scanned():
    """A glob that matches nothing would make every test above vacuous."""
    assert len(python_files()) > 50
