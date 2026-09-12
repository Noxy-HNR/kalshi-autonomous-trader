"""Exceptions reach the log, and reach it redacted.

These exist because every traceback was being thrown away. ``exc_info=True``
is a request to read ``sys.exc_info()`` when the record is *rendered*, and the
file handler renders in a ``QueueListener`` thread -- a different thread,
moments later, where there is no exception at all. So the JSON line held the
literal ``"exc_info": true`` and nothing else: four collector crashes in
``logs/polymarket.log`` were recorded with no cause, and nothing could say why
the collector had died.

Rendering the traceback also creates text that did not exist when the
redaction processor ran, so the secret test is here too: an exception message
that quotes a signature must not become the one place a signature is written
down in full.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kat.observability import logging as kat_logging

# Long base64 containing non-hex characters: the shape the redactor masks.
TOKEN = (
    "dGhpcyBpcyBhIHZlcnkgbG9uZyBiYXNlNjQgc2lnbmF0dXJlIHRoYXQgc2hvdWxk"
    "IGJlIG1hc2tlZA=="
)


def _cross_the_book() -> None:
    raise ValueError("book for KX-TEST was crossed")


def _reject_signature() -> None:
    raise PermissionError(f"signature rejected: {TOKEN}")


def _restore_quiet_logging() -> None:
    """Put back what the autouse fixture in conftest installed."""
    kat_logging.shutdown_logging()
    kat_logging.configure_logging("ERROR")
    logging.getLogger().setLevel(logging.ERROR)


def _named(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    # structlog records carry ``event``; stdlib records are renamed to ``message``.
    found = [r for r in records if name in (r.get("event"), r.get("message"))]
    assert len(found) == 1, records
    return found[0]


@pytest.fixture
def file_log(tmp_path: Path) -> Iterator[Callable[[], list[dict[str, Any]]]]:
    """File logging into ``tmp_path``. Calling the result drains and parses it.

    Draining means stopping the listener: ``QueueListener.stop`` processes
    everything already queued before its thread exits, so the file is complete
    when it returns. Reading earlier would race the background writer.
    """
    kat_logging.shutdown_logging()
    with contextlib.redirect_stdout(io.StringIO()):
        kat_logging.configure_logging("INFO", log_dir=tmp_path, filename="test.log")
    listener = kat_logging._listener
    assert listener is not None
    handlers = list(listener.handlers)

    def drain() -> list[dict[str, Any]]:
        kat_logging.shutdown_logging()
        for handler in handlers:
            handler.close()
        text = (tmp_path / "test.log").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    try:
        yield drain
    finally:
        for handler in handlers:
            handler.close()
        _restore_quiet_logging()


class TestTracebacksReachTheFile:
    def test_log_exception_records_the_type_message_and_frames(self, file_log):
        log = kat_logging.get_logger("kat.test.tracebacks")
        try:
            _cross_the_book()
        except ValueError:
            log.exception("test.crossed", ticker="KX-TEST")

        record = _named(file_log(), "test.crossed")
        exception = record["exception"]
        assert "Traceback (most recent call last)" in exception
        assert "in _cross_the_book" in exception
        assert "ValueError: book for KX-TEST was crossed" in exception
        # What the incident actually logged: a flag, and no exception.
        assert "exc_info" not in record
        assert record["ticker"] == "KX-TEST"

    def test_exc_info_true_on_an_error_call_is_captured_too(self, file_log):
        """The shape ``trading.intent_source_failed`` uses."""
        log = kat_logging.get_logger("kat.test.tracebacks")
        try:
            _cross_the_book()
        except ValueError as exc:
            log.error("test.source_failed", error=str(exc), exc_info=True)

        record = _named(file_log(), "test.source_failed")
        assert "ValueError: book for KX-TEST was crossed" in record["exception"]
        assert "in _cross_the_book" in record["exception"]

    def test_a_stdlib_logger_exception_reaches_the_file(self, file_log):
        try:
            _cross_the_book()
        except ValueError:
            logging.getLogger("kat.test.stdlib").exception("stdlib crossed")

        record = _named(file_log(), "stdlib crossed")
        assert "ValueError: book for KX-TEST was crossed" in record["exception"]
        assert "in _cross_the_book" in record["exception"]

    @pytest.mark.parametrize("origin", ["structlog", "stdlib"])
    def test_a_secret_in_the_exception_text_is_redacted(self, file_log, tmp_path, origin):
        try:
            _reject_signature()
        except PermissionError:
            if origin == "structlog":
                kat_logging.get_logger("kat.test.secret").exception("test.rejected")
            else:
                logging.getLogger("kat.test.secret").exception("test.rejected")

        record = _named(file_log(), "test.rejected")
        assert "PermissionError: signature rejected: <redacted>" in record["exception"]
        raw = (tmp_path / "test.log").read_text(encoding="utf-8")
        assert TOKEN[:24] not in raw

    def test_a_record_without_exc_info_gains_nothing(self, file_log):
        """Even one written from inside an ``except`` block.

        A processor that attached the ambient exception whenever there was one
        would put a traceback on every line a handler happens to write, and a
        caught-and-handled exception would read in the log like a crash.
        """
        log = kat_logging.get_logger("kat.test.plain")
        try:
            _cross_the_book()
        except ValueError:
            log.warning("test.handled", outcome="skipped")
            logging.getLogger("kat.test.plain").warning("stdlib handled")
        # Asking for an exception when none is active has nothing to render.
        log.error("test.nothing_active", exc_info=True)

        records = file_log()
        for name in ("test.handled", "stdlib handled", "test.nothing_active"):
            record = _named(records, name)
            assert "exception" not in record, record
            assert "exc_info" not in record, record


class TestConsole:
    @pytest.mark.parametrize("json_console", [False, True])
    def test_the_console_shows_the_exception_redacted_without_warnings(self, json_console):
        buffer = io.StringIO()
        kat_logging.shutdown_logging()
        try:
            with contextlib.redirect_stdout(buffer):
                kat_logging.configure_logging("INFO", json_console=json_console)
            log = kat_logging.get_logger("kat.test.console")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    _reject_signature()
                except PermissionError:
                    log.exception("test.console_rejected")
        finally:
            _restore_quiet_logging()

        out = buffer.getvalue()
        assert "test.console_rejected" in out
        assert "in _reject_signature" in out
        assert "PermissionError: signature rejected: <redacted>" in out
        assert TOKEN[:24] not in out
        # structlog's ConsoleRenderer warns on every call when it is handed an
        # already-formatted ``exception`` instead of ``exc_info``.
        assert not [w for w in caught if "format_exc_info" in str(w.message)]
