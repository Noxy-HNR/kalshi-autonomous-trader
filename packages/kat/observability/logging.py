"""Structured logging.

Three properties this configuration guarantees:

1. **Structured.** Every record is JSON with a stable field set, so logs can be
   grepped, shipped and joined against database rows without parsing prose.
2. **Correlated.** The active correlation id (see :mod:`kat.common.ids`) is
   attached automatically, so one market tick, the decision it produced, the
   order that followed and the fill that came back all share a key.
3. **Non-blocking on the trading path.** File writes go through a
   ``QueueHandler`` drained by a background thread. A slow or stalled disk
   delays log *persistence*, never order submission.

Secrets are redacted by a processor rather than by discipline at call sites,
because discipline fails and a processor does not.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import logging.handlers
import queue
import re
import sys
from pathlib import Path
from typing import Any

import structlog

from kat.common.ids import current_correlation_id

__all__ = ["configure_logging", "get_logger", "shutdown_logging"]

_configured = False
_listener: logging.handlers.QueueListener | None = None

# Keys whose values are replaced wholesale, whatever they contain.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "api_key_id",
        "authorization",
        "kalshi-access-key",
        "kalshi-access-signature",
        "password",
        "private_key",
        "private_key_path",
        "secret",
        "signature",
        "token",
    }
)

# Values that look like credentials even under an innocent key name.
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b"),  # long base64: signatures, tokens
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                return "<redacted>"
    return value


def _redact(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip credentials from every record before it is rendered."""
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS:
            event_dict[key] = "<redacted>"
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def _add_correlation(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach the ambient correlation id when the call site did not supply one."""
    if "correlation_id" not in event_dict:
        cid = current_correlation_id()
        if cid is not None:
            event_dict["correlation_id"] = cid
    return event_dict


def _rename_event(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Use ``message`` rather than structlog's ``event`` key.

    ``event`` collides with the domain meaning of "event contract" everywhere
    else in this codebase, and that collision is genuinely confusing in a log
    search.
    """
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def configure_logging(
    level: str = "INFO",
    *,
    log_dir: Path | str | None = None,
    json_console: bool = False,
    filename: str = "kat.log",
) -> None:
    """Install the logging configuration. Idempotent.

    Parameters
    ----------
    level:
        Root log level.
    log_dir:
        Directory for the rotating JSON log. ``None`` disables file logging,
        which is what tests want.
    json_console:
        JSON on stdout instead of the human-readable console renderer. Use this
        when running as a Windows service, where stdout is captured to a file.
    """
    global _configured, _listener
    if _configured:
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_correlation,
        _redact,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console_renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_console
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=console_renderer,
            foreign_pre_chain=shared_processors,
        )
    )
    root.addHandler(console)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / filename,
            maxBytes=64 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=[*shared_processors, _rename_event],
            )
        )
        # Hand file I/O to a background thread so a stalled disk cannot block
        # an order submission that happens to log on its way out.
        log_queue: queue.Queue[Any] = queue.Queue(maxsize=50_000)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root.addHandler(queue_handler)
        _listener = logging.handlers.QueueListener(
            log_queue, file_handler, respect_handler_level=True
        )
        _listener.start()
        atexit.register(shutdown_logging)

    # Third-party loggers are noisy at INFO and tell us nothing we act on.
    for noisy in ("httpx", "httpcore", "websockets", "asyncio", "urllib3", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def shutdown_logging() -> None:
    """Drain and stop the background log writer."""
    global _listener, _configured
    if _listener is not None:
        # Shutdown must never raise: a failure to stop the log writer cannot
        # be allowed to mask whatever we were shutting down for.
        with contextlib.suppress(Exception):
            _listener.stop()
        _listener = None
    _configured = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``.

    Prefer module-level ``log = get_logger(__name__)`` so the ``logger`` field
    identifies the component without a manual string.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
