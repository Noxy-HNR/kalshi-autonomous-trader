"""Identifier generation and correlation.

Two properties matter here:

1. **Idempotency.** Kalshi accepts a ``client_order_id``; reusing the same one
   for a retry is what makes an order submission safe to repeat. We therefore
   generate it once, at order construction, and never regenerate it on retry.
2. **Traceability.** Every decision, order, prediction and log line carries a
   correlation id so that a fill can be walked back to the market tick that
   caused it.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
import uuid

__all__ = [
    "correlation_id_var",
    "current_correlation_id",
    "new_client_order_id",
    "new_correlation_id",
    "new_decision_id",
    "new_id",
    "use_correlation_id",
]

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_counter_lock = threading.Lock()
_counter = 0

# Correlation id for the current async task / thread. Log records and audit rows
# read this automatically so call sites do not have to thread it through.
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kat_correlation_id", default=None
)


def _b36(value: int, width: int = 0) -> str:
    if value == 0:
        return "0".rjust(width, "0")
    out: list[str] = []
    while value:
        value, rem = divmod(value, 36)
        out.append(_ALPHABET[rem])
    return "".join(reversed(out)).rjust(width, "0")


def new_id(prefix: str) -> str:
    """Return a short, sortable, process-unique id such as ``ord_m3k9x2_0f``.

    The time component sorts lexicographically in creation order, which makes
    raw log greps and database scans behave sensibly without a join.
    """
    global _counter
    with _counter_lock:
        _counter = (_counter + 1) % 1_296  # 36^2
        seq = _counter
    millis = int(time.time() * 1000)
    return f"{prefix}_{_b36(millis)}_{_b36(seq, 2)}{_b36(os.getpid() % 1296, 2)}"


def new_correlation_id() -> str:
    """Id linking every artefact produced by one triggering event."""
    return new_id("cor")


def new_decision_id() -> str:
    """Id for one structured trading decision record."""
    return new_id("dec")


def new_client_order_id() -> str:
    """Idempotency key sent to Kalshi as ``client_order_id``.

    A UUID4 is used rather than the compact form because this value leaves our
    process and must not collide with anything, ever, including across restarts
    and across machines after the server migration.
    """
    return str(uuid.uuid4())


def current_correlation_id() -> str | None:
    return correlation_id_var.get()


class use_correlation_id:  # noqa: N801 - used as a context manager, reads as a verb
    """Bind a correlation id for the duration of a block.

    Usage::

        with use_correlation_id(cid):
            ...  # every log line and audit row inside carries cid
    """

    __slots__ = ("_token", "_value")

    def __init__(self, value: str | None = None) -> None:
        self._value = value or new_correlation_id()
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> str:
        self._token = correlation_id_var.set(self._value)
        return self._value

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            correlation_id_var.reset(self._token)
            self._token = None
