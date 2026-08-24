"""Exception hierarchy.

The distinction that matters operationally is between errors that make it
*unsafe to keep trading* (``UnsafeStateError`` and subclasses) and errors that
are merely a failed operation. The engine halts on the former and retries or
skips on the latter.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "DataQualityError",
    "ExchangeError",
    "IdempotencyError",
    "KatError",
    "KillSwitchEngaged",
    "LeverageViolation",
    "RateLimited",
    "ReconciliationError",
    "RiskRejection",
    "StaleDataError",
    "UnsafeStateError",
]


class KatError(Exception):
    """Base class for everything this platform raises deliberately."""


class ConfigError(KatError):
    """Configuration is missing, malformed, or attempts something forbidden."""


class ExchangeError(KatError):
    """Kalshi returned an error, or returned something we cannot parse.

    ``status`` is the HTTP status where one exists. ``retryable`` is decided by
    the client, never by the caller, because whether a *non-idempotent* order
    submission may be retried depends on how far it got.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retryable: bool = False,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.body = body


class RateLimited(ExchangeError):
    """HTTP 429. Always retryable after backoff; carries no penalty at Kalshi."""

    def __init__(self, message: str = "rate limited", *, bucket: str | None = None) -> None:
        super().__init__(message, status=429, retryable=True)
        self.bucket = bucket


class IdempotencyError(KatError):
    """A non-idempotent submission was about to be repeated unsafely."""


class RiskRejection(KatError):
    """The local risk engine refused an order.

    This is a normal, expected outcome -- most candidate orders are rejected --
    so it carries structured fields for the audit trail rather than just a
    message.
    """

    def __init__(self, reason_code: str, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.detail = detail or {}


class LeverageViolation(RiskRejection):
    """An order would push perpetual exposure above the 1.00x hard maximum.

    Deliberately a distinct type: it is logged at a higher severity, surfaced on
    its own dashboard panel, and is never swallowed by a generic risk handler.
    """

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__("LEVERAGE_LIMIT", message, detail=detail)


class KillSwitchEngaged(RiskRejection):
    """A kill switch covering this order is engaged."""

    def __init__(self, scope: str, message: str) -> None:
        super().__init__("KILL_SWITCH", message, detail={"scope": scope})
        self.scope = scope


class UnsafeStateError(KatError):
    """The platform cannot prove its state is correct, so it must not trade."""


class ReconciliationError(UnsafeStateError):
    """Local state and exchange state disagree and could not be resolved."""


class StaleDataError(UnsafeStateError):
    """Market data is older than the freshness budget for a trading decision."""


class DataQualityError(KatError):
    """An ingested record failed validation and was quarantined."""
