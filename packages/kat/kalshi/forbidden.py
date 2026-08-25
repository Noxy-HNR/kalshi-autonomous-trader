"""Kalshi endpoints this platform will not call, ever.

The question this answers is "could this thing move my money?", and the honest
answer needs to be enforced rather than asserted.

What Kalshi's API actually permits
----------------------------------
Read from the published OpenAPI specifications (131 paths across
``openapi.yaml`` and ``perps_openapi.yaml``), not from memory:

* ``/portfolio/withdrawals`` and ``/portfolio/deposits`` are **GET only**. There
  is no POST, PUT or DELETE on either. An API key cannot initiate a withdrawal
  or a deposit; money leaves Kalshi through Kalshi's own interface, not through
  this or any other API client.
* Transfers that *are* writable -- ``/portfolio/subaccounts/transfer`` and
  ``/portfolio/intra_exchange_instance_transfer`` -- move funds between the
  account holder's own subaccounts and exchange instances. The money stays on
  Kalshi.
* ``POST /api_keys`` mints a new API key and ``DELETE /api_keys/{api_key}``
  removes one. Neither moves money, but a process that can mint credentials can
  outlive its own shutdown, which is its own kind of problem.

So the worst an API key can do is trade badly and shuffle balances between an
account's own pockets. That is a real risk and it is the one the risk engine and
the leverage invariant exist to bound. Extraction to a bank account is not on
the menu at all.

Why refuse them anyway
----------------------
Because "we don't call that endpoint" is a property of today's code, and this
platform is designed to modify itself: strategies are generated, models are
retrained, an LLM advises. A deny-list at the single chokepoint every Kalshi
request passes through turns absence into prohibition. A future change that
tried to call one of these fails loudly at the client instead of succeeding
quietly.

This is the same argument as :mod:`kat.risk.invariants`: the guarantee that
matters is the one that cannot be reached around.
"""

from __future__ import annotations

import re

from kat.common.errors import KatError

__all__ = ["ForbiddenEndpoint", "assert_endpoint_permitted", "is_forbidden"]


class ForbiddenEndpoint(KatError):
    """A call was attempted to an endpoint this platform refuses to use.

    Not retryable and not catchable into a fallback: reaching here means code
    tried to do something the platform is not permitted to do, and the correct
    outcome is a loud failure with the path named.
    """

    def __init__(self, method: str, path: str, reason: str) -> None:
        super().__init__(
            f"refusing to call {method} {path}: {reason}. "
            "This endpoint is on the permanent deny-list in kat.kalshi.forbidden; "
            "no configuration, strategy or model may reach it."
        )
        self.method = method
        self.path = path
        self.reason = reason
        # Never retried. A refusal is not a transient condition, and the retry
        # loop must not treat it as one.
        self.retryable = False


#: (pattern, reason). Matched against the request path, case-insensitively.
#:
#: Deliberately matched on the path rather than on a method, so a future
#: endpoint that gains a POST it does not have today is refused by default
#: rather than admitted by omission.
_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"/portfolio/withdrawals?", re.I),
        "withdrawals are not something an automated trader has any business touching",
    ),
    (
        re.compile(r"/portfolio/deposits?", re.I),
        "deposits are not something an automated trader has any business touching",
    ),
    (
        re.compile(r"/subaccounts", re.I),
        "moving balances between subaccounts is an account-administration action, "
        "not a trading one",
    ),
    (
        re.compile(r"/intra_exchange_instance_transfers?", re.I),
        "moving balances between exchange instances is an account-administration "
        "action, not a trading one",
    ),
    (
        re.compile(r"/api_keys", re.I),
        "a trading process must not be able to mint or revoke the credentials it "
        "runs on",
    ),
)


def is_forbidden(path: str) -> str | None:
    """The reason ``path`` is refused, or ``None`` if it is permitted."""
    for pattern, reason in _FORBIDDEN:
        if pattern.search(path):
            return reason
    return None


def assert_endpoint_permitted(method: str, path: str) -> None:
    """Raise :class:`ForbiddenEndpoint` if this call is not allowed.

    Called on every request, before authentication and before the rate limiter,
    so a refused call never even acquires a token or produces a signature.
    """
    reason = is_forbidden(path)
    if reason is not None:
        raise ForbiddenEndpoint(method, path, reason)
