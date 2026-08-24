"""Token-bucket rate limiting matching Kalshi's documented mechanics.

Kalshi meters requests in tokens, not request counts:

* Two independent budgets, **Read** and **Write**. Write covers order placement,
  amends, cancels and order groups; Read covers everything else.
* Perps are metered in *their own* Read and Write buckets, so there are four
  buckets in total and perp traffic never starves event-contract traffic.
* Most endpoints cost 10 tokens; a documented minority cost more or less.
  ``GET /account/endpoint_costs`` is authoritative and is loaded at start-up.
* Buckets refill continuously and hold one or two seconds of budget, which is
  what allows an idle client to burst.
* A 429 carries no penalty and no ``Retry-After``; the bucket simply keeps
  refilling.

We deliberately spend only a configured fraction of the published budget. The
headroom exists so that a burst of *order* activity is never blocked by
background market-data polling that happens to be draining the same bucket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from kat.observability.logging import get_logger
from kat.observability.metrics import M_RATE_LIMITED, metrics

log = get_logger(__name__)

__all__ = ["BucketKind", "RateLimiter", "TokenBucket", "DEFAULT_ENDPOINT_COST"]

#: Documented default cost for an endpoint with no override.
DEFAULT_ENDPOINT_COST = 10

#: Non-default costs documented at the time of writing. The authoritative list
#: is fetched from ``GET /account/endpoint_costs`` at start-up and replaces
#: these; they are only a sensible starting point before that call completes.
KNOWN_ENDPOINT_COSTS: dict[str, int] = {
    "GET /margin/balance": 5,
    "GET /margin/balance?compute_available_balance=true": 50,
    "DELETE /portfolio/orders": 2,  # cancels are cheaper than creates
}


class BucketKind:
    EVENT_READ = "event_read"
    EVENT_WRITE = "event_write"
    PERP_READ = "perp_read"
    PERP_WRITE = "perp_write"


@dataclass
class TokenBucket:
    """Continuously refilling token bucket.

    ``capacity`` above ``refill_per_second`` is what models Kalshi's burst
    allowance: two seconds of budget for Write above the Basic tier, one second
    otherwise.
    """

    refill_per_second: float
    capacity: float
    tokens: float = field(init=False)
    updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.updated_at = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated_at
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.updated_at = now

    def try_consume(self, cost: float) -> bool:
        self._refill(time.monotonic())
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def wait_time(self, cost: float) -> float:
        """Seconds until ``cost`` tokens will be available. 0 when available now."""
        self._refill(time.monotonic())
        if self.tokens >= cost:
            return 0.0
        if self.refill_per_second <= 0:
            return float("inf")
        return (cost - self.tokens) / self.refill_per_second

    def reconfigure(self, refill_per_second: float, capacity: float) -> None:
        """Adopt the real tier budget once ``/account/limits`` has been read.

        The current token balance is preserved (clamped to the new capacity)
        rather than reset, so a mid-flight burst is not accidentally granted a
        second full bucket.
        """
        self._refill(time.monotonic())
        self.refill_per_second = refill_per_second
        self.capacity = capacity
        self.tokens = min(self.tokens, capacity)


class RateLimiter:
    """Four buckets, an async acquire, and a cost table.

    ``acquire`` waits rather than raising: at Kalshi's refill rates the wait for
    a 10-token request is on the order of 10-50ms, and waiting is strictly
    better than a 429 round-trip. Callers that cannot tolerate a wait pass
    ``max_wait_s`` and handle the ``False`` return.
    """

    def __init__(
        self,
        *,
        event_read_per_s: int = 200,
        event_write_per_s: int = 100,
        perp_read_per_s: int = 200,
        perp_write_per_s: int = 100,
        utilisation: float = 0.80,
        write_burst_multiplier: float = 2.0,
    ) -> None:
        self._utilisation = max(0.05, min(0.95, utilisation))
        self._buckets: dict[str, TokenBucket] = {
            BucketKind.EVENT_READ: self._make(event_read_per_s, 1.0),
            BucketKind.EVENT_WRITE: self._make(event_write_per_s, write_burst_multiplier),
            BucketKind.PERP_READ: self._make(perp_read_per_s, 1.0),
            BucketKind.PERP_WRITE: self._make(perp_write_per_s, write_burst_multiplier),
        }
        self._locks = {name: asyncio.Lock() for name in self._buckets}
        self._costs: dict[str, int] = dict(KNOWN_ENDPOINT_COSTS)

    def _make(self, per_second: int, burst: float) -> TokenBucket:
        rate = per_second * self._utilisation
        return TokenBucket(refill_per_second=rate, capacity=rate * burst)

    # -- configuration -------------------------------------------------------

    def apply_tier_limits(
        self,
        bucket: str,
        refill_rate: int,
        bucket_capacity: int,
    ) -> None:
        """Adopt real limits from ``GET /account/limits`` (or its perps twin)."""
        b = self._buckets.get(bucket)
        if b is None:
            return
        b.reconfigure(refill_rate * self._utilisation, bucket_capacity * self._utilisation)
        log.info(
            "ratelimit.tier_applied",
            bucket=bucket,
            refill_per_s=round(b.refill_per_second, 1),
            capacity=round(b.capacity, 1),
        )

    def set_endpoint_costs(self, costs: dict[str, int]) -> None:
        """Replace the cost table with the authoritative one from the exchange."""
        self._costs.update(costs)
        log.info("ratelimit.costs_loaded", count=len(costs))

    def cost_for(self, method: str, path: str) -> int:
        return self._costs.get(f"{method.upper()} {path}", DEFAULT_ENDPOINT_COST)

    # -- consumption ---------------------------------------------------------

    @staticmethod
    def bucket_for(*, is_perp: bool, is_write: bool) -> str:
        if is_perp:
            return BucketKind.PERP_WRITE if is_write else BucketKind.PERP_READ
        return BucketKind.EVENT_WRITE if is_write else BucketKind.EVENT_READ

    async def acquire(
        self,
        bucket: str,
        cost: int = DEFAULT_ENDPOINT_COST,
        *,
        max_wait_s: float | None = None,
    ) -> bool:
        """Consume ``cost`` tokens, waiting if necessary.

        Returns ``True`` when the tokens were consumed and ``False`` when
        ``max_wait_s`` would have been exceeded (nothing is consumed then).
        """
        b = self._buckets[bucket]
        lock = self._locks[bucket]
        deadline = None if max_wait_s is None else time.monotonic() + max_wait_s

        while True:
            async with lock:
                if b.try_consume(cost):
                    return True
                wait = b.wait_time(cost)

            if deadline is not None and time.monotonic() + wait > deadline:
                metrics.increment("ratelimit.declined", bucket=bucket)
                return False

            metrics.increment("ratelimit.waits", bucket=bucket)
            # Cap a single sleep so a reconfiguration mid-wait is picked up.
            await asyncio.sleep(min(wait, 0.25) if wait > 0 else 0.001)

    def note_429(self, bucket: str) -> None:
        """Record an actual 429 and drain the bucket to resynchronise.

        Receiving a 429 means our local accounting was more optimistic than the
        exchange's. Zeroing the bucket forces a full refill interval before the
        next attempt, which converges on the true budget instead of hammering.
        """
        metrics.increment(M_RATE_LIMITED, bucket=bucket)
        b = self._buckets.get(bucket)
        if b is not None:
            b.tokens = 0.0
            b.updated_at = time.monotonic()
        log.warning("ratelimit.429_received", bucket=bucket)

    def snapshot(self) -> dict[str, dict[str, float]]:
        now = time.monotonic()
        out: dict[str, dict[str, float]] = {}
        for name, b in self._buckets.items():
            b._refill(now)  # noqa: SLF001 - same module, deliberate
            out[name] = {
                "tokens": round(b.tokens, 1),
                "capacity": round(b.capacity, 1),
                "refill_per_s": round(b.refill_per_second, 1),
                "utilisation_pct": round(100 * (1 - b.tokens / b.capacity), 1) if b.capacity else 0,
            }
        return out
