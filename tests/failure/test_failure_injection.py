"""Failure injection.

Every test here breaks something on purpose and asserts the platform fails in the
*safe* direction. The recurring theme: when the system cannot prove a thing is
true, it must refuse to trade rather than assume the optimistic case.

The specific failures modelled are the ones that actually happen to a trading
process running unattended for weeks: the exchange times out mid-order, the
websocket drops a message, the database goes away, the clock jumps, a duplicate
fill arrives after a reconnect, and the machine is killed without warning.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from kat.common.enums import (
    BookSide,
    MarketType,
    OrderState,
    OutcomeSide,
    RiskDecision,
    TradingEnvironment,
)
from kat.common.errors import ExchangeError, RateLimited
from kat.db.models import AuditLog, Quote
from kat.db.writer import AsyncWriter
from kat.execution.order import ManagedOrder
from kat.kalshi.http import AmbiguousSubmission, KalshiHTTPClient
from kat.kalshi.ratelimit import RateLimiter
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.risk.perp_risk import PerpAccountState, PerpPositionState
from kat.risk.types import OrderIntent

NOW = 1_700_000_000_000


# ---------------------------------------------------------------------------
# Exchange API failures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def throwaway_signer():
    """A real RSA signer over a throwaway key.

    Order submission is an authenticated path, so a client with no signer fails
    on auth before the transport is ever reached -- which would test the wrong
    thing. The key never leaves this process.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from kat.kalshi.auth import KalshiSigner

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return KalshiSigner.from_pem("test-key-id", pem, label="test")


def make_client(handler, *, is_perp: bool = False, signer=None) -> KalshiHTTPClient:
    """An HTTP client whose transport is a scripted failure."""
    client = KalshiHTTPClient(
        base_url="https://example.invalid/trade-api/v2",
        signer=signer,
        rate_limiter=RateLimiter(),
        is_perp=is_perp,
        max_retries=2,
    )
    client._client = httpx.AsyncClient(
        base_url="https://example.invalid/trade-api/v2",
        transport=httpx.MockTransport(handler),
    )
    return client


class TestAmbiguousSubmission:
    """A submission whose outcome was never observed must never be retried."""

    async def test_a_read_timeout_on_an_order_is_never_retried(self, throwaway_signer):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("timed out", request=request)

        client = make_client(handler, signer=throwaway_signer)
        try:
            with pytest.raises(AmbiguousSubmission) as exc:
                await client.request(
                    "POST", "/portfolio/orders", json_body={}, is_write=True,
                    idempotent=False, client_order_id="cli-1",
                )
            assert attempts == 1, "the order was retried after an ambiguous timeout"
            assert exc.value.client_order_id == "cli-1"
        finally:
            await client.aclose()

    async def test_a_gateway_error_on_an_order_is_never_retried(self, throwaway_signer):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503, json={"error": "unavailable"})

        client = make_client(handler, signer=throwaway_signer)
        try:
            with pytest.raises(AmbiguousSubmission):
                await client.request(
                    "POST", "/portfolio/orders", json_body={}, is_write=True,
                    idempotent=False, client_order_id="cli-2",
                )
            assert attempts == 1
        finally:
            await client.aclose()

    async def test_a_connect_error_is_retried_because_nothing_was_sent(
        self, throwaway_signer
    ):
        """The request never reached the exchange, so repeating it is safe."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("refused", request=request)

        client = make_client(handler, signer=throwaway_signer)
        try:
            with pytest.raises(ExchangeError):
                await client.request(
                    "POST", "/portfolio/orders", json_body={}, is_write=True,
                    idempotent=False, client_order_id="cli-3",
                )
            assert attempts > 1, "a connect error should be retried"
        finally:
            await client.aclose()

    async def test_a_read_timeout_on_a_get_is_retried(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler)
        try:
            assert await client.get("/markets", authenticated=False) == {"ok": True}
            assert attempts == 2
        finally:
            await client.aclose()


class TestRateLimiting:
    async def test_a_429_is_retried_and_then_succeeds(self):
        """Kalshi documents a 429 as rejected before execution, so it is safe."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, json={"error": "too many requests"})
            return httpx.Response(200, json={"ok": True})

        client = make_client(handler)
        try:
            assert await client.get("/markets", authenticated=False) == {"ok": True}
            assert attempts == 2
        finally:
            await client.aclose()

    async def test_the_local_limiter_can_decline_rather_than_wait(self):
        limiter = RateLimiter(event_read_per_s=1, utilisation=0.05)
        # Drain whatever the bucket starts with.
        while await limiter.acquire("event_read", 10, max_wait_s=0.0):
            pass
        assert await limiter.acquire("event_read", 10, max_wait_s=0.0) is False

    async def test_a_429_drains_the_bucket_to_resynchronise(self):
        """A 429 means our accounting was more optimistic than the exchange's.

        Asserted against a second's worth of refill rather than against zero.
        ``snapshot`` refills before reporting, so an exact-zero assertion is
        really a claim that no measurable time passes between two statements --
        which held until the suite ran on a busy machine and the process was
        descheduled for a few milliseconds between them.
        """
        limiter = RateLimiter()
        before = limiter.snapshot()["event_write"]
        assert before["tokens"] > 0, "bucket should start full"

        limiter.note_429("event_write")

        after = limiter.snapshot()["event_write"]
        assert after["tokens"] < before["tokens"]
        assert after["tokens"] <= after["refill_per_s"]

    async def test_a_declined_acquisition_raises_rather_than_blocking(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        client = make_client(handler)
        client.rate_limiter = RateLimiter(event_read_per_s=1, utilisation=0.05)
        while await client.rate_limiter.acquire("event_read", 10, max_wait_s=0.0):
            pass
        try:
            with pytest.raises(RateLimited):
                await client.get("/markets", authenticated=False, max_wait_s=0.0)
        finally:
            await client.aclose()


class TestMalformedResponses:
    async def test_an_undecodable_body_is_an_error_not_an_empty_dict(self):
        """Papering over a broken body would let a schema change pass silently."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>not json</html>")

        client = make_client(handler)
        try:
            with pytest.raises(ExchangeError, match="undecodable"):
                await client.get("/markets", authenticated=False)
        finally:
            await client.aclose()

    async def test_an_error_body_that_is_not_json_still_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, content=b"plain text problem")

        client = make_client(handler)
        try:
            with pytest.raises(ExchangeError):
                await client.get("/markets", authenticated=False)
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Market data failures
# ---------------------------------------------------------------------------


class TestStaleAndBrokenMarketData:
    def test_a_market_never_updated_is_not_fresh(self):
        """Unknown is not the same as fine."""
        state = MarketState(MarketType.EVENT, max_age_ms=5_000)
        state.ensure("MKT")
        assert state.is_fresh("MKT", now_ms=NOW) is False
        assert state.age_ms("MKT", now_ms=NOW) is None

    def test_data_older_than_the_budget_is_not_fresh(self):
        """Past the quiet bound, a book is refused however healthy the feed.

        Ten seconds used to be enough here only because a single market makes
        the feed clock and the market clock the same number. On a delta feed a
        ten-second-old book is a book that has not changed, which is why the
        refusal is pinned at the quiet bound instead.
        """
        state = MarketState(
            MarketType.EVENT, max_age_ms=5_000, quiet_max_age_ms=300_000
        )
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [(4100, 1000)])
        state.apply_snapshot("MKT", book, now_ms=NOW - 400_000)
        # Something else arrived just now, so the feed is demonstrably alive and
        # only this market's own age can condemn it.
        state.apply_snapshot(
            "OTHER",
            OrderBook.from_two_sided_levels("OTHER", [(4000, 1000)], [(4100, 1000)]),
            now_ms=NOW,
        )
        assert state.is_fresh("MKT", now_ms=NOW) is False

    def test_a_delta_that_crosses_the_book_discards_it(self):
        """A crossed book means a message was missed; the book is not trusted."""
        state = MarketState(MarketType.EVENT, max_age_ms=5_000)
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [(4100, 1000)])
        state.apply_snapshot("MKT", book, now_ms=NOW)

        # Add a bid above the best ask.
        applied = state.apply_book_delta(
            "MKT", price_cc=4200, delta_qc=500, is_bid=True, now_ms=NOW
        )
        assert applied is False
        assert state.book("MKT") is None, "a crossed book must be dropped, not kept"
        assert state.books_resynced == 1

    def test_a_delta_for_an_unknown_market_is_refused(self):
        """No snapshot means no book to apply a delta to."""
        state = MarketState(MarketType.EVENT, max_age_ms=5_000)
        assert (
            state.apply_book_delta("NEW", price_cc=4000, delta_qc=100, is_bid=True, now_ms=NOW)
            is False
        )

    def test_a_one_sided_book_reports_no_midpoint(self):
        book = OrderBook.from_two_sided_levels("MKT", [(4000, 1000)], [])
        assert book.mid_cc is None
        assert book.spread_cc is None


# ---------------------------------------------------------------------------
# Duplicate messages
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_a_duplicate_fill_is_ignored(self):
        """A websocket replay after reconnect resends fills."""
        order = ManagedOrder(
            market_type=MarketType.EVENT, ticker="MKT",
            requested_qty_qc=1000, limit_price_cc=4000, outcome_side=OutcomeSide.YES,
        )
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.ACKNOWLEDGED)

        assert order.apply_fill(fill_id="f1", qty_qc=500, price_cc=4000) is True
        assert order.apply_fill(fill_id="f1", qty_qc=500, price_cc=4000) is False
        assert order.filled_qty_qc == 500

    async def test_a_duplicate_row_is_upserted_not_duplicated(self, sqlite_db):
        from sqlalchemy import func, select

        writer = AsyncWriter(sqlite_db)
        await writer.start()
        for _ in range(5):
            writer.enqueue(
                AuditLog(actor="test", action="dup", subject="same"), critical=True
            )
        await asyncio.sleep(1.0)
        await writer.stop()
        async with sqlite_db.session() as session:
            count = (await session.execute(select(func.count()).select_from(AuditLog))).scalar()
        # These have surrogate keys, so five enqueues are five rows -- the point
        # is that none were lost, not that they collapsed.
        assert count == 5


# ---------------------------------------------------------------------------
# Database failures
# ---------------------------------------------------------------------------


class TestDatabaseOutage:
    async def test_enqueue_does_not_raise_when_the_database_is_gone(self, sqlite_db):
        """The trading path must not care that persistence is broken."""
        writer = AsyncWriter(sqlite_db)
        await writer.start()
        await sqlite_db.disconnect()  # pull the database out from under it

        for i in range(10):
            assert (
                writer.enqueue(
                    Quote(
                        ticker=f"T{i}", market_type="event",
                        ts=datetime.now(UTC), bid_cc=4000, source="t",
                    )
                )
                is True
            )
        await asyncio.sleep(0.6)
        await writer.stop(drain=False, timeout_s=2.0)

    async def test_a_full_queue_evicts_non_critical_before_critical(self, sqlite_db):
        writer = AsyncWriter(sqlite_db)
        writer._max_size = 5
        for i in range(5):
            writer.enqueue(
                Quote(ticker=f"Q{i}", market_type="event", ts=datetime.now(UTC), source="t")
            )
        assert writer.enqueue(AuditLog(actor="t", action="critical"), critical=True) is True
        assert writer.stats()["dropped"] == 1

    async def test_a_ping_failure_marks_the_database_unhealthy_without_raising(
        self, sqlite_db
    ):
        await sqlite_db.disconnect()
        # Reconnect the object but point it at a broken engine.
        assert await sqlite_db.ping() is False
        assert sqlite_db.healthy is False


# ---------------------------------------------------------------------------
# Risk under degraded conditions
# ---------------------------------------------------------------------------


class TestRiskFailsSafe:
    def _intent(self, qty_qc: int = 100) -> OrderIntent:
        return OrderIntent(
            market_type=MarketType.PERP, ticker="P", qty_qc=qty_qc,
            limit_price_cc=5000, book_side=BookSide.BID, strategy_id="t",
        )

    def _book(self) -> OrderBook:
        return OrderBook.from_two_sided_levels("P", [(4990, 900_000)], [(5010, 900_000)])

    def test_stale_account_state_blocks_trading(self, perp_risk, now_ms):
        state = PerpAccountState(
            equity_micros=200_000_000, equity_source="t", equity_is_known=True,
            updated_at_ms=now_ms - 10_000_000,
        )
        verdict = perp_risk.evaluate(self._intent(), state, book=self._book(), now_ms=now_ms)
        assert verdict.decision is RiskDecision.REJECTED
        assert verdict.reason_code == "STALE_ACCOUNT_STATE"

    def test_missing_equity_blocks_increases_but_permits_reduction(self, perp_risk, now_ms):
        state = PerpAccountState(
            equity_micros=0, equity_source="unavailable", equity_is_known=False,
            updated_at_ms=now_ms,
        )
        state.positions["P"] = PerpPositionState(
            ticker="P", position_qc=10_000, mark_price_cc=5000
        )

        increase = perp_risk.evaluate(
            self._intent(), state, book=self._book(), now_ms=now_ms
        )
        assert increase.decision is RiskDecision.REJECTED

        reduce = self._intent(1_000)
        reduce.book_side = BookSide.ASK
        reduce.reduce_only = True
        assert perp_risk.evaluate(
            reduce, state, book=self._book(), now_ms=now_ms
        ).approved is True

    def test_a_missing_mark_price_blocks_trading(self, perp_risk, perp_state, now_ms):
        """No mark means no exposure figure, so no leverage check is possible."""
        verdict = perp_risk.evaluate(self._intent(), perp_state, book=None, now_ms=now_ms)
        assert verdict.decision is RiskDecision.REJECTED
        assert verdict.reason_code == "NO_MARK_PRICE"

    def test_an_empty_book_blocks_trading(self, perp_risk, perp_state, now_ms):
        perp_state.positions["P"] = PerpPositionState(
            ticker="P", position_qc=0, mark_price_cc=5000
        )
        verdict = perp_risk.evaluate(
            self._intent(), perp_state, book=OrderBook("P"), now_ms=now_ms
        )
        assert not verdict.approved


# ---------------------------------------------------------------------------
# Clock problems
# ---------------------------------------------------------------------------


class TestClockProblems:
    def test_a_simulated_clock_refuses_to_move_backwards(self):
        """Time going backwards in a simulation would replay decisions."""
        from kat.common.timeutil import SimulatedClock

        clock = SimulatedClock(NOW)
        clock.set(NOW + 1000)
        with pytest.raises(ValueError, match="backwards"):
            clock.set(NOW)

    def test_a_negative_advance_is_refused(self):
        from kat.common.timeutil import SimulatedClock

        with pytest.raises(ValueError):
            SimulatedClock(NOW).advance(-1)

    def test_a_naive_datetime_is_refused(self):
        """Silently assuming UTC for a local timestamp is how a news article
        moves across a decision boundary in a backtest."""
        from kat.common.timeutil import to_ms

        with pytest.raises(ValueError, match="naive"):
            to_ms(datetime(2026, 1, 1))

    def test_a_clock_jump_backwards_cannot_produce_negative_latency(self):
        """Latency uses the monotonic counter, which an NTP correction cannot
        move."""
        from kat.common.timeutil import SystemClock

        clock = SystemClock()
        first = clock.monotonic_ns()
        second = clock.monotonic_ns()
        assert second >= first


# ---------------------------------------------------------------------------
# Execution under failure
# ---------------------------------------------------------------------------


class TestExecutionFailsSafe:
    async def test_an_unapproved_verdict_is_refused_by_the_execution_engine(self):
        from kat.common.errors import RiskRejection
        from kat.common.timeutil import SystemClock
        from kat.execution.event_exec import EventExecutionEngine
        from kat.risk.types import RiskVerdict

        engine = EventExecutionEngine(
            rest=None,
            market_state=MarketState(MarketType.EVENT),
            clock=SystemClock(),
            environment=TradingEnvironment.PAPER,
        )
        intent = OrderIntent(
            market_type=MarketType.EVENT, ticker="M", qty_qc=100,
            limit_price_cc=4000, outcome_side=OutcomeSide.YES,
        )
        with pytest.raises(RiskRejection):
            await engine.submit(RiskVerdict.reject(intent, "NOPE", "refused by risk"))

    async def test_a_live_environment_that_is_not_armed_refuses_to_submit(self):
        """Belt and braces: settings validation should catch this first."""
        from kat.common.timeutil import SystemClock
        from kat.execution.event_exec import EventExecutionEngine
        from kat.risk.types import RiskVerdict

        engine = EventExecutionEngine(
            rest=None,
            market_state=MarketState(MarketType.EVENT),
            clock=SystemClock(),
            environment=TradingEnvironment.LIVE,
            live_armed=False,
        )
        intent = OrderIntent(
            market_type=MarketType.EVENT, ticker="M", qty_qc=100,
            limit_price_cc=4000, outcome_side=OutcomeSide.YES,
        )
        verdict = RiskVerdict(
            decision=RiskDecision.APPROVED, intent=intent, approved_qty_qc=100,
            reason_code="APPROVED", reason="ok",
        )
        order = await engine.submit(verdict)
        assert order.state is OrderState.RISK_REJECTED
        assert "not armed" in (order.reject_reason or "")

    async def test_an_order_for_the_wrong_domain_is_refused(self):
        from kat.common.errors import RiskRejection
        from kat.common.timeutil import SystemClock
        from kat.execution.event_exec import EventExecutionEngine
        from kat.risk.types import RiskVerdict

        engine = EventExecutionEngine(
            rest=None,
            market_state=MarketState(MarketType.EVENT),
            clock=SystemClock(),
            environment=TradingEnvironment.PAPER,
        )
        perp_intent = OrderIntent(
            market_type=MarketType.PERP, ticker="P", qty_qc=100,
            limit_price_cc=5000, book_side=BookSide.BID,
        )
        verdict = RiskVerdict(
            decision=RiskDecision.APPROVED, intent=perp_intent, approved_qty_qc=100,
            reason_code="APPROVED", reason="ok",
        )
        with pytest.raises(RiskRejection) as exc:
            await engine.submit(verdict)
        assert exc.value.reason_code == "WRONG_DOMAIN"

    async def test_a_kill_switch_engaged_after_approval_still_stops_the_order(
        self, sqlite_db
    ):
        """The switch is checked again at the last point before an order leaves."""
        from kat.common.enums import KillSwitchScope
        from kat.common.errors import KillSwitchEngaged
        from kat.common.timeutil import SystemClock
        from kat.execution.event_exec import EventExecutionEngine
        from kat.risk.killswitch import KillSwitchManager
        from kat.risk.types import RiskVerdict

        switches = KillSwitchManager(sqlite_db)
        await switches.load()
        engine = EventExecutionEngine(
            rest=None,
            market_state=MarketState(MarketType.EVENT),
            clock=SystemClock(),
            kill_switches=switches,
            environment=TradingEnvironment.PAPER,
        )
        intent = OrderIntent(
            market_type=MarketType.EVENT, ticker="M", qty_qc=100,
            limit_price_cc=4000, outcome_side=OutcomeSide.YES,
        )
        verdict = RiskVerdict(
            decision=RiskDecision.APPROVED, intent=intent, approved_qty_qc=100,
            reason_code="APPROVED", reason="ok",
        )
        await switches.engage(KillSwitchScope.GLOBAL, reason="test", actor="test")
        with pytest.raises(KillSwitchEngaged):
            await engine.submit(verdict)


class TestKillSwitchFailsClosed:
    async def test_unreadable_state_is_treated_as_engaged(self):
        """An unknown halt state is a halt."""
        from kat.common.enums import KillSwitchScope
        from kat.risk.killswitch import KillSwitchManager

        class BrokenDatabase:
            def session(self):
                raise RuntimeError("database is gone")

        switches = KillSwitchManager(BrokenDatabase())  # type: ignore[arg-type]
        await switches.load()
        assert switches.is_engaged(KillSwitchScope.GLOBAL) is True
        assert switches.blocking_switch(MarketType.EVENT) == KillSwitchScope.GLOBAL

    async def test_engaging_works_when_persistence_fails(self):
        """A halt must work during exactly the failure that makes you want one."""
        from kat.common.enums import KillSwitchScope
        from kat.risk.killswitch import KillSwitchManager

        class BrokenDatabase:
            def session(self):
                raise RuntimeError("database is gone")

        switches = KillSwitchManager(BrokenDatabase())  # type: ignore[arg-type]
        switches._load_failed = False
        switches._loaded = True
        await switches.engage(KillSwitchScope.PERP, reason="test", actor="test")
        assert switches.is_engaged(KillSwitchScope.PERP) is True
