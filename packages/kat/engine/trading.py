"""The trading loop.

This is the critical path, and it is deliberately short::

    market state -> strategy -> portfolio sizing -> risk engine -> execution

Everything else -- persistence, analytics, dashboard updates, research, news,
model training -- happens asynchronously somewhere else. There is no database
read between a market tick and an order, and no dashboard round-trip either.

One loop per domain
-------------------
Event contracts and perpetuals each get their own :class:`TradingLoop` with their
own strategies, their own portfolio book, their own risk engine and their own
execution engine. They share this class only because the *shape* of the loop is
the same; no calculation crosses between them. Halting one leaves the other
running, which is what the separate kill switches are for.

Ordering within a cycle
-----------------------
Exits are considered before entries. If capital is short, closing a losing
position must not be starved by a queue of new ideas competing for the same
dollars.

What stops a cycle
------------------
A cycle is skipped entirely -- not degraded, not best-effort -- when the
application is not permitted to trade, a kill switch covering this domain is
engaged, or account state is stale. Trading on data we cannot vouch for is worse
than not trading.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kat.common.enums import (
    MarketType,
    OperatingMode,
    StrategyState,
    TradingEnvironment,
)
from kat.common.errors import KillSwitchEngaged, RiskRejection
from kat.common.fixedpoint import CC_PER_DOLLAR
from kat.common.ids import new_decision_id, use_correlation_id
from kat.common.timeutil import Clock
from kat.engine.decision import DecisionRecorder, TradeDecision
from kat.execution.base import ExecutionEngine
from kat.kalshi.fees import FeeSchedule, estimate_fee_micros
from kat.marketdata.state import MarketState, funding_rate_bps_per_hour
from kat.observability.logging import get_logger
from kat.observability.metrics import M_DECISION_LATENCY, M_RISK_LATENCY, metrics
from kat.risk.event_risk import EventAccountState, EventPositionState
from kat.risk.killswitch import KillSwitchManager
from kat.risk.types import OrderIntent, RiskVerdict
from kat.strategies.base import Strategy, StrategyContext

log = get_logger(__name__)

#: How often the loop says what it is doing. Sixty seconds is often
#: enough that a stalled or silently-declining loop is obvious within a
#: minute, and rare enough that a quiet overnight run stays readable.
_SUMMARY_INTERVAL_S = 60.0

__all__ = ["CycleResult", "TradingLoop"]

#: Strategy states whose intents may reach a real order. PAPER and PROBATION are
#: included because they trade -- with simulated fills and reduced capital
#: respectively -- which is the entire point of those stages.
_TRADEABLE_STATES = frozenset(
    {StrategyState.PAPER, StrategyState.PROBATION, StrategyState.LIVE, StrategyState.MONITORING}
)


@dataclass(slots=True)
class CycleResult:
    """What one pass over the watchlist did."""

    markets_examined: int = 0
    markets_skipped_stale: int = 0
    intents_generated: int = 0
    intents_approved: int = 0
    intents_rejected: int = 0
    orders_submitted: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    reasons: dict[str, int] = field(default_factory=dict)

    def note_rejection(self, reason_code: str) -> None:
        self.intents_rejected += 1
        self.reasons[reason_code] = self.reasons.get(reason_code, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "markets_examined": self.markets_examined,
            "markets_skipped_stale": self.markets_skipped_stale,
            "intents_generated": self.intents_generated,
            "intents_approved": self.intents_approved,
            "intents_rejected": self.intents_rejected,
            "orders_submitted": self.orders_submitted,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 2),
            "rejection_reasons": dict(sorted(self.reasons.items())),
        }


class TradingLoop:
    """One domain's trading loop.

    ``risk_engine`` is typed loosely because the two engines are deliberately
    different classes with different state objects; the loop only relies on the
    shared ``evaluate(intent, state, ...) -> RiskVerdict`` shape, which both
    provide. Making them share a base class would have invited exactly the
    generic risk calculation this platform is required not to have.
    """

    def __init__(
        self,
        *,
        market_type: MarketType,
        market_state: MarketState,
        risk_engine: Any,
        execution: ExecutionEngine,
        recorder: DecisionRecorder,
        clock: Clock,
        kill_switches: KillSwitchManager | None = None,
        interval_s: float = 1.0,
        max_intents_per_cycle: int = 20,
        order_ttl_ms: int = 60_000,
    ) -> None:
        self.market_type = market_type
        self.market_state = market_state
        self.risk_engine = risk_engine
        self.execution = execution
        self.recorder = recorder
        self.clock = clock
        self.kill_switches = kill_switches
        self.interval_s = interval_s
        self.max_intents_per_cycle = max_intents_per_cycle
        #: How long an unfilled order may rest before the loop cancels it.
        self.order_ttl_ms = order_ttl_ms

        self.strategies: list[Strategy] = []
        #: Set by the owning application; the loop consults it every cycle.
        self.trading_permitted: bool = False
        self.operating_mode: OperatingMode = OperatingMode.GATED_DISCOVERY
        #: Domain account state, supplied and refreshed by the reconciler.
        self.account_state: Any = None
        #: Per-market fee schedules, loaded from the series metadata. This is the
        #: execution engine's own mapping rather than a copy: the fee that judges
        #: a trade and the fee charged for it must come from one schedule, and two
        #: dicts that are only *usually* equal is how they stop being. Test
        #: doubles that implement only the submit path get their own, since there
        #: is nothing on the other side of them to disagree with.
        self.fee_schedules: dict[str, FeeSchedule] = getattr(
            execution, "fee_schedules", None
        ) or {}
        #: Model probabilities keyed by ticker, refreshed by the inference worker.
        self.model_probabilities: dict[str, tuple[float, str, str]] = {}
        #: Research/news signals keyed by ticker, already time-filtered.
        self.research: dict[str, dict[str, Any]] = {}
        #: Cross-market intent producers; see :meth:`add_intent_source`.
        self.intent_sources: list[tuple[str, Any]] = []
        #: Full feature vectors keyed by ticker, published by the feature
        #: worker. Strategies read ``features``; inference needs the timestamps
        #: these carry.
        self.feature_vectors: dict[str, Any] = {}
        #: Detected regime keyed by ticker.
        self.regimes: dict[str, str] = {}
        #: Features keyed by ticker, already filtered to what is knowable now.
        self.features: dict[str, dict[str, float]] = {}

        #: Whether simulated fills should be booked against the local account.
        #: Set by the entrypoint in PAPER only; live and demo get positions
        #: from the reconciler and must not have a second source for them.
        self.paper_fills: bool = False

        self.last_cycle: CycleResult | None = None
        self.cycles = 0
        self._last_summary_s = 0.0
        self._running = False

    # -- registration --------------------------------------------------------

    def add_intent_source(self, name: str, produce: Any) -> None:
        """Register a producer of intents that is not a per-ticker strategy.

        The strategy sweep asks each strategy about each market in turn, which
        is the wrong shape for anything cross-market: an arbitrage between three
        contracts is not an opinion about any one of them. A source is called
        once per cycle with ``now_ms`` and returns intents that then travel the
        *same* path as every other intent -- risk engine, decision record,
        execution. There is no shortcut here, only a different way in.
        """
        self.intent_sources.append((name, produce))

    def add_strategy(self, strategy: Strategy) -> None:
        if strategy.market_type is not self.market_type:
            raise ValueError(
                f"{strategy.strategy_id} is a {strategy.market_type.value} strategy; "
                f"this loop trades {self.market_type.value}"
            )
        self.strategies.append(strategy)
        log.info(
            "trading.strategy_registered",
            domain=self.market_type.value,
            strategy_id=strategy.strategy_id,
            version=strategy.version,
            state=strategy.state.value,
        )

    def active_strategies(self) -> list[Strategy]:
        """Strategies permitted to trade right now.

        Mode B restricts the set to what a human enabled; the lifecycle state
        restricts it further. Both filters apply -- being enabled does not
        promote a strategy that has not earned its rung.
        """
        out: list[Strategy] = []
        for strategy in self.strategies:
            if strategy.state not in _TRADEABLE_STATES:
                continue
            if self.kill_switches is not None and self.kill_switches.is_engaged(
                "strategy", strategy.strategy_id
            ):
                continue
            out.append(strategy)
        return out

    # -- the loop ------------------------------------------------------------

    async def run(self) -> None:
        """Run until cancelled. Registered as a supervised component."""
        self._running = True
        log.info("trading.loop_started", domain=self.market_type.value, interval_s=self.interval_s)
        while self._running:
            started = time.monotonic()
            try:
                self.last_cycle = await self.run_cycle()
                self.cycles += 1
                self._log_periodic_summary()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error(
                    "trading.cycle_failed",
                    domain=self.market_type.value, error=str(exc), exc_info=True,
                )
            # Pace on elapsed time so a slow cycle does not compound into a
            # backlog of cycles all firing at once.
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, self.interval_s - elapsed))

    def _log_periodic_summary(self) -> None:
        """Say, periodically, what the loop is doing and why nothing is trading.

        The loop used to log ``trading.loop_started`` once and then nothing,
        forever. Five hours of that is indistinguishable from a loop that died
        on its first cycle, and it left the single question an operator actually
        asks -- "why has it not traded?" -- answerable only by querying an API
        that has to be running to answer.

        So the answer goes in the log. The skip tallies come from the strategies
        themselves, which have always counted them; they were simply never
        reported anywhere a person would look.
        """
        now = time.monotonic()
        if now - self._last_summary_s < _SUMMARY_INTERVAL_S:
            return
        self._last_summary_s = now

        cycle = self.last_cycle
        reasons: Counter[str] = Counter()
        for strategy in self.strategies:
            health = getattr(strategy, "health", None)
            if not callable(health):
                continue
            with contextlib.suppress(Exception):
                reasons.update(health().get("skips") or {})

        log.info(
            "trading.summary",
            domain=self.market_type.value,
            cycles=self.cycles,
            markets_examined=cycle.markets_examined if cycle else 0,
            markets_skipped_stale=cycle.markets_skipped_stale if cycle else 0,
            intents_generated=cycle.intents_generated if cycle else 0,
            orders_submitted=cycle.orders_submitted if cycle else 0,
            strategies_active=len(self.active_strategies()),
            # The whole point of the line: the reasons, most common first.
            declined_because=dict(reasons.most_common(6)),
        )

    def stop(self) -> None:
        self._running = False

    async def run_cycle(self) -> CycleResult:
        """One pass: gather intents, size them, risk them, send what survives."""
        result = CycleResult()
        started = time.monotonic()

        blocked = self._blocked_reason()
        if blocked is not None:
            result.note_rejection(blocked)
            result.duration_ms = (time.monotonic() - started) * 1000
            return result

        now_ms = self.clock.now_ms()
        # Before anything is measured: a cancelled order releases the market's
        # headroom, and the refresh below is what tells risk about it.
        await self.execution.expire_stale_orders(
            now_ms=now_ms, ttl_ms=self.order_ttl_ms
        )
        self._mark_positions()
        self._refresh_from_execution(now_ms)
        strategies = self.active_strategies()
        if not strategies:
            result.duration_ms = (time.monotonic() - started) * 1000
            return result

        # ``None`` where the intent came from a cross-market source rather than
        # from a per-ticker strategy; see :meth:`_process`.
        intents: list[tuple[Strategy | None, OrderIntent]] = []
        for ticker in self.market_state.tickers():
            snapshot = self.market_state.get(ticker)
            if snapshot is None:
                continue
            result.markets_examined += 1

            if not self.market_state.is_fresh(ticker, now_ms=now_ms):
                result.markets_skipped_stale += 1
                continue

            for strategy in strategies:
                if strategy.metadata.requires_book and snapshot.book is None:
                    continue
                if not self.market_state.is_fresh(
                    ticker, now_ms=now_ms, max_age_ms=strategy.metadata.max_data_age_ms
                ):
                    continue
                try:
                    ctx = self._context(strategy, snapshot, now_ms)
                    produced = strategy.generate(ctx)
                except Exception as exc:
                    result.errors += 1
                    log.error(
                        "trading.strategy_failed",
                        domain=self.market_type.value,
                        strategy_id=strategy.strategy_id, ticker=ticker,
                        error=str(exc), exc_info=True,
                    )
                    continue
                for intent in produced or []:
                    intents.append((strategy, intent))

        for name, produce in self.intent_sources:
            try:
                for intent in produce(now_ms) or []:
                    intents.append((None, intent))
            except Exception as exc:
                result.errors += 1
                log.error(
                    "trading.intent_source_failed",
                    domain=self.market_type.value, source=name,
                    error=str(exc), exc_info=True,
                )

        result.intents_generated = len(intents)
        if not intents:
            result.duration_ms = (time.monotonic() - started) * 1000
            return result

        # Exits first: closing a position must not be starved by new ideas
        # competing for the same capital.
        intents.sort(key=lambda pair: (not pair[1].reduce_only, -(pair[1].expected_value_micros or 0)))

        # ``owner`` rather than ``strategy``: this one can be None, and reusing
        # the sweep's name above would hide that from the reader and the checker.
        for owner, intent in intents[: self.max_intents_per_cycle]:
            await self._process(owner, intent, now_ms, result)

        result.duration_ms = (time.monotonic() - started) * 1000
        metrics.observe(M_DECISION_LATENCY, result.duration_ms, domain=self.market_type.value)
        return result

    # -- one intent ----------------------------------------------------------

    async def _process(
        self, strategy: Strategy | None, intent: OrderIntent, now_ms: int, result: CycleResult
    ) -> None:
        """Take one intent from opinion to submitted order (or not).

        ``strategy`` is ``None`` for an intent from a cross-market source, which
        has no per-ticker strategy behind it. Such an intent must already carry
        its own identity -- everything downstream, including the audit record,
        is keyed on it.
        """
        if strategy is None:
            # Both halves of the identity, not just the id. The version is what
            # ties a decision to the code that made it, and the old guard
            # checked only the id before reading ``strategy.version`` off the
            # None it had just proved was there -- so a source that supplied an
            # id and no version raised AttributeError out of the cycle, taking
            # every intent behind it in the batch with it.
            if not intent.strategy_id or not intent.strategy_version:
                result.errors += 1
                log.error(
                    "trading.unattributed_intent",
                    domain=self.market_type.value, ticker=intent.ticker,
                    strategy_id=intent.strategy_id or None,
                    strategy_version=intent.strategy_version or None,
                )
                return
        else:
            intent.strategy_id = intent.strategy_id or strategy.strategy_id
            intent.strategy_version = intent.strategy_version or strategy.version
        intent.decision_id = intent.decision_id or new_decision_id()

        with use_correlation_id(intent.correlation_id) as correlation_id:
            intent.correlation_id = correlation_id
            snapshot = self.market_state.get(intent.ticker)
            market_price_cc = snapshot.reference_price_cc() if snapshot else None
            book = snapshot.book if snapshot else None

            fees = estimate_fee_micros(
                price_cc=intent.limit_price_cc,
                qty_qc=intent.qty_qc,
                schedule=self.fee_schedules.get(intent.ticker),
                is_taker=True,
            )

            try:
                risk_started = time.monotonic()
                verdict: RiskVerdict = self.risk_engine.evaluate(
                    intent,
                    self.account_state,
                    book=book,
                    now_ms=now_ms,
                    estimated_fees_micros=fees.fee_micros,
                    **self._domain_kwargs(snapshot),
                )
                metrics.observe(
                    M_RISK_LATENCY,
                    (time.monotonic() - risk_started) * 1000,
                    domain=self.market_type.value,
                )
            except Exception as exc:
                result.errors += 1
                log.error(
                    "trading.risk_evaluation_failed",
                    ticker=intent.ticker, strategy_id=intent.strategy_id,
                    error=str(exc), exc_info=True,
                )
                return

            probability = self.model_probabilities.get(intent.ticker)
            decision: TradeDecision = self.recorder.record(
                self.recorder.build(
                    verdict,
                    market_price_cc=market_price_cc,
                    prediction=probability[0] if probability else None,
                    model_id=probability[1] if probability else None,
                    model_version=probability[2] if probability else None,
                    features=self.features.get(intent.ticker),
                )
            )

            if not verdict.approved:
                result.note_rejection(verdict.reason_code)
                return

            result.intents_approved += 1

            try:
                order = await self.execution.submit(verdict)
            except KillSwitchEngaged as exc:
                # Engaged between risk approval and submission. Working as designed.
                result.note_rejection("KILL_SWITCH")
                log.warning(
                    "trading.blocked_by_kill_switch",
                    ticker=intent.ticker, scope=exc.scope,
                )
                return
            except RiskRejection as exc:
                result.note_rejection(exc.reason_code)
                log.warning(
                    "trading.execution_refused",
                    ticker=intent.ticker, reason_code=exc.reason_code, reason=str(exc),
                )
                return
            except Exception as exc:
                result.errors += 1
                log.error(
                    "trading.submit_failed",
                    ticker=intent.ticker, error=str(exc), exc_info=True,
                )
                return

            result.orders_submitted += 1
            self.recorder.attach_execution(
                decision,
                order_id=order.order_id,
                result={
                    "state": order.state.value,
                    "filled_qty_qc": order.filled_qty_qc,
                    "avg_fill_price_cc": order.avg_fill_price_cc,
                    "submit_latency_ms": order.submit_latency_ms,
                    "slippage_cc": order.slippage_cc(),
                    "exchange_order_id": order.exchange_order_id,
                },
            )
            if order.filled_qty_qc > 0:
                self._apply_paper_fill(intent, order)
                if strategy is not None:
                    strategy.on_fill(
                        intent.ticker, order.filled_qty_qc, order.avg_fill_price_cc or 0
                    )

    def _mark_positions(self) -> None:
        """Value open positions at the current market, in paper.

        Without this a position is only ever worth what it cost. Equity moves
        when a trade closes and at no other time, every position reports zero
        unrealised P&L for its whole life, and the question the dashboard exists
        to answer -- is this working -- has no answer until something is sold.

        A long holds YES and is worth the YES price; a short holds NO and is
        worth what is left of the dollar.

        Marking is gated on the *feed*, not on the market
        ------------------------------------------------
        A quiet market is not an unpriced one. This is a delta feed: no message
        means the book has not changed, so the last mid is still the current mid
        and marking a position at it is correct rather than optimistic.

        This used to require ``is_fresh``, which is the test for whether a market
        may be *traded* -- a stricter question, and rightly so. The two came
        apart on the real board: the median watched market updates every 489
        seconds against a 300-second quiet bound, so most positions failed the
        test, kept a frozen mark, and equity sat at exactly $190.346066 across
        eighteen open positions for as long as anyone watched it. Paper equity
        that cannot move cannot answer the one question paper trading exists to
        answer.

        What is still refused is marking while disconnected. If the feed as a
        whole has gone silent the last mid is no longer evidence of anything, and
        the position keeps its previous mark rather than being revalued against a
        price we can no longer vouch for -- and never written to zero, because an
        absent price is not a price of nothing.
        """
        if not self.paper_fills or self.account_state is None:
            return
        positions = getattr(self.account_state, "positions", None)
        if not positions:
            self.account_state.portfolio_value_micros = 0
            return

        now_ms = self.clock.now_ms()
        feed_live = self._feed_live(now_ms)
        total = 0
        for ticker, position in positions.items():
            qty = abs(position.position_qc)
            if qty == 0:
                continue
            basis = (position.avg_entry_cc or 0) * qty

            snapshot = self.market_state.get(ticker)
            mid = snapshot.mid_cc if snapshot is not None else None
            if mid is None or not feed_live:
                # Keep the last mark. The position is still worth whatever it was
                # worth a moment ago; what is missing is our view of it.
                total += basis + position.unrealized_pnl_micros
                continue

            value_cc = mid if position.position_qc > 0 else CC_PER_DOLLAR - mid
            value = value_cc * qty
            position.unrealized_pnl_micros = value - basis
            total += value

        self.account_state.portfolio_value_micros = total

    def _refresh_from_execution(self, now_ms: int) -> None:
        """Put onto the account state the two facts only execution knows.

        The event risk engine reads ``orders_this_minute`` and each position's
        ``pending_exposure_micros``, and until now nothing outside the backtester
        ever wrote either. Both sat at their defaults of zero on every live and
        paper evaluation, so ``max_orders_per_minute`` compared ``0 < 30`` and
        could not fire, and capital already promised to resting orders counted
        for nothing against the cash and concentration limits.

        Both defaults read as "nothing pending, nothing submitted", which is the
        most permissive answer either field has -- the opposite of this
        platform's posture everywhere else, where what cannot be established is
        refused rather than assumed favourable. The backtester filled both in,
        so it enforced limits the live engine did not, and a backtest was
        measuring a stricter platform than the one that would trade.

        Observed: 55 good-till-cancelled orders on one market in 55 seconds,
        about twice the configured rate limit, none of them filling, none of them
        counted against capital, and none refused.

        A ticker with working orders but no position gets an entry so its
        commitment reaches the portfolio total. That entry holds no contracts, so
        it does not inflate the open-position count, which reads ``position_qc``.
        """
        state = self.account_state
        # Named rather than duck-typed: these two fields belong to the
        # event-contract account state. Perpetuals express both ideas through the
        # leverage invariant, on a state object that deliberately shares no base
        # class with this one, so there is nothing here to generalise over.
        if state is None or self.market_type is not MarketType.EVENT:
            return
        if not isinstance(state, EventAccountState):
            return

        state.orders_this_minute = self.execution.orders_in_last_minute(now_ms=now_ms)

        committed = self.execution.working_exposure_micros()
        for ticker, micros in committed.items():
            position = state.positions.get(ticker)
            if position is None:
                snapshot = self.market_state.get(ticker)
                position = EventPositionState(
                    ticker=ticker,
                    event_ticker=(snapshot.event_ticker if snapshot else "") or "",
                )
                state.positions[ticker] = position
            position.pending_exposure_micros = micros
        for ticker, position in state.positions.items():
            if ticker not in committed:
                position.pending_exposure_micros = 0

    def _apply_paper_fill(self, intent: OrderIntent, order: Any) -> None:
        """Book a simulated fill against the local account.

        Only in paper. Live and demo rebuild positions from the exchange through
        the reconciler, and a local guess layered on top of that would be a
        second source of truth for the same number.

        Paper has no exchange to ask, and nothing was doing this job: 108
        simulated fills were recorded against an account that still read exactly
        its seeded balance, with no positions, no exposure and no profit or
        loss. The strategies were trading and the dashboard had no way to show
        whether any of it worked.
        """
        if not self.paper_fills or self.account_state is None:
            return
        apply = getattr(self.account_state, "apply_fill", None)
        if apply is None:
            return
        realised = apply(
            ticker=intent.ticker,
            qty_qc=order.filled_qty_qc,
            price_cc=order.avg_fill_price_cc or intent.limit_price_cc or 0,
            is_buy=intent.is_buy,
            fee_micros=order.fees_micros,
            event_ticker=getattr(intent, "event_ticker", "") or "",
            now_ms=self.clock.now_ms(),
        )
        # Re-marked immediately so equity does not dip by the cost of the
        # position until the next cycle: entering a trade is not a loss.
        self._mark_positions()
        log.info(
            "trading.paper_fill",
            domain=self.market_type.value,
            ticker=intent.ticker,
            qty_qc=order.filled_qty_qc,
            price_cc=order.avg_fill_price_cc,
            realised_micros=realised,
            balance_micros=self.account_state.balance_micros,
        )

    # -- helpers -------------------------------------------------------------

    def _blocked_reason(self) -> str | None:
        """Why this cycle must not run, or ``None`` to proceed."""
        if not self.trading_permitted:
            return "TRADING_NOT_PERMITTED"
        if self.kill_switches is not None:
            scope = self.kill_switches.blocking_switch(self.market_type)
            if scope is not None:
                return f"KILL_SWITCH_{scope.upper()}"
        if self.account_state is None:
            return "NO_ACCOUNT_STATE"
        return None

    def _context(self, strategy: Strategy, snapshot: Any, now_ms: int) -> StrategyContext:
        probability = self.model_probabilities.get(snapshot.ticker)
        return StrategyContext(
            now_ms=now_ms,
            market=snapshot,
            features=self.features.get(snapshot.ticker, {}),
            model_probability=probability[0] if probability else None,
            model_id=probability[1] if probability else None,
            model_version=probability[2] if probability else None,
            position_qc=self._position_qc(snapshot.ticker),
            regime=self.regimes.get(snapshot.ticker),
            research=self.research.get(snapshot.ticker, {}),
            fee_multiplier=(
                float(self.fee_schedules[snapshot.ticker].multiplier)
                if snapshot.ticker in self.fee_schedules
                else None
            ),
            # Decided here rather than in the strategy: only the market state
            # sees every market at once, which is what separates "this market is
            # quiet" from "the connection is gone". Both look identical from one
            # snapshot's timestamp.
            feed_live=self._feed_live(now_ms),
        )

    def _feed_live(self, now_ms: int) -> bool:
        """Whether the feed as a whole is currently delivering.

        Measured against ``feed_max_age_ms``, the budget for the question this
        actually asks. It used to use ``max_age_ms`` -- five seconds, the budget
        for whether one market is current -- which is a different question, and
        five seconds of silence across the whole board is quiet rather than
        broken.

        The loop's own staleness test was moved off that budget; this was left
        behind, and it is the copy the *strategies* read. So the loop would
        examine all sixty markets and every strategy would then decline them
        with ``stale_market_data``: 0 skipped at the top of the cycle and 7,211
        strategy-level refusals underneath it, in the same run.
        """
        age = self.market_state.feed_age_ms(now_ms=now_ms)
        return age is not None and age <= self.market_state.feed_max_age_ms

    def _position_qc(self, ticker: str) -> int:
        state = self.account_state
        if state is None:
            return 0
        position = getattr(state, "positions", {}).get(ticker)
        return getattr(position, "position_qc", 0) if position else 0

    def _domain_kwargs(self, snapshot: Any) -> dict[str, Any]:
        """Extra arguments the two risk engines take differently.

        The event engine wants the event ticker for its correlated-exposure
        check; the perp engine wants the funding rate. Neither accepts the
        other's argument, which is the separation working as intended.
        """
        if self.market_type is MarketType.EVENT:
            return {"event_ticker": getattr(snapshot, "event_ticker", "") or ""}
        # The raw rate is a per-interval fraction; the risk engine works in
        # basis points per hour. One shared helper owns that conversion so the
        # strategy and the risk engine cannot disagree about the units.
        bps_per_hour = funding_rate_bps_per_hour(
            getattr(snapshot, "funding_rate", None),
            getattr(snapshot, "funding_interval_hours", 1.0),
        )
        if bps_per_hour is None:
            return {}
        from decimal import Decimal

        return {"funding_rate_bps_per_hour": Decimal(str(bps_per_hour))}

    def stats(self) -> dict[str, Any]:
        return {
            "domain": self.market_type.value,
            "cycles": self.cycles,
            "strategies_registered": len(self.strategies),
            "strategies_active": len(self.active_strategies()),
            "trading_permitted": self.trading_permitted,
            "environment": self.execution.environment.value,
            "markets_watched": len(self.market_state),
            "last_cycle": self.last_cycle.to_dict() if self.last_cycle else None,
            "execution": self.execution.stats(),
            "recorder": self.recorder.stats(),
        }


def is_paper(loop: TradingLoop) -> bool:
    return loop.execution.environment is TradingEnvironment.PAPER
