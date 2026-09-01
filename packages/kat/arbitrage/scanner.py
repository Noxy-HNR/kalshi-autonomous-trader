"""Arbitrage scanning and intent construction.

The scanner walks live market state looking for the inconsistencies in
:mod:`kat.arbitrage.logical`, prices them honestly, and records **every**
candidate it evaluated -- including the ones it declined and why.

Recording declines is not bookkeeping. If the platform never declines anything,
its cost model is understating costs; if it declines everything, the model is
overstating them. The distribution of declined edges is the only direct evidence
that the fee, depth and leg-risk assumptions are calibrated, and it is worth more
than the record of the trades that were taken.

The scanner does not execute
----------------------------
It emits :class:`~kat.risk.types.OrderIntent` objects that go through the normal
Portfolio -> Risk -> Execution pipeline like everything else. An arbitrage is not
exempt from position limits, kill switches, or -- for anything touching
perpetuals -- the 1.00x leverage invariant.

Leg ordering
------------
Multi-leg intents are emitted **hardest leg first**: the leg with the least
available depth is the one most likely to fail, and discovering that after
filling the easy legs leaves an unhedged position. Failing on the first leg costs
nothing.
"""

from __future__ import annotations

from typing import Any

from kat.arbitrage.logical import (
    LadderRung,
    find_complementary_pair,
    find_ladder_violation,
    find_mutually_exclusive_set,
)
from kat.arbitrage.pricing import PricedOpportunity
from kat.common.enums import MarketType
from kat.common.ids import new_id, use_correlation_id
from kat.common.timeutil import Clock
from kat.db.models import ArbitrageOpportunity as ArbRow
from kat.db.writer import AsyncWriter
from kat.kalshi.fees import FeeSchedule
from kat.marketdata.book import OrderBook
from kat.marketdata.state import MarketState
from kat.observability.logging import get_logger
from kat.risk.types import OrderIntent

log = get_logger(__name__)

__all__ = ["ArbitrageScanner", "opportunity_to_intents"]


class ArbitrageScanner:
    """Scans one domain's market state for logical pricing inconsistencies."""

    def __init__(
        self,
        *,
        market_state: MarketState,
        clock: Clock,
        writer: AsyncWriter | None = None,
        default_size_qc: int = 500,
        record_declined: bool = True,
    ) -> None:
        self.market_state = market_state
        self.clock = clock
        self.writer = writer
        self.default_size_qc = default_size_qc
        self.record_declined = record_declined

        #: Event membership, supplied by the ingestion layer: which markets
        #: belong to which event, and whether that event is genuinely mutually
        #: exclusive and exhaustive.
        self.event_members: dict[str, list[str]] = {}
        self.event_flags: dict[str, tuple[bool, bool]] = {}
        #: Threshold ladders keyed by series, as (ticker, threshold) pairs.
        self.ladders: dict[str, list[tuple[str, float]]] = {}
        self.fee_schedules: dict[str, FeeSchedule] = {}

        self.scanned = 0
        self.found = 0
        self.declined = 0

    # -- scanning ------------------------------------------------------------

    def scan(self) -> list[PricedOpportunity]:
        """One full pass. Returns only the opportunities worth acting on."""
        actionable: list[PricedOpportunity] = []
        now_ms = self.clock.now_ms()

        for candidate in self._candidates(now_ms):
            self.scanned += 1
            self._record(candidate)
            if candidate.is_actionable:
                self.found += 1
                actionable.append(candidate)
                log.info(
                    "arbitrage.found",
                    kind=candidate.kind,
                    expected_value_usd=str(candidate.expected_value_micros / 1_000_000),
                    net_edge_usd=str(candidate.net_edge_micros / 1_000_000),
                    execution_probability=str(candidate.execution_probability),
                    legs=len(candidate.legs),
                )
            else:
                self.declined += 1
        return actionable

    def _candidates(self, now_ms: int):
        """Yield every priced candidate, actionable or not."""
        # 1. Complementary pairs, one per market with a live two-sided book.
        for ticker in self.market_state.tickers():
            if not self.market_state.is_fresh(ticker, now_ms=now_ms):
                continue
            book = self.market_state.book(ticker)
            if book is None or book.is_crossed:
                continue
            candidate = find_complementary_pair(
                ticker,
                book,
                qty_qc=self.default_size_qc,
                fee_schedule=self.fee_schedules.get(ticker),
            )
            if candidate is not None:
                yield candidate

        # 2. Mutually exclusive sets, only where the exchange says the event is
        #    genuinely exclusive and exhaustive.
        for event_ticker, members in self.event_members.items():
            exclusive, exhaustive = self.event_flags.get(event_ticker, (False, False))
            if not (exclusive and exhaustive):
                continue
            books: dict[str, OrderBook] = {}
            for ticker in members:
                if not self.market_state.is_fresh(ticker, now_ms=now_ms):
                    books = {}
                    break
                book = self.market_state.book(ticker)
                if book is None or book.is_crossed:
                    books = {}
                    break
                books[ticker] = book
            if len(books) < 2:
                continue
            candidate = find_mutually_exclusive_set(
                event_ticker,
                books,
                qty_qc=self.default_size_qc,
                fee_schedules=self.fee_schedules,
                is_mutually_exclusive=exclusive,
                is_exhaustive=exhaustive,
            )
            if candidate is not None:
                yield candidate

        # 3. Threshold ladders.
        for series, rung_spec in self.ladders.items():
            rungs: list[LadderRung] = []
            for ticker, threshold in rung_spec:
                if not self.market_state.is_fresh(ticker, now_ms=now_ms):
                    rungs = []
                    break
                book = self.market_state.book(ticker)
                if book is None or book.is_crossed:
                    continue
                rungs.append(
                    LadderRung(
                        ticker=ticker,
                        threshold=threshold,
                        book=book,
                        fee_schedule=self.fee_schedules.get(ticker),
                    )
                )
            if len(rungs) < 2:
                continue
            candidate = find_ladder_violation(rungs, qty_qc=self.default_size_qc)
            if candidate is not None:
                candidate.detail["series"] = series
                yield candidate

    # -- persistence ---------------------------------------------------------

    def _worth_recording(self, candidate: PricedOpportunity) -> bool:
        """Whether this candidate is evidence of anything.

        ``record_declined`` is for the near misses -- the cases where an
        arbitrage genuinely existed and costs, size or leg risk took it away.
        Those are worth keeping: they say what the platform would need to be
        cheaper or faster to capture.

        A candidate with no gross edge is not a near miss. It is two prices that
        do not cross, which is what prices normally do, and recording it says
        only that the scanner ran. That distinction was missing and the cost was
        not subtle: 901,781 rows accumulated in seven hours, 2,086 a minute,
        every one of them ``actioned=False`` with the reason "no gross edge" and
        a negative edge, re-recording the same 1,366 leg sets about 660 times
        each. Nothing could be found in the table afterwards, which is the point
        of having it, and the write path carried three million rows a day to say
        that the market was behaving normally.

        The counters (``scanned``, ``found``, ``declined``) already report that
        the scanner ran, so nothing observable is lost by keeping quiet here.
        """
        if candidate.is_actionable:
            return True
        # A real cross that costs killed: this is the near miss worth studying.
        return self.record_declined and candidate.gross_edge_micros > 0

    def _record(self, candidate: PricedOpportunity) -> None:
        if self.writer is None:
            return
        if not self._worth_recording(candidate):
            return
        self.writer.enqueue(
            ArbRow(
                opportunity_id=str(candidate.detail.get("opportunity_id") or new_id("arb")),
                kind=candidate.kind,
                market_type=self.market_state.market_type.value,
                legs=[
                    {
                        "ticker": leg.ticker,
                        "side": leg.side.value,
                        "price_cc": leg.price_cc,
                        "qty_qc": leg.qty_qc,
                    }
                    for leg in candidate.legs
                ],
                gross_edge_micros=candidate.gross_edge_micros,
                fees_micros=candidate.fees_micros,
                spread_cost_micros=candidate.slippage_micros,
                slippage_micros=candidate.slippage_micros,
                net_edge_micros=candidate.net_edge_micros,
                execution_probability=float(candidate.execution_probability),
                residual_exposure_micros=candidate.residual_exposure_micros,
                worst_case_micros=candidate.worst_case_micros,
                expected_value_micros=candidate.expected_value_micros,
                actioned=candidate.is_actionable,
                reason=candidate.reason or None,
            )
        )

    def stats(self) -> dict[str, Any]:
        return {
            "domain": self.market_state.market_type.value,
            "scanned": self.scanned,
            "found": self.found,
            "declined": self.declined,
            "decline_rate": round(self.declined / self.scanned, 4) if self.scanned else None,
        }


def opportunity_to_intents(
    candidate: PricedOpportunity, *, strategy_id: str = "arbitrage"
) -> list[OrderIntent]:
    """Turn a priced opportunity into intents for the normal pipeline.

    Ordered hardest leg first, so a leg that cannot fill fails before any
    capital has been committed to the others. All legs share one correlation id
    so the audit trail shows them as a single attempt.
    """
    if not candidate.is_actionable:
        return []

    correlation_id = new_id("cor")
    ordered = sorted(candidate.legs, key=lambda leg: leg.available_qc)

    intents: list[OrderIntent] = []
    with use_correlation_id(correlation_id):
        for index, leg in enumerate(ordered):
            intents.append(
                OrderIntent(
                    market_type=MarketType.EVENT,
                    ticker=leg.ticker,
                    qty_qc=leg.qty_qc,
                    limit_price_cc=leg.price_cc,
                    outcome_side=leg.side,
                    strategy_id=strategy_id,
                    strategy_version="1.0.0",
                    correlation_id=correlation_id,
                    # The whole set is priced together, so the expected value is
                    # attributed to the first leg and the rest carry zero rather
                    # than being counted several times over.
                    expected_value_micros=(
                        candidate.expected_value_micros if index == 0 else 0
                    ),
                    confidence=float(candidate.execution_probability),
                    metadata={
                        "arbitrage_kind": candidate.kind,
                        "leg_index": index,
                        "leg_count": len(ordered),
                        "hardest_leg_first": True,
                        "opportunity_id": candidate.detail.get("opportunity_id"),
                    },
                )
            )
    return intents
