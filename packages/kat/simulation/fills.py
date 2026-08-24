"""Fill simulation: the part of a backtest that is most often wrong.

Three assumptions separate an honest fill simulator from a generator of
imaginary profit, and all three are implemented here rather than configured
away:

**1. A resting order is not a filled order.**
A limit order that is not marketable when it arrives goes onto the book and
fills only when the market later trades at or through its price -- and even
then, only after the size that was already resting there. We assume we are
behind *all* of it (:attr:`~kat.simulation.costs.CostModel.queue_ahead_bps`),
because this platform is a small participant that is never the first quote at a
price. A simulator that fills a resting bid the moment the mid touches it is
measuring a strategy nobody can trade.

**2. The book on arrival is not the book at the decision.**
Every order carries a latency between the moment the strategy decided and the
moment the exchange sees it. During that interval the simulated clock advances
and the book keeps updating from the feed. The order is then matched against
whatever is there *on arrival*. This is the whole difference between a naive
and an honest backtest: a strategy whose edge is a stale quote will fill at the
new price here, and its measured edge will evaporate, which is the correct
result.

**3. Depth is finite.**
Marketable orders walk the real book level by level, stopping at the limit price
and at the configured level cap. When the book is too thin the order fills
partially and the remainder rests. Nothing is ever filled at a price that was
not on the book.

Event-contract direction
------------------------
Everything in this module is **YES-denominated**, matching
:mod:`kat.marketdata.book`. A NO buy is a YES sell at ``$1.00 - no_price`` and
is converted exactly once, in :func:`yes_denominated_order`. Fees are charged on
the price of the contract actually traded (the NO price for a NO fill), which
matters for the flat fee schedule even though the quadratic formula happens to
be symmetric about $0.50.

The one order path
------------------
:meth:`FillSimulator.submit` takes a :class:`~kat.risk.types.RiskVerdict` and
nothing else, exactly like :meth:`kat.execution.base.ExecutionEngine.submit`. A
strategy cannot produce a verdict, so the simulator inherits the same structural
guarantee as the live path: orders exist only downstream of a risk engine.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kat.common.enums import MarketType, OrderState, OutcomeSide
from kat.common.errors import RiskRejection
from kat.common.fixedpoint import CC_PER_DOLLAR, QC_PER_CONTRACT
from kat.common.ids import new_id
from kat.marketdata.book import OrderBook
from kat.risk.types import RiskVerdict
from kat.simulation.costs import CostModel, CostReport

__all__ = [
    "FillSimulator",
    "SimFill",
    "SimOrder",
    "yes_denominated_order",
]


def yes_denominated_order(verdict: RiskVerdict) -> tuple[bool, int]:
    """Convert an approved intent into ``(is_buy, limit_price_cc)`` on the YES book.

    Perpetual intents already speak the book's language. Event-contract intents
    speak in outcome sides, and a NO buy at ``p`` is a YES sell at ``1 - p``.
    Doing this conversion anywhere else is how a strategy ends up short when it
    thinks it is long.
    """
    intent = verdict.intent
    if intent.market_type is MarketType.PERP:
        return intent.is_buy, intent.limit_price_cc
    if intent.outcome_side is OutcomeSide.NO:
        return False, CC_PER_DOLLAR - intent.limit_price_cc
    return True, intent.limit_price_cc


@dataclass(slots=True)
class SimFill:
    """One execution. All prices and notionals are YES-denominated.

    ``notional_micros`` is the exact integer sum of ``price * size`` across the
    levels consumed, so the ledger never has to divide. ``price_cc`` is that
    total expressed as an average price, rounded against us, and exists for
    reporting only.
    """

    order_id: str
    ticker: str
    market_type: MarketType
    ts_ms: int
    is_buy: bool
    qty_qc: int
    price_cc: int
    notional_micros: int
    fee_micros: int
    is_taker: bool
    #: Price of the contract actually traded (NO price for a NO fill). This is
    #: what the fee was computed against.
    fee_price_cc: int
    slippage_cc: int
    latency_ms: int
    decision_ms: int
    strategy_id: str | None = None
    outcome_side: OutcomeSide | None = None
    decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "ticker": self.ticker,
            "ts_ms": self.ts_ms,
            "is_buy": self.is_buy,
            "qty_qc": self.qty_qc,
            "price_cc": self.price_cc,
            "notional_micros": self.notional_micros,
            "fee_micros": self.fee_micros,
            "is_taker": self.is_taker,
            "slippage_cc": self.slippage_cc,
            "latency_ms": self.latency_ms,
            "strategy_id": self.strategy_id,
            "outcome_side": self.outcome_side.value if self.outcome_side else None,
        }


@dataclass(slots=True)
class SimOrder:
    """A simulated order and everything needed to decide its fate."""

    order_id: str
    ticker: str
    market_type: MarketType
    is_buy: bool
    qty_qc: int
    limit_price_cc: int
    decision_ms: int
    arrival_ms: int
    state: OrderState = OrderState.SUBMITTED
    post_only: bool = False
    reduce_only: bool = False
    strategy_id: str | None = None
    decision_id: str | None = None
    outcome_side: OutcomeSide | None = None

    filled_qc: int = 0
    #: Size assumed to be ahead of us at our price, decremented by prints at it.
    queue_ahead_qc: int = 0
    #: Touch price on the side we intended to take, at the decision instant.
    reference_price_cc: int = 0
    resting_since_ms: int | None = None

    @property
    def remaining_qc(self) -> int:
        return max(0, self.qty_qc - self.filled_qc)

    @property
    def is_working(self) -> bool:
        return self.state.is_working

    @property
    def contract_price_cc(self) -> int:
        """Price of the contract this order trades, for exposure and fees."""
        if self.market_type is MarketType.EVENT and not self.is_buy:
            return CC_PER_DOLLAR - self.limit_price_cc
        return self.limit_price_cc

    def pending_exposure_micros(self) -> int:
        """Capital an unfilled remainder could still commit."""
        return self.remaining_qc * self.contract_price_cc


class FillSimulator:
    """Matches simulated orders against a real order book over simulated time.

    The simulator owns no clock. The engine advances time and calls in; that
    keeps every fill decision a pure function of (order, book, trade prints,
    now), which is what makes a backtest reproducible.
    """

    def __init__(
        self,
        *,
        costs: CostModel,
        seed: int = 0,
        consume_book: bool = True,
    ) -> None:
        self.costs = costs
        self.report = CostReport()
        # A dedicated RNG so latency draws are reproducible and independent of
        # any other random stream in the run.
        self._rng = random.Random(seed)
        # Our taker fills remove the depth they consume from the simulated book.
        # Without this a strategy can take the same 10 contracts repeatedly
        # within one instant and never move the market it is trading.
        self._consume_book = consume_book

        self._pending: list[SimOrder] = []      # accepted, not yet arrived
        self._working: dict[str, SimOrder] = {}  # arrived and resting
        self.fills: list[SimFill] = []
        self.orders: dict[str, SimOrder] = {}
        self.sub_contract_rejections = 0

    # -- submission ----------------------------------------------------------

    def submit(
        self,
        verdict: RiskVerdict,
        *,
        now_ms: int,
        reference_price_cc: int | None = None,
    ) -> SimOrder | None:
        """Accept an order a risk engine has authorised.

        Returns ``None`` when the approved size rounds down to less than one
        whole contract, which is not an expressible order on Kalshi and is
        counted rather than silently rounded up.
        """
        if not verdict.approved:
            raise RiskRejection(
                verdict.reason_code,
                f"fill simulator refused an unapproved verdict: {verdict.reason}",
            )
        if verdict.approved_qty_qc <= 0:
            raise RiskRejection("ZERO_QUANTITY", "approved quantity is zero")

        # Kalshi trades whole contracts; a sub-contract remainder is dropped.
        qty_qc = (verdict.approved_qty_qc // QC_PER_CONTRACT) * QC_PER_CONTRACT
        if qty_qc <= 0:
            self.sub_contract_rejections += 1
            return None

        intent = verdict.intent
        is_buy, limit_cc = yes_denominated_order(verdict)
        latency_ms = self.costs.latency.sample_arrival_delay_ms(self._rng)
        self.report.record_latency(latency_ms)

        order = SimOrder(
            order_id=new_id("sim"),
            ticker=intent.ticker,
            market_type=intent.market_type,
            is_buy=is_buy,
            qty_qc=qty_qc,
            limit_price_cc=limit_cc,
            decision_ms=now_ms,
            arrival_ms=now_ms + latency_ms,
            post_only=intent.post_only,
            reduce_only=intent.reduce_only,
            strategy_id=intent.strategy_id,
            decision_id=intent.decision_id,
            outcome_side=intent.outcome_side,
            reference_price_cc=reference_price_cc if reference_price_cc is not None else limit_cc,
        )
        self._pending.append(order)
        self.orders[order.order_id] = order
        return order

    # -- time -----------------------------------------------------------------

    def step(
        self, *, now_ms: int, book_for: Callable[[str], OrderBook | None]
    ) -> list[SimFill]:
        """Advance the simulator to ``now_ms``.

        Arriving orders are matched against the book *as it is now*, not as it
        was when the decision was made. Resting orders are aged out if a TTL is
        configured; they are never filled here, only by :meth:`on_trade`.
        """
        produced: list[SimFill] = []

        if self._pending:
            still_pending: list[SimOrder] = []
            for order in self._pending:
                if order.arrival_ms > now_ms:
                    still_pending.append(order)
                    continue
                produced.extend(self._on_arrival(order, book_for(order.ticker), now_ms))
            self._pending = still_pending

        ttl = self.costs.order_ttl_ms
        if ttl is not None and self._working:
            for order in list(self._working.values()):
                since = order.resting_since_ms
                if since is not None and now_ms - since >= ttl:
                    self._retire(order, OrderState.EXPIRED)
                    self.report.expired_orders += 1

        return produced

    def _on_arrival(
        self, order: SimOrder, book: OrderBook | None, now_ms: int
    ) -> list[SimFill]:
        if book is None or book.is_crossed:
            # No usable book on arrival. The order rests with no knowledge of the
            # queue in front of it; it can still only fill on a later print.
            order.state = OrderState.ACKNOWLEDGED
            order.resting_since_ms = now_ms
            self._working[order.order_id] = order
            return []

        marketable = self._is_marketable(order, book)
        if marketable and order.post_only:
            # A maker-only order that would cross is refused by the exchange
            # rather than repriced. Counting it is how a quoting strategy learns
            # its quotes are systematically too aggressive.
            self._retire(order, OrderState.CANCELLED)
            self.report.rejected_post_only += 1
            return []

        fills: list[SimFill] = []
        if marketable:
            fill = self._take_liquidity(order, book, now_ms)
            if fill is not None:
                fills.append(fill)

        if order.remaining_qc > 0:
            order.state = (
                OrderState.PARTIALLY_FILLED if order.filled_qc else OrderState.ACKNOWLEDGED
            )
            order.resting_since_ms = now_ms
            resting = (book.bids if order.is_buy else book.asks).get(order.limit_price_cc, 0)
            order.queue_ahead_qc = self.costs.queue_ahead_qc(resting)
            self._working[order.order_id] = order
        else:
            self._retire(order, OrderState.FILLED)
        return fills

    @staticmethod
    def _is_marketable(order: SimOrder, book: OrderBook) -> bool:
        if order.is_buy:
            best_ask = book.best_ask_cc
            return best_ask is not None and best_ask <= order.limit_price_cc
        best_bid = book.best_bid_cc
        return best_bid is not None and best_bid >= order.limit_price_cc

    # -- taking liquidity -----------------------------------------------------

    def _take_liquidity(
        self, order: SimOrder, book: OrderBook, now_ms: int
    ) -> SimFill | None:
        """Walk the book within the limit price and the configured level cap."""
        prices = book.ask_prices if order.is_buy else book.bid_prices
        side = book.asks if order.is_buy else book.bids

        remaining = order.remaining_qc
        total_micros = 0
        filled = 0
        consumed: list[tuple[int, int]] = []

        for price in prices[: self.costs.max_walk_levels]:
            if order.is_buy and price > order.limit_price_cc:
                break
            if not order.is_buy and price < order.limit_price_cc:
                break
            available = side[price]
            take = min(remaining, available)
            if take <= 0:
                continue
            total_micros += price * take
            consumed.append((price, take))
            filled += take
            remaining -= take
            if remaining == 0:
                break

        if filled <= 0:
            return None

        if self._consume_book:
            for price, take in consumed:
                book.apply_delta(price, -take, is_bid=not order.is_buy)

        # Average price, rounded against us in both directions.
        vwap = -(-total_micros // filled) if order.is_buy else total_micros // filled
        vwap = self.costs.apply_taker_concession_cc(vwap, is_buy=order.is_buy)
        if self.costs.taker_concession_cc > 0:
            # Keep the notional consistent with the concession that was applied.
            total_micros = vwap * filled

        slippage = (
            vwap - order.reference_price_cc if order.is_buy else order.reference_price_cc - vwap
        )
        return self._record_fill(
            order,
            now_ms=now_ms,
            qty_qc=filled,
            price_cc=vwap,
            notional_micros=total_micros,
            is_taker=True,
            slippage_cc=slippage,
        )

    # -- providing liquidity --------------------------------------------------

    def on_trade(
        self, *, ticker: str, price_cc: int, qty_qc: int, now_ms: int
    ) -> list[SimFill]:
        """Apply a market print to our resting orders.

        A print *through* our price means everything at our price was taken, so
        price priority fills us regardless of queue. A print *at* our price
        consumes the queue in front of us first, and only what is left reaches
        our order. That asymmetry is the whole of queue modelling and it is why
        a strategy that quotes at the touch fills far less often than a naive
        simulator suggests.
        """
        if qty_qc <= 0:
            return []
        produced: list[SimFill] = []
        remaining_print = qty_qc

        for order in list(self._working.values()):
            if order.ticker != ticker or order.arrival_ms > now_ms or remaining_print <= 0:
                continue

            if order.is_buy:
                if price_cc > order.limit_price_cc:
                    continue
                through = price_cc < order.limit_price_cc
            else:
                if price_cc < order.limit_price_cc:
                    continue
                through = price_cc > order.limit_price_cc

            if not through:
                consumed = min(order.queue_ahead_qc, remaining_print)
                order.queue_ahead_qc -= consumed
                remaining_print -= consumed
                if remaining_print <= 0:
                    continue

            take = min(order.remaining_qc, remaining_print)
            if take <= 0:
                continue
            remaining_print -= take

            fill = self._record_fill(
                order,
                now_ms=now_ms,
                qty_qc=take,
                price_cc=order.limit_price_cc,
                notional_micros=order.limit_price_cc * take,
                is_taker=False,
                # A maker fill got exactly the price it asked for. The cost of
                # resting shows up as the fills that never happened, not here.
                slippage_cc=0,
            )
            produced.append(fill)
            if order.remaining_qc == 0:
                self._retire(order, OrderState.FILLED)
            else:
                order.state = OrderState.PARTIALLY_FILLED

        return produced

    # -- bookkeeping ----------------------------------------------------------

    def _record_fill(
        self,
        order: SimOrder,
        *,
        now_ms: int,
        qty_qc: int,
        price_cc: int,
        notional_micros: int,
        is_taker: bool,
        slippage_cc: int,
    ) -> SimFill:
        fee_price = (
            CC_PER_DOLLAR - price_cc
            if order.market_type is MarketType.EVENT and not order.is_buy
            else price_cc
        )
        fee_micros = self.costs.fee_micros(
            price_cc=fee_price, qty_qc=qty_qc, is_taker=is_taker
        )
        order.filled_qc += qty_qc

        fill = SimFill(
            order_id=order.order_id,
            ticker=order.ticker,
            market_type=order.market_type,
            ts_ms=now_ms,
            is_buy=order.is_buy,
            qty_qc=qty_qc,
            price_cc=price_cc,
            notional_micros=notional_micros,
            fee_micros=fee_micros,
            is_taker=is_taker,
            fee_price_cc=fee_price,
            slippage_cc=slippage_cc,
            latency_ms=order.arrival_ms - order.decision_ms,
            decision_ms=order.decision_ms,
            strategy_id=order.strategy_id,
            outcome_side=order.outcome_side,
            decision_id=order.decision_id,
        )
        self.fills.append(fill)
        self.report.record_fill(
            qty_qc=qty_qc, fee_micros=fee_micros, is_taker=is_taker, slippage_cc=slippage_cc
        )
        return fill

    def _retire(self, order: SimOrder, state: OrderState) -> None:
        order.state = state
        order.queue_ahead_qc = 0
        self._working.pop(order.order_id, None)

    # -- queries --------------------------------------------------------------

    def working_orders(self, ticker: str | None = None) -> list[SimOrder]:
        """Orders that could still fill, including those still in flight."""
        out = list(self._working.values())
        out.extend(self._pending)
        if ticker is not None:
            out = [o for o in out if o.ticker == ticker]
        return out

    def pending_exposure_micros(self, ticker: str) -> int:
        return sum(o.pending_exposure_micros() for o in self.working_orders(ticker))

    def pending_qty_qc(self, ticker: str) -> tuple[int, int]:
        """``(buy_qc, sell_qc)`` still working on this market, for perp risk."""
        buy = sell = 0
        for order in self.working_orders(ticker):
            if order.is_buy:
                buy += order.remaining_qc
            else:
                sell += order.remaining_qc
        return buy, sell

    def cancel(self, order_id: str) -> bool:
        order = self._working.get(order_id)
        if order is None:
            return False
        self._retire(order, OrderState.CANCELLED)
        return True

    def cancel_all(self, ticker: str | None = None) -> int:
        cancelled = 0
        for order in list(self._working.values()):
            if ticker is None or order.ticker == ticker:
                self._retire(order, OrderState.CANCELLED)
                cancelled += 1
        if ticker is None:
            cancelled += len(self._pending)
            self._pending = []
        else:
            keep = [o for o in self._pending if o.ticker != ticker]
            cancelled += len(self._pending) - len(keep)
            self._pending = keep
        return cancelled

    def finalise(self) -> None:
        """Count what never filled. Called once at the end of a run.

        Unfilled orders are a headline number, not a footnote: a strategy whose
        backtest depends on orders that mostly did not fill has not been tested,
        it has been imagined.
        """
        for order in list(self._working.values()):
            self.report.unfilled_orders += 1
            self._retire(order, OrderState.CANCELLED)
        for order in self._pending:
            self.report.unfilled_orders += 1
            order.state = OrderState.CANCELLED
        self._pending = []

    def stats(self) -> dict[str, Any]:
        data = self.report.to_dict()
        data["orders"] = len(self.orders)
        data["sub_contract_rejections"] = self.sub_contract_rejections
        return data
